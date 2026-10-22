"""Adapters handing a driven optical lattice to the time-dependent potential classes of BECs.

Three routes are provided, all built on the expansion of the dynamics module:

- 'add_to_potentialT' with mode="shapes": one PotentialT shape per grating, each multiplied by its own
  time function. The terms compose with the rest of the potential and can be inspected one by one.
- 'add_to_potentialT' with mode="stacked": the gratings are kept stacked in one array and combined by a
  single function. Around five times faster per time step.
- 'add_to_analytic': a single function of time and coordinates, for the AnalyticPotential class, which
  can be evaluated on a grid that changes at every step. No grating is stored.

Every route keeps every parameter dimension of the lattice: those of the drive, of the beams (a
polarization angle, a wavelength, a waist, ...) and of the polarizabilities. The shapes route simply
stores them in the shapes. The two other routes register a function that carries them itself: it lists
them in 'param_coords' and reduces itself to a point of them with 'sel', which the BECs potentials call
once per run with the label of the point a solver runs (see ParametricPotential.add_function there).

Nothing here imports the BECs package: the adapters only use the public methods of the potential
object they are given.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from optilat.dynamics import Expansion, LatticeDrive, expand
from optilat.parameters import (
    array_module,
    collect_dims,
    merge_dims,
    scalarize,
    select,
    to_number,
)
from optilat.potentials import axes_from_coords, cartesian_axes, is_zero


def _spatial_zero(pot) -> xr.DataArray:
    """A zero array over the spatial grid of a potential only, without its parameter dimensions.

    Adding it to a shape forces the shape onto the full grid, which PotentialT.add_shape requires,
    without dragging in whatever parameter dimensions the potential's own landscape may carry.

    Args:
        pot: A PotentialT-like object.

    Returns:
        xr.DataArray: Zeros over the spatial dimensions ('a1', 'a2', ...) of the potential.
    """
    coords = {dim: pot.V.coords[dim] for dim in pot.spatial_dims}
    return xr.DataArray(
        np.zeros([coords[dim].size for dim in coords], dtype=float), coords=coords
    )


def _check_free(name: str, taken, kind: str):
    """Raise if a generated name is already used by the potential."""
    if name in taken:
        raise ValueError(
            f"The potential already has a {kind} named '{name}'. Pass a different 'name' to "
            "avoid the collision."
        )


def _check_drive(lattice, drive: LatticeDrive):
    """Raise if a drive does not describe the beams of a lattice."""
    if len(drive) != len(lattice.beams):
        raise ValueError(
            f"The drive describes {len(drive)} beams but the lattice has {len(lattice.beams)}."
        )


def _numpy_beams(lattice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The k-vectors, complex amplitude vectors and beam frames of every beam, as plain numpy arrays.

    Args:
        lattice (OptiLat): The lattice to read, reduced to a single point of its parameter space.

    Raises:
        ValueError: If a beam still carries a parameter dimension, which cannot be reduced to a number.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The (n_beams, 3) k-vectors and complex
        amplitudes, and the (n_beams, 4, 3) beam frames: the focus, then the direction, TE and TM unit
        vectors, as used to evaluate the lateral profiles.
    """
    extra = list(lattice.param_coords)
    if extra:
        raise ValueError(
            f"The beams carry the parameter dimension(s) {extra}. Select them first, with "
            "OptiLat.sel."
        )

    k = np.array([[float(beam.k.isel(component=i)) for i in range(3)] for _, beam in lattice.beams])
    A = np.array(
        [[complex(beam.A.isel(component=i)) for i in range(3)] for _, beam in lattice.beams]
    )
    frames = np.array(
        [
            [
                [float(vec.isel(component=i)) for i in range(3)]
                for vec in (
                    beam.profile.focus if beam.profile is not None else xr.zeros_like(beam.k),
                    beam.direction,
                    beam.TE,
                    beam.TM,
                )
            ]
            for _, beam in lattice.beams
        ]
    )
    return k, A, frames


### ====================
### Analytic route
### ====================


@dataclass
class _NumericLattice:
    """A lattice reduced to a single point of its parameter space, in plain numbers."""

    k: np.ndarray  # (n_beams, 3) k-vectors
    kls: np.ndarray  # (n_beams,) wavenumbers
    A: np.ndarray  # (n_beams, 3) complex amplitude vectors
    frames: np.ndarray  # (n_beams, 4, 3) focus, direction, TE and TM of every beam
    layers: np.ndarray  # (n_beams,) coherent field index of every beam
    profiles: list  # the lateral profile of every beam, with plain-number inputs, or None
    alpha_s: float
    alpha_v: float
    M: float
    F: float

    @property
    def with_vector(self) -> bool:
        """Whether the polarization-dependent part of the potential is needed at all."""
        return not (is_zero(self.alpha_v) or is_zero(self.M))


class LatticeFunction:
    """The Stark potential of a driven optical lattice, as a function f(t, *coords, **parameters).

    That is the signature of the functions of an AnalyticPotential: the time, one coordinate per
    spatial dimension in the order x, y, z (the missing axes sit at 0), and the drive parameters as
    keywords. The parameter dimensions of the beams and of the polarizabilities are the function's
    own: it lists them in 'param_coords' and reduces itself to a point of them with 'sel', which is
    what an AnalyticPotential looks for in order to sweep them. Register it with the drive
    parameters, 'self.parameters', or use 'add_to_analytic' which does exactly that.

    The function is evaluated in one of two ways:

    - reduced, 'self.sel(selection)(t, *coords, **parameters)', as the solvers call it: the lattice
      is reduced to the selected point of its parameter space once, and every call sums the beams in
      the array module of the coordinates, numpy or cupy.
    - unreduced, 'self(t, *coords, **parameters)', as AnalyticPotential.to_potential and plot_t
      call it, with DataArray times or coordinates: the result keeps every dimension. It is then
      built from the static gratings of 'expand', as in the PotentialT route, which costs far less
      memory than summing the complex fields.
    """

    def __init__(
        self,
        lattice,
        drive: LatticeDrive | None = None,
        *,
        alpha_s: float | xr.DataArray = 1.0,
        alpha_v: float | xr.DataArray = 0.0,
        M: float | xr.DataArray = 0.0,
        F: float | xr.DataArray = 1.0,
        rtol: float = 1e-12,
    ):
        """Initialize the function of a lattice and its drive.

        Args:
            lattice (OptiLat): The lattice. Its beams may carry parameter dimensions.
            drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
            lattice.
            alpha_s (Union[float,xr.DataArray], optional): The scalar polarizability. Defaults to 1.
            alpha_v (Union[float,xr.DataArray], optional): The vector polarizability. Defaults to 0.
            M (Union[float,xr.DataArray], optional): The spin projection on z. Defaults to 0.
            F (Union[float,xr.DataArray], optional): The total angular momentum. Defaults to 1.
            rtol (float, optional): The relative threshold below which a grating is dropped, in the
            labelled evaluation only. Defaults to 1e-12.

        Raises:
            ValueError: If the drive does not match the lattice, if a parameter dimension has no
            coordinates, or if a dimension is given different coordinates by different inputs.
        """
        self.lattice = lattice
        self.drive = LatticeDrive(lattice) if drive is None else drive
        _check_drive(lattice, self.drive)
        self.stark = {"alpha_s": alpha_s, "alpha_v": alpha_v, "M": M, "F": F}
        self.rtol = rtol

        # The dimensions this function selects itself, on the beams and the polarizabilities
        self.param_coords = merge_dims(lattice.param_coords, collect_dims(*self.stark.values()))
        # A dimension shared with the drive must have the same coordinate there, or the potential
        # would silently keep only one of the two
        for value in self.drive.parameters.values():
            if isinstance(value, xr.DataArray):
                shared = [
                    dim for dim in value.dims if dim in self.param_coords and dim in value.coords
                ]
                merge_dims(self.param_coords, collect_dims(*(value.coords[dim] for dim in shared)))

        # What the potential has to hand over as keywords: the drive parameters. The dimensions of
        # the beams and polarizabilities reach the function through 'sel' instead.
        self.parameters = dict(self.drive.parameters)

        self._numeric_cache: dict[tuple, _NumericLattice] = {}
        self._coords_cache: tuple | None = None

    def __repr__(self) -> str:
        return (
            f"LatticeFunction over {len(self.lattice.beams)} beams, parameter dimensions "
            f"{list(self.param_coords)}, drive parameters {list(self.drive.parameters)}"
        )

    def sel(self, selection: dict) -> Callable:
        """The function at one point of its parameter space, as the solvers evaluate it.

        The lattice is reduced to that point here, once, and the function returned only sums the
        beams. The reduced lattice is also cached by point, so reducing to a point seen before
        costs nothing.

        Args:
            selection (dict): The label of each parameter dimension in 'param_coords'. Entries for
            other dimensions are ignored.

        Raises:
            ValueError: If the label of a parameter dimension is missing.

        Returns:
            Callable: f(t, *coords, **parameters), with the drive parameters as keywords. Its
            coordinates are plain numpy or cupy arrays broadcastable against each other, in the
            order x, y, z, or DataArrays; it returns an array of the coordinates' module, or a
            DataArray over their dimensions for labelled ones.
        """
        lat = self._numeric(selection)

        def at(t, *coords, **params):
            return self._fast(lat, t, coords, params)

        return at

    def __call__(self, t, *coords, **params):
        """The Stark potential of the driven lattice over whole dimensions, keeping every
        parameter dimension of the lattice.

        Plain arrays carry no dimension names to keep the result's dimensions apart by, so plain
        coordinates are only accepted for a lattice without parameter dimensions, which is then
        summed exactly as 'sel({})' would. Any other is reduced with 'sel' first.

        Args:
            t (Union[float,xr.DataArray]): The time, or a whole time coordinate.
            *coords: One coordinate per spatial dimension, in the order x, y, z: DataArrays, such as
            a potential's own 'coords', or single values.
            **params: The drive parameters, as listed in 'self.parameters'.

        Raises:
            ValueError: For plain coordinate arrays, if the lattice has parameter dimensions.

        Returns:
            xr.DataArray: The potential over the dimensions of its inputs and of the lattice.
        """
        if any(not isinstance(c, xr.DataArray) and np.ndim(c) > 0 for c in coords):
            if self.param_coords:
                raise ValueError(
                    "Plain coordinate arrays carry no dimension names, so the parameter "
                    f"dimension(s) {list(self.param_coords)} of the lattice cannot be kept apart "
                    "from them. Reduce the function to a point with 'sel' first, or pass "
                    "DataArray coordinates."
                )
            return self.sel({})(t, *coords, **params)
        return self._labelled(t, coords, params)

    def _drive_params(self, params: dict) -> dict:
        """The entries of a parameter dictionary that belong to the drive."""
        return {name: params[name] for name in self.drive.parameters if name in params}

    def _numeric(self, selection: dict) -> _NumericLattice:
        """The lattice at the point of its parameter space selected by 'selection', in plain numbers.

        Raises:
            ValueError: If the label of a parameter dimension is missing from 'selection'.
        """
        missing = [dim for dim in self.param_coords if dim not in selection]
        if missing:
            raise ValueError(
                f"No value was given for the parameter dimension(s) {missing}. Pass them to 'sel', "
                "or register the function with a potential, which does."
            )
        selection = {dim: to_number(selection[dim]) for dim in self.param_coords}

        key = tuple(selection.values())
        if key not in self._numeric_cache:
            lattice = self.lattice.sel(selection)
            k, A, frames = _numpy_beams(lattice)
            self._numeric_cache[key] = _NumericLattice(
                k=k,
                kls=np.linalg.norm(k, axis=1),
                A=A,
                frames=frames,
                layers=np.array([layer for layer, _ in lattice.beams]),
                profiles=[
                    None if beam.profile is None else scalarize(beam.profile)
                    for _, beam in lattice.beams
                ],
                **{
                    name: to_number(select(value, selection))
                    for name, value in self.stark.items()
                },
            )
        return self._numeric_cache[key]

    def _plain_coords(self, coords: tuple) -> tuple[list, tuple[str, ...] | None]:
        """The coordinates as plain arrays, and the dimensions to label the result with if they came
        as DataArrays.

        Labelled coordinates are broadcast against each other by dimension name first, which plain
        positional broadcasting would get wrong. The solvers hand over the very same coordinates at
        every step, so the last broadcast is kept and reused.
        """
        if not any(isinstance(c, xr.DataArray) for c in coords):
            return list(coords), None

        cached = self._coords_cache
        if (
            cached is not None
            and len(cached[0]) == len(coords)
            and all(a is b for a, b in zip(cached[0], coords))
        ):
            return cached[1], cached[2]

        labelled = xr.broadcast(*(xr.DataArray(c) for c in coords))
        arrays = [c.data for c in labelled]
        self._coords_cache = (coords, arrays, labelled[0].dims)
        return arrays, labelled[0].dims

    def _fast(self, lat: _NumericLattice, t, coords: tuple, params: dict):
        """Sum the beams of a lattice reduced to a single point, as the solvers need it."""
        drive_params = self._drive_params(params)
        arrays, labels = self._plain_coords(coords)
        xp = array_module(*arrays)

        shape = np.broadcast_shapes(*(np.shape(c) for c in arrays))
        V = xp.zeros(shape)

        for layer in np.unique(lat.layers):
            E = xp.zeros((3,) + shape, dtype=complex)
            for j in np.flatnonzero(lat.layers == layer):
                # k has 3 components, but only the axes the solver provides are resolved
                kdr = sum(lat.k[j, i] * c for i, c in enumerate(arrays))
                factor = self.drive[int(j)].factor(t, **drive_params) * xp.exp(1j * kdr)
                if lat.profiles[j] is not None:
                    # The axes the solver does not provide sit at 0, as in cartesian_axes
                    rel = [
                        (arrays[i] if i < len(arrays) else 0) - lat.frames[j, 0, i]
                        for i in range(3)
                    ]
                    s, t1, t2 = (
                        sum(lat.frames[j, a, i] * rel[i] for i in range(3)) for a in (1, 2, 3)
                    )
                    factor = factor * lat.profiles[j].envelope(s, t1, t2, lat.kls[j])
                for comp in range(3):
                    E[comp] = E[comp] + lat.A[j, comp] * factor

            V = V + lat.alpha_s * (abs(E) ** 2).sum(0)
            if lat.with_vector:
                V = V + lat.alpha_v * lat.M / (2 * lat.F) * 2 * (E[0].conj() * E[1]).imag

        V = -0.25 * V
        if labels is None:
            return V
        # Labelled by dimension only: enough to combine with the other terms of the potential by
        # name, and an order of magnitude cheaper per step than carrying the coordinates along
        return xr.DataArray(V, dims=labels)

    def _labelled(self, t, coords: tuple, params: dict) -> xr.DataArray:
        """Evaluate the potential over whole dimensions, through the static gratings."""
        expansion = expand(
            self.lattice, self.drive, **axes_from_coords(coords), **self.stark, rtol=self.rtol
        )
        drive_params = self._drive_params(params)
        V = 0
        for grating in expansion.gratings:
            V = V + grating.shape * grating.coeff(t, **drive_params)
        return V


def add_to_analytic(
    anpot,
    lattice,
    drive: LatticeDrive | None = None,
    name: str = "lat",
    term: bool = True,
    *,
    alpha_s: float | xr.DataArray = 1.0,
    alpha_v: float | xr.DataArray = 0.0,
    M: float | xr.DataArray = 0.0,
    F: float | xr.DataArray = 1.0,
) -> LatticeFunction:
    """Add a driven optical lattice to an AnalyticPotential object, as a function of time and space.

    Contrarily to the PotentialT route, nothing is precomputed on a grid: the beam superposition is
    evaluated on whichever coordinates it is handed, which is what a solver whose grid changes at
    every time step needs. The parameter dimensions of the beams, of the polarizabilities and of the
    drive all become parameter dimensions of the potential, swept by the solvers, and
    'to_potential' and 'plot_t' keep them as dimensions of the result.

    Args:
        anpot: An AnalyticPotential-like object, used through its add_function and add_term methods.
        lattice (OptiLat): The lattice to add.
        drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
        lattice.
        name (str, optional): The name of the generated function and term. Defaults to "lat".
        term (bool, optional): Whether to add the function to the potential as a term of its own.
        Leave it to False to only register the function, and use it in a term of your own, such as
        f"{name} * gate". Defaults to True.
        alpha_s (Union[float,xr.DataArray], optional): The scalar polarizability. Defaults to 1.
        alpha_v (Union[float,xr.DataArray], optional): The vector polarizability. Defaults to 0,
        which leaves out the polarization-dependent part entirely.
        M (Union[float,xr.DataArray], optional): The spin projection on z. Defaults to 0.
        F (Union[float,xr.DataArray], optional): The total angular momentum. Defaults to 1.

    Raises:
        ValueError: If the generated name is already taken, or if the inputs carry inconsistent
        parameter dimensions.

    Returns:
        LatticeFunction: The function that was registered, useful for inspection.
    """
    _check_free(name, anpot.funcs, "function")

    function = LatticeFunction(lattice, drive, alpha_s=alpha_s, alpha_v=alpha_v, M=M, F=F)
    anpot.add_function(name, function, parameters=function.parameters)
    if term:
        anpot.add_term(name, name=name)
    return function


### ====================
### PotentialT route
### ====================


class _StackedSum:
    """A single function of time summing every grating of an expansion with its coefficient.

    The gratings are stacked along a 'pair' dimension, keeping whatever parameter dimensions they
    carry. Those are the function's own, listed in 'param_coords' and reduced with 'sel': a solver
    run selects its point of the stack once, and every time step is then a single tensor
    contraction.
    """

    def __init__(
        self, expansion: Expansion, zero: xr.DataArray, spatial: tuple[str, ...], pair: str
    ):
        """
        Args:
            expansion (Expansion): The expansion to sum.
            zero (xr.DataArray): Zeros over the spatial grid, which every grating is broadcast onto.
            spatial (tuple[str, ...]): The spatial dimensions of the potential.
            pair (str): The name of the stacking dimension.
        """
        stack = xr.concat(
            [grating.shape + zero for grating in expansion.gratings], dim=pair, coords="minimal"
        )
        self.param_coords = collect_dims(stack, skip=(pair, *spatial))
        self.stack = stack.transpose(pair, *spatial, *self.param_coords)
        self.spatial = spatial
        self.pair = pair
        self.grid_coords = dict(zero.coords)
        self.coeffs = [grating.coeff for grating in expansion.gratings]
        self.drive_names = set(expansion.parameters)

    def _coefficients(self, t, params: dict) -> list:
        """The coefficient of every grating at time t."""
        drive_params = {name: value for name, value in params.items() if name in self.drive_names}
        return [coeff(t, **drive_params) for coeff in self.coeffs]

    def sel(self, selection: dict) -> Callable:
        """The sum at one point of its parameter space, as the solvers evaluate it.

        Raises:
            ValueError: If the label of a parameter dimension is missing.
        """
        missing = [dim for dim in self.param_coords if dim not in selection]
        if missing:
            raise ValueError(f"No value was given for the parameter dimension(s) {missing}.")
        point = {dim: selection[dim] for dim in self.param_coords}
        stack = np.ascontiguousarray(
            select(self.stack, point).transpose(self.pair, *self.spatial).values
        )

        def lattice_sum(t, **params):
            """The sum of every grating times its coefficient at time t."""
            values = np.asarray(self._coefficients(t, params), dtype=float)
            return xr.DataArray(
                np.tensordot(values, stack, axes=(0, 0)),
                dims=self.spatial,
                coords=self.grid_coords,
            )

        return lattice_sum

    def __call__(self, t, **params):
        """The sum over whole dimensions, as to_potential and plot_t call it: the coefficients then
        carry dimensions of their own that have to stay labelled."""
        c = xr.concat(
            [xr.DataArray(value) for value in self._coefficients(t, params)],
            dim=self.pair,
            coords="minimal",
        )
        return xr.dot(self.stack, c, dim=self.pair)


def add_to_potentialT(
    potT,
    lattice,
    drive: LatticeDrive | None = None,
    name: str = "lat",
    mode: str = "shapes",
    rtol: float = 1e-12,
    *,
    alpha_s: float | xr.DataArray = 1.0,
    alpha_v: float | xr.DataArray = 0.0,
    M: float | xr.DataArray = 0.0,
    F: float | xr.DataArray = 1.0,
) -> Expansion:
    """Add a driven optical lattice to a PotentialT object, as shapes and time functions.

    The lattice is evaluated on the potential's own cartesian grid and expanded into static gratings
    times pure time functions, so that 'make_Vt', 'to_potential' and 'plot_t' all work as usual. The
    terms are added alongside whatever the potential already holds: an existing trap in V0 or a shape
    added with 'circle_t' is left untouched.

    Args:
        potT: A PotentialT-like object, used through its add_shape, add_function and add_term
        methods only. Build it with v0=0 if the lattice is to be the whole potential.
        lattice (OptiLat): The lattice to add.
        drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
        lattice.
        name (str, optional): The prefix of every generated shape, function and term name. Defaults
        to "lat".
        mode (str, optional): "shapes" registers one shape and one time function per grating.
        "stacked" keeps the gratings in a single array summed by one function, which is several
        times faster per time step. Both keep every parameter dimension. Defaults to "shapes".
        rtol (float, optional): The relative threshold below which a grating is dropped. Defaults to
        1e-12.
        alpha_s (Union[float,xr.DataArray], optional): The scalar polarizability. Defaults to 1.
        alpha_v (Union[float,xr.DataArray], optional): The vector polarizability. Defaults to 0,
        which leaves out the polarization-dependent gratings entirely.
        M (Union[float,xr.DataArray], optional): The spin projection on z. Defaults to 0.
        F (Union[float,xr.DataArray], optional): The total angular momentum. Defaults to 1.

    Raises:
        ValueError: If 'mode' is unknown or if a generated name is already taken.

    Returns:
        Expansion: The expansion that was registered, useful for inspection.
    """
    if mode not in ("shapes", "stacked"):
        raise ValueError(f"'mode' must be 'shapes' or 'stacked', got {mode!r}")

    drive = LatticeDrive(lattice) if drive is None else drive
    _check_drive(lattice, drive)
    expansion = expand(
        lattice,
        drive,
        **cartesian_axes(potT),
        alpha_s=alpha_s,
        alpha_v=alpha_v,
        M=M,
        F=F,
        rtol=rtol,
    )
    zero = _spatial_zero(potT)

    if mode == "shapes":
        terms = []
        for grating in expansion.gratings:
            sname, fname = f"{name}_{grating.name}", f"{name}_c_{grating.name}"
            _check_free(sname, potT.shapes_t, "shape")
            _check_free(fname, potT.funcs, "time function")

            potT.add_shape(sname, grating.shape + zero)
            potT.add_function(fname, grating.coeff, parameters=expansion.parameters)
            terms.append(f"{sname} * {fname}")

    else:
        fname = f"{name}_sum"
        _check_free(fname, potT.funcs, "time function")

        function = _StackedSum(expansion, zero, tuple(potT.spatial_dims), f"_{name}_pair")
        potT.add_function(fname, function, parameters=expansion.parameters)
        terms = [fname]

    potT.add_term(" + ".join(terms), name=name)
    return expansion
