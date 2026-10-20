"""Adapters handing a driven optical lattice to the time-dependent potential classes of BECs.

Three routes are provided, all built on the expansion of the dynamics module:

- 'add_to_potentialT' with mode="shapes": one PotentialT shape per grating, each multiplied by its own
  time function. The terms compose with the rest of the potential and can be inspected one by one.
- 'add_to_potentialT' with mode="stacked": the gratings are kept stacked in one array and combined by a
  single function. Around five times faster per time step.
- 'add_to_analytic': a single function of time and coordinates, for the AnalyticPotential class and the
  rescaling solver, whose grid changes at every step. No grating is stored.

Every route keeps every parameter dimension of the lattice: those of the drive, of the beams (a
polarization angle, a wavelength, a waist, ...) and of the polarizabilities. The shapes route simply
stores them in the shapes. The two other routes register each of them with the potential under the key
'_dim_<name>' (see the parameters module), so that the solvers hand them the label of the point they
run, and reduce the lattice to that point themselves, once per run.

Nothing here imports the BECs package: the adapters only use the public methods of the potential
object they are given.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from optilat.dynamics import Expansion, LatticeDrive, expand
from optilat.parameters import (
    DIM_PREFIX,
    array_module,
    collect_dims,
    dim_parameters,
    dim_selection,
    is_labelled,
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
    """Raise if a drive does not describe the beams of a lattice, or uses a reserved name."""
    if len(drive) != len(lattice.beams):
        raise ValueError(
            f"The drive describes {len(drive)} beams but the lattice has {len(lattice.beams)}."
        )
    reserved = [name for name in drive.parameters if name.startswith(DIM_PREFIX)]
    if reserved:
        raise ValueError(
            f"The drive parameter name(s) {reserved} start with the reserved prefix "
            f"'{DIM_PREFIX}'. Rename them."
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
    spatial dimension in the order x, y, z (the missing axes sit at 0), and the parameters as
    keywords. Register it with the parameters it needs, 'self.parameters', or use 'add_to_analytic'
    which does exactly that.

    The function is evaluated in one of two ways, chosen at every call:

    - with a plain time and plain parameters, as the solvers call it, the lattice is reduced to the
      selected point of its parameter space and the beams are summed in the array module of the
      coordinates, numpy or cupy. The reduced lattice is cached, so only the first call of a run
      pays for the selection.
    - with a DataArray time or parameters, as AnalyticPotential.to_potential and plot_t call it,
      the result keeps every dimension. It is then built from the static gratings of 'expand', as
      in the PotentialT route, which costs far less memory than summing the complex fields.
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
        self.dims = merge_dims(lattice.param_coords, collect_dims(*self.stark.values()))
        # A dimension shared with the drive must have the same coordinate there, or the potential
        # would silently keep only one of the two
        for value in self.drive.parameters.values():
            if isinstance(value, xr.DataArray):
                shared = [dim for dim in value.dims if dim in self.dims and dim in value.coords]
                merge_dims(self.dims, collect_dims(*(value.coords[dim] for dim in shared)))

        # Everything the potential has to hand over: the drive parameters, under their own names,
        # and the label of every parameter dimension, under '_dim_<name>'
        self.parameters = {**self.drive.parameters, **dim_parameters(self.dims)}

        self._numeric_cache: dict[tuple, _NumericLattice] = {}
        self._coords_cache: tuple | None = None

    def __repr__(self) -> str:
        return (
            f"LatticeFunction over {len(self.lattice.beams)} beams, parameter dimensions "
            f"{list(self.dims)}, drive parameters {list(self.drive.parameters)}"
        )

    def __call__(self, t, *coords, **params):
        """The Stark potential of the driven lattice at time t, over the coordinates given.

        Args:
            t (Union[float,xr.DataArray]): The time, or a whole time coordinate.
            *coords: One coordinate per spatial dimension, in the order x, y, z. Plain numpy or cupy
            arrays, broadcastable against each other, or DataArrays.
            **params: The drive parameters, and the label of every parameter dimension under
            '_dim_<name>', as listed in 'self.parameters'.

        Returns:
            The potential: an array of the coordinates' module for plain coordinates, a DataArray
            over their dimensions for labelled ones.
        """
        if is_labelled(t, params):
            return self._labelled(t, coords, params)
        return self._fast(t, coords, params)

    def _drive_params(self, params: dict) -> dict:
        """The entries of a parameter dictionary that belong to the drive."""
        return {name: params[name] for name in self.drive.parameters if name in params}

    def _numeric(self, params: dict) -> _NumericLattice:
        """The lattice at the point of its parameter space selected by 'params', in plain numbers.

        Raises:
            ValueError: If the label of a parameter dimension is missing from 'params'.
        """
        selection = dim_selection(params, self.dims)
        missing = [dim for dim in self.dims if dim not in selection]
        if missing:
            raise ValueError(
                f"No value was given for the parameter dimension(s) {missing}. Pass them as "
                f"'{DIM_PREFIX}<name>' keywords, or register 'self.parameters' with the potential."
            )

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

    def _fast(self, t, coords: tuple, params: dict):
        """Sum the beams at a single point of the parameter space, as the solvers need it."""
        lat = self._numeric(params)
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
        """Evaluate the potential over whole dimensions, through the static gratings.

        Raises:
            ValueError: If a coordinate is a bare array, whose dimensions cannot be told apart.
        """
        if any(not isinstance(c, xr.DataArray) and np.ndim(c) > 0 for c in coords):
            raise ValueError(
                "A labelled evaluation (DataArray time or parameters) needs DataArray coordinates, "
                "such as the potential's own 'coords', or single values."
            )

        # A parameter dimension handed over as a single value still selects the lattice
        selection = dim_selection(params, self.dims)
        lattice = self.lattice.sel(selection) if selection else self.lattice
        stark = {name: select(value, selection) for name, value in self.stark.items()}

        expansion = expand(
            lattice, self.drive, **axes_from_coords(coords), **stark, rtol=self.rtol
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
    evaluated on whichever coordinates the solver hands over, which is what the rescaling solver
    needs, since its grid changes at every time step. The parameter dimensions of the beams, of the
    polarizabilities and of the drive all become parameter dimensions of the potential, swept by the
    solvers, and 'to_potential' and 'plot_t' keep them as dimensions of the result.

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


def _stacked_sum(
    expansion: Expansion, zero: xr.DataArray, spatial: tuple[str, ...], pair: str
) -> tuple[Callable, dict[str, xr.DataArray]]:
    """A single function of time summing every grating of an expansion with its coefficient.

    The gratings are stacked along a 'pair' dimension, keeping whatever parameter dimensions they
    carry. A solver run selects its point of that stack once, and every time step is then a single
    tensor contraction.

    Args:
        expansion (Expansion): The expansion to sum.
        zero (xr.DataArray): Zeros over the spatial grid, which every grating is broadcast onto.
        spatial (tuple[str, ...]): The spatial dimensions of the potential.
        pair (str): The name of the stacking dimension.

    Returns:
        tuple[Callable, dict[str, xr.DataArray]]: The function f(t, **parameters), returning the
        potential over the grid, and the parameters to register it with.
    """
    stack = xr.concat(
        [grating.shape + zero for grating in expansion.gratings], dim=pair, coords="minimal"
    )
    dims = collect_dims(stack, skip=(pair, *spatial))
    stack = stack.transpose(pair, *spatial, *dims)
    grid_coords = dict(zero.coords)
    coeffs = [grating.coeff for grating in expansion.gratings]
    drive_names = set(expansion.parameters)
    cache: dict[tuple, np.ndarray] = {}

    def lattice_sum(t, **params):
        """The sum of every grating times its coefficient at time t."""
        drive_params = {name: value for name, value in params.items() if name in drive_names}
        values = [coeff(t, **drive_params) for coeff in coeffs]
        selection = dim_selection(params, dims)

        if is_labelled(t, params):
            # to_potential and plot_t hand over whole coordinates, and the coefficients then carry
            # dimensions of their own that have to stay labelled
            c = xr.concat([xr.DataArray(value) for value in values], dim=pair, coords="minimal")
            return xr.dot(select(stack, selection), c, dim=pair)

        key = tuple(selection.values())
        if key not in cache:
            if len(selection) != len(dims):
                missing = [dim for dim in dims if dim not in selection]
                raise ValueError(f"No value was given for the parameter dimension(s) {missing}.")
            cache[key] = np.ascontiguousarray(
                select(stack, selection).transpose(pair, *spatial).values
            )
        return xr.DataArray(
            np.tensordot(np.asarray(values, dtype=float), cache[key], axes=(0, 0)),
            dims=spatial,
            coords=grid_coords,
        )

    return lattice_sum, {**expansion.parameters, **dim_parameters(dims)}


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

        function, parameters = _stacked_sum(
            expansion, zero, tuple(potT.spatial_dims), f"_{name}_pair"
        )
        potT.add_function(fname, function, parameters=parameters)
        terms = [fname]

    potT.add_term(" + ".join(terms), name=name)
    return expansion
