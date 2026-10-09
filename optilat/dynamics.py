"""Time-dependent optical lattices, for the time-dependent potential classes of the BECs package.

Give every beam of a lattice a complex time factor c_j(t) = s_j(t) exp(i phi_j(t)) -- an amplitude
ramp times a phase modulation -- and the field of a coherent layer L becomes

    E_L(r,t) = sum_{j in L} c_j(t) A_j u_j(r) exp(i k_j.r)

Squaring it splits the space and the time dependence apart exactly:

    |E_L(r,t)|^2 = sum_j |c_j(t)|^2 |A_j|^2
                 + 2 sum_{j<l} ( Re[c_j c_l*] Re G_jl(r) - Im[c_j c_l*] Im G_jl(r) )

where the *static* pairwise grating G_jl(r) = (A_j . A_l*) u_j(r) u_l*(r) exp(i (k_j - k_l).r) does
not depend on time at all, u_j being the envelope of the beam's lateral profile (1 for a plane wave). The polarization-dependent part of the Stark potential splits the same way, since it
too is a quadratic form of the field.

This is exact for arbitrary amplitude *and* phase modulation so lattice loading ramps, amplitude-modulation 
spectroscopy, Floquet shaking and moving lattices are all covered. And it is precisely the "static shape 
times pure time function" model of the PotentialT class, which is what lets the adapters below hand a beam 
lattice to the BECs solvers.

Three adapters are provided:

- 'add_to_potentialT' with mode="shapes": one PotentialT shape per grating. The parameter
  dimensions carried by the beams (a polarization angle, a wavelength, ...) survive into the shapes
  and are swept by the solvers like any other parameter.
- 'add_to_potentialT' with mode="stacked": the gratings are kept stacked in one array and combined
  in a single term. Around five times faster per time step, but the beams must then be free of
  parameter dimensions.
- 'add_to_analytic': a single function of time and coordinates, for the rescaling solver, whose grid
  changes at every step. No grating is stored.

Nothing here imports the BECs package: the adapters only use the public methods of the potential
object they are given.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field as dfield

import numpy as np
import xarray as xr

from optilat.potentials import cartesian_axes, is_zero

type tparam = float | xr.DataArray


def _accepted_names(func: Callable) -> set[str] | None:
    """The parameter names a drive function accepts, besides the time.

    Args:
        func (Callable): The function to inspect.

    Returns:
        Union[set[str],None]: The accepted names, or None if the function takes **kwargs and should
        therefore be given every parameter.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # builtins and C functions have no inspectable signature
        return None

    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None

    named = [
        p.name
        for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
    ]
    return set(named[1:])  # the first argument is the time


def _resolver(value: tparam, registry: dict[str, xr.DataArray]) -> Callable:
    """Turn a drive argument into a lookup function, registering it if it is a parameter dimension.

    A scalar is baked into the returned function, while a DataArray is registered under its own name
    and looked up at call time. Naming parameters after their coordinate is what makes two beams
    sharing the same parameter array share it in the potential as well, with no risk of collision.

    Args:
        value (Union[float,xr.DataArray]): The argument, either a scalar or a parameter dimension.
        registry (dict[str, xr.DataArray]): The parameter dictionary to update in place.

    Raises:
        ValueError: If a DataArray parameter has no name to register it under.

    Returns:
        Callable: A function taking the parameter dictionary and returning the value to use.
    """
    if isinstance(value, xr.DataArray):
        if value.name is None:
            raise ValueError(
                "A parameter given as a DataArray must be named, so that it can be identified "
                "as a parameter dimension. Use bloch_schrodinger.potential.create_parameter."
            )
        name = str(value.name)
        registry[name] = value
        return lambda params: params[name]

    return lambda params: value


def _ramp(t, vi, vf, ti, tf, smooth: bool):
    """A linear or smoothstep ramp from vi to vf between ti and tf, clamped outside."""
    span = tf - ti
    x = np.clip((t - ti) / span, 0.0, 1.0) if span != 0 else np.where(t < ti, 0.0, 1.0)
    if smooth:
        x = x * x * (3 - 2 * x)
    return vi + (vf - vi) * x


@dataclass
class BeamDrive:
    """The time dependence of a single beam: c(t) = amplitude(t) * exp(i * phase(t)).

    Both fields are either scalars or callables. A callable receives the time as first argument and
    then, by keyword, whichever entries of 'parameters' it names in its signature (or all of them if
    it takes **kwargs). Any parameter given as a named DataArray becomes a parameter dimension of the
    potential, swept by the solvers.
    """

    amplitude: tparam | Callable = 1.0
    phase: tparam | Callable = 0.0
    parameters: dict[str, xr.DataArray] = dfield(default_factory=dict)

    def __post_init__(self):
        self._amp_names = _accepted_names(self.amplitude) if callable(self.amplitude) else set()
        self._phase_names = _accepted_names(self.phase) if callable(self.phase) else set()

    @staticmethod
    def _call(value, names, t, params):
        """Evaluate a scalar-or-callable field on the time and the parameters it accepts."""
        if not callable(value):
            return value
        if names is None:
            return value(t, **params)
        return value(t, **{k: v for k, v in params.items() if k in names})

    def factor(self, t: float, **params) -> complex:
        """The complex factor c(t) multiplying the beam's amplitude.

        Args:
            t (float): The time.
            **params: The values of the parameter dimensions.

        Returns:
            complex: c(t) = amplitude(t) * exp(i phase(t)).
        """
        amp = self._call(self.amplitude, self._amp_names, t, params)
        phase = self._call(self.phase, self._phase_names, t, params)
        return amp * np.exp(1j * phase)

    ### ====================
    ### Named drives
    ### ====================

    @classmethod
    def constant(cls, amplitude: tparam = 1.0, phase: tparam = 0.0) -> "BeamDrive":
        """A beam that does not vary in time.

        Args:
            amplitude (Union[float,xr.DataArray], optional): The constant amplitude. Defaults to 1.
            phase (Union[float,xr.DataArray], optional): The constant phase. Defaults to 0.
        """
        registry = {}
        get_a, get_p = _resolver(amplitude, registry), _resolver(phase, registry)
        return cls(
            amplitude=lambda t, **p: get_a(p) + 0 * t,
            phase=lambda t, **p: get_p(p) + 0 * t,
            parameters=registry,
        )

    @classmethod
    def ramp(
        cls,
        vi: tparam = 0.0,
        vf: tparam = 1.0,
        ti: tparam = 0.0,
        tf: tparam = 1.0,
        smooth: bool = True,
        on: str = "depth",
    ) -> "BeamDrive":
        """A beam whose power is ramped, as when loading atoms into a lattice.

        Args:
            vi (Union[float,xr.DataArray], optional): The value before ti. Defaults to 0.
            vf (Union[float,xr.DataArray], optional): The value after tf. Defaults to 1.
            ti (Union[float,xr.DataArray], optional): Start of the ramp. Defaults to 0.
            tf (Union[float,xr.DataArray], optional): End of the ramp. Defaults to 1.
            smooth (bool, optional): Whether to use a smoothstep instead of a linear ramp, which
            avoids the kinks at ti and tf that excite the cloud. Defaults to True.
            on (str, optional): Whether the ramped quantity is the lattice "depth" (that is, the
            potential, which goes as the square of the field) or the field "amplitude" itself.
            Ramping the depth is what lattice depths usually mean. Defaults to "depth".

        Raises:
            ValueError: If 'on' is neither "depth" nor "amplitude".
        """
        if on not in ("depth", "amplitude"):
            raise ValueError(f"'on' must be 'depth' or 'amplitude', got {on!r}")

        registry = {}
        gvi, gvf, gti, gtf = (_resolver(v, registry) for v in (vi, vf, ti, tf))

        def amplitude(t, **p):
            value = _ramp(t, gvi(p), gvf(p), gti(p), gtf(p), smooth)
            # A depth goes as the square of the field, so the field carries its square root
            return np.sqrt(np.abs(value)) if on == "depth" else value

        return cls(amplitude=amplitude, parameters=registry)

    @classmethod
    def shake(cls, amp: tparam, freq: tparam, phase0: tparam = 0.0) -> "BeamDrive":
        """A phase-modulated beam, the drive used to shake a lattice (Floquet engineering).

        The lattice this beam belongs to is displaced back and forth without its depth changing.

        Args:
            amp (Union[float,xr.DataArray]): The modulation amplitude, in radians of optical phase.
            freq (Union[float,xr.DataArray]): The shaking frequency.
            phase0 (Union[float,xr.DataArray], optional): The phase of the drive itself, used to
            offset the shaking of different beams. Defaults to 0.
        """
        registry = {}
        ga, gf, gp = (_resolver(v, registry) for v in (amp, freq, phase0))
        return cls(
            phase=lambda t, **p: ga(p) * np.sin(2 * np.pi * gf(p) * t + gp(p)),
            parameters=registry,
        )

    @classmethod
    def move(cls, detuning: tparam) -> "BeamDrive":
        """A beam detuned from its partners, which makes the lattice they form travel.

        A single beam detuned by 'detuning' translates the interference pattern along its own
        direction at a speed of 2 pi * detuning / |k|.

        Args:
            detuning (Union[float,xr.DataArray]): The frequency offset of this beam.
        """
        registry = {}
        gd = _resolver(detuning, registry)
        return cls(phase=lambda t, **p: 2 * np.pi * gd(p) * t, parameters=registry)

    @classmethod
    def modulate(cls, depth: tparam, freq: tparam, mean: tparam = 1.0) -> "BeamDrive":
        """An amplitude-modulated beam, as used for lattice modulation spectroscopy.

        Args:
            depth (Union[float,xr.DataArray]): The relative modulation depth of the beam power.
            freq (Union[float,xr.DataArray]): The modulation frequency.
            mean (Union[float,xr.DataArray], optional): The mean beam power. Defaults to 1.
        """
        registry = {}
        gd, gf, gm = (_resolver(v, registry) for v in (depth, freq, mean))
        return cls(
            amplitude=lambda t, **p: np.sqrt(
                np.abs(gm(p) * (1 + gd(p) * np.sin(2 * np.pi * gf(p) * t)))
            ),
            parameters=registry,
        )

    @classmethod
    def pulse(cls, amplitude: tparam, t0: tparam, sigma: tparam) -> "BeamDrive":
        """A gaussian pulse of beam power, as used for Bragg or Kapitza-Dirac pulses.

        Args:
            amplitude (Union[float,xr.DataArray]): The peak amplitude of the field.
            t0 (Union[float,xr.DataArray]): The time of the peak.
            sigma (Union[float,xr.DataArray]): The rms duration of the pulse.
        """
        registry = {}
        ga, gt, gs = (_resolver(v, registry) for v in (amplitude, t0, sigma))
        return cls(
            amplitude=lambda t, **p: ga(p) * np.exp(-((t - gt(p)) ** 2) / (2 * gs(p) ** 2)),
            parameters=registry,
        )


class LatticeDrive:
    """The time dependence of every beam of a lattice, as one BeamDrive per beam.

    Beams are indexed in the order they were added to the OptiLat object. Any beam left unset simply
    keeps a constant unit factor, so only the driven beams have to be described.
    """

    def __init__(self, lattice, default: BeamDrive | None = None):
        """Initialize a drive for a given lattice, with every beam undriven.

        Args:
            lattice (OptiLat): The lattice to drive.
            default (BeamDrive, optional): The drive given to every beam initially. Defaults to a
            constant unit factor.
        """
        self.lattice = lattice
        self.drives: list[BeamDrive] = [
            default if default is not None else BeamDrive() for _ in lattice.beams
        ]

    def __len__(self) -> int:
        return len(self.drives)

    def __getitem__(self, key) -> BeamDrive | list[BeamDrive]:
        return self.drives[key]

    def __setitem__(self, key, drive: BeamDrive):
        """Assign a drive to one beam, a list of beams, or a slice of beams."""
        for i in self._indexes(key):
            self.drives[i] = drive

    def set(self, key, drive: BeamDrive) -> "LatticeDrive":
        """Assign a drive to one or several beams and return self, so that calls can be chained.

        Args:
            key (Union[int,list[int],slice]): The beam index, indexes or slice to drive.
            drive (BeamDrive): The drive to assign.

        Returns:
            LatticeDrive: self.
        """
        self[key] = drive
        return self

    def _indexes(self, key) -> list[int]:
        """Normalize an index, a list of indexes or a slice into a list of beam indexes."""
        if isinstance(key, slice):
            return list(range(len(self.drives)))[key]
        if isinstance(key, (list, tuple, np.ndarray)):
            return [int(i) for i in key]
        return [int(key)]

    @property
    def parameters(self) -> dict[str, xr.DataArray]:
        """Every parameter dimension of every beam drive, merged by name.

        Raises:
            ValueError: If two drives use the same parameter name for different coordinates.
        """
        merged: dict[str, xr.DataArray] = {}
        for drive in self.drives:
            for name, coord in drive.parameters.items():
                if name in merged and not merged[name].equals(coord):
                    raise ValueError(
                        f"Two beam drives use the parameter name '{name}' for different "
                        "coordinates. Parameters are shared by name, so give one of them a "
                        "different name."
                    )
                merged[name] = coord
        return merged

    def __repr__(self) -> str:
        return f"LatticeDrive over {len(self.drives)} beams, parameters {list(self.parameters)}"


@dataclass
class Grating:
    """One static shape of the expansion, together with the pure time function multiplying it."""

    name: str  # A suffix identifying the grating, used to build the potential's shape names
    shape: xr.DataArray  # The static spatial shape, real, physical constants included
    coeff: Callable  # (t, **parameters) -> float, a pure function of time
    layer: int  # The index of the coherent field this grating belongs to


@dataclass
class Expansion:
    """The full expansion of a driven lattice into static shapes times time functions."""

    gratings: list[Grating]
    parameters: dict[str, xr.DataArray]

    @property
    def param_dims(self) -> list[str]:
        """The dimensions the static shapes carry, beyond the coordinates they were evaluated on."""
        dims = {dim for g in self.gratings for dim in g.shape.dims}
        return sorted(dims)


def _coeff_product(drive_j: BeamDrive, drive_l: BeamDrive, part: str) -> Callable:
    """The real time function multiplying a pairwise grating.

    The time may be a scalar, as the solvers use it, or a whole time coordinate, as 'to_potential'
    and 'plot_t' use it, in which case the result keeps its dimensions.

    Args:
        drive_j (BeamDrive): The drive of the first beam of the pair.
        drive_l (BeamDrive): The drive of the second beam.
        part (str): Which part of c_j c_l* to return, "re", "im" or "abs2" for a diagonal pair.

    Returns:
        Callable: A function (t, **parameters) -> float, or an array of them.
    """
    if part == "abs2":
        return lambda t, **p: abs(drive_j.factor(t, **p)) ** 2

    sign = 1.0 if part == "re" else -1.0  # the minus of the Im term is folded in here
    take = np.real if part == "re" else np.imag
    return lambda t, **p: sign * take(
        drive_j.factor(t, **p) * np.conj(drive_l.factor(t, **p))
    )


def expand(
    lattice,
    drive: LatticeDrive | None = None,
    x: float | xr.DataArray = 0,
    y: float | xr.DataArray = 0,
    z: float | xr.DataArray = 0,
    alpha_s: float | xr.DataArray = 1.0,
    alpha_v: float | xr.DataArray = 0.0,
    M: float | xr.DataArray = 0.0,
    F: float | xr.DataArray = 1.0,
    rtol: float = 1e-12,
) -> Expansion:
    """Expand a driven optical lattice into static shapes multiplied by pure time functions.

    This is the core of the module, and knows nothing about the BECs package. The result is exact:
    summing shape * coeff(t) over the gratings gives the same potential as rebuilding every beam
    with amplitude c_j(t) and calling OptiLat.compute_fields.

    Args:
        lattice (OptiLat): The lattice to expand. Only beams sharing a field index interfere.
        drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
        lattice, in which case the expansion has constant coefficients.
        x (Union[float,xr.DataArray], optional): The x coordinates to evaluate the shapes on, with
        the same broadcasting rules as OptiLat.compute_fields. Defaults to 0.
        y (Union[float,xr.DataArray], optional): Same for y. Defaults to 0.
        z (Union[float,xr.DataArray], optional): Same for z. Defaults to 0.
        alpha_s (Union[float,xr.DataArray], optional): The scalar polarizability. Defaults to 1.
        alpha_v (Union[float,xr.DataArray], optional): The vector polarizability. Defaults to 0,
        which leaves out the polarization-dependent gratings entirely.
        M (Union[float,xr.DataArray], optional): The spin projection on z. Defaults to 0.
        F (Union[float,xr.DataArray], optional): The total angular momentum. Defaults to 1.
        rtol (float, optional): Gratings whose largest value is below this fraction of the largest
        grating are dropped: a Re or Im part often vanishes by symmetry, and an all-zero shape still
        costs a full grid multiplication at every time step. Defaults to 1e-12.

    Returns:
        Expansion: The static shapes, their time functions and the parameter dimensions of the drive.
    """
    drive = LatticeDrive(lattice) if drive is None else drive
    if len(drive) != len(lattice.beams):
        raise ValueError(
            f"The drive describes {len(drive)} beams but the lattice has {len(lattice.beams)}."
        )

    # Beams only interfere with the beams of their own coherent layer
    layers: dict[int, list[int]] = {}
    for i, (layer, _) in enumerate(lattice.beams):
        layers.setdefault(layer, []).append(i)

    scalar_pref = -0.25 * alpha_s
    vector_pref = -0.25 * alpha_v * M / (2 * F) * 2  # the 2 of 2 Im(Ex* Ey)
    with_vector = not (is_zero(alpha_v) or is_zero(M))

    gratings: list[Grating] = []

    for layer, indexes in layers.items():
        for a, j in enumerate(indexes):
            beam_j, drive_j = lattice.beams[j][1], drive[j]

            for l in indexes[a:]:
                beam_l, drive_l = lattice.beams[l][1], drive[l]
                diagonal = j == l
                weight = 1.0 if diagonal else 2.0

                # exp(i (k_j - k_l).r), the grating vector of the pair
                dk = beam_j.k - beam_l.k
                phase = xr.ufuncs.exp(
                    1j
                    * (
                        dk.isel(component=0) * x
                        + dk.isel(component=1) * y
                        + dk.isel(component=2) * z
                    )
                )
                # The lateral profiles, which leave plane waves untouched
                if beam_j.profile is not None or beam_l.profile is not None:
                    phase = phase * beam_j.envelope(x, y, z) * np.conj(beam_l.envelope(x, y, z))

                # --- scalar part: (A_j . A_l*) exp(i dk.r), contracted over the components ---
                kernel = (beam_j.A * beam_l.A.conj()).sum("component") * phase
                base = f"s{j}_{l}"
                gratings.append(
                    Grating(
                        f"{base}_re",
                        scalar_pref * weight * kernel.real,
                        _coeff_product(drive_j, drive_l, "abs2" if diagonal else "re"),
                        layer,
                    )
                )
                if not diagonal:
                    gratings.append(
                        Grating(
                            f"{base}_im",
                            scalar_pref * weight * kernel.imag,
                            _coeff_product(drive_j, drive_l, "im"),
                            layer,
                        )
                    )

                if not with_vector:
                    continue

                # --- vector part: 2 Im(Ex* Ey) needs the A*_x A_y contraction of both orderings ---
                # h_jl = A_j,x* A_l,y exp(-i dk.r) and h_lj = A_l,x* A_j,y exp(i dk.r)
                h_jl = beam_j.A.isel(component=0).conj() * beam_l.A.isel(component=1) * phase.conj()
                base = f"v{j}_{l}"
                if diagonal:
                    gratings.append(
                        Grating(
                            f"{base}_re",
                            vector_pref * h_jl.imag,
                            _coeff_product(drive_j, drive_l, "abs2"),
                            layer,
                        )
                    )
                    continue

                h_lj = beam_l.A.isel(component=0).conj() * beam_j.A.isel(component=1) * phase
                # Grouping the (j,l) and (l,j) orderings leaves Re[c_j c_l*] and Im[c_j c_l*] again
                gratings.append(
                    Grating(
                        f"{base}_re",
                        vector_pref * (h_jl.imag + h_lj.imag),
                        _coeff_product(drive_j, drive_l, "re"),
                        layer,
                    )
                )
                gratings.append(
                    Grating(
                        f"{base}_im",
                        vector_pref * (h_jl.real - h_lj.real),
                        _coeff_product(drive_j, drive_l, "im"),
                        layer,
                    )
                )

    gratings = _prune(gratings, rtol)
    return Expansion(gratings, drive.parameters)


def _prune(gratings: list[Grating], rtol: float) -> list[Grating]:
    """Drop the gratings that are negligible compared to the largest one.

    Args:
        gratings (list[Grating]): The gratings to filter.
        rtol (float): The relative threshold below which a grating is dropped.

    Returns:
        list[Grating]: The surviving gratings, or all of them if they are all negligible.
    """
    scales = [float(abs(g.shape).max()) for g in gratings]
    largest = max(scales, default=0.0)
    if largest == 0.0:
        return gratings
    return [g for g, scale in zip(gratings, scales) if scale > rtol * largest]


### ====================
### Adapters
### ====================


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


def _numpy_beams(lattice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The k-vectors, complex amplitude vectors and beam frames of every beam, as plain numpy arrays.

    Args:
        lattice (OptiLat): The lattice to read.

    Raises:
        ValueError: If a beam carries a parameter dimension, which cannot be reduced to a number.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The (n_beams, 3) k-vectors and complex
        amplitudes, and the (n_beams, 4, 3) beam frames: the focus, then the direction, TE and TM unit
        vectors, as used to evaluate the lateral profiles.
    """

    def arrays(beam):
        """Every array a beam is evaluated from, profile included."""
        arrs = [beam.k, beam.A]
        if beam.profile is not None:
            arrs += [p for p in beam.profile.parameters() if isinstance(p, xr.DataArray)]
        return arrs

    extra = sorted(
        {
            dim
            for _, beam in lattice.beams
            for arr in arrays(beam)
            for dim in arr.dims
            if dim != "component"
        }
    )
    if extra:
        raise ValueError(
            f"The beams carry the parameter dimension(s) {extra}, which this route cannot keep. "
            "Select them on the beams first, or use add_to_potentialT with mode='shapes'."
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


def add_to_potentialT(
    potT,
    lattice,
    drive: LatticeDrive | None = None,
    name: str = "lat",
    mode: str = "shapes",
    rtol: float = 1e-12,
    **stark,
) -> Expansion:
    """Add a driven optical lattice to a PotentialT object, as shapes and time functions.

    The lattice is evaluated on the potential's own cartesian grid and expanded into static gratings
    times pure time functions, so that 'make_Vt', 'to_potential' and 'plot_t' all work as usual. The
    terms are added alongside whatever the potential already holds: an existing trap in V0 or a shape
    added with 'circle_t' is left untouched.

    Args:
        potT: A PotentialT-like object, used through its add_shape, add_function, add_context_func
        and add_term methods only. Build it with v0=0 if the lattice is to be the whole potential.
        lattice (OptiLat): The lattice to add.
        drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
        lattice.
        name (str, optional): The prefix of every generated shape, function and term name. Defaults
        to "lat".
        mode (str, optional): "shapes" registers one shape per grating, which keeps the parameter
        dimensions of the beams sweepable. "stacked" keeps the gratings in a single array combined by
        one term, which is several times faster per time step but requires parameter-free beams.
        Defaults to "shapes".
        rtol (float, optional): The relative threshold below which a grating is dropped. Defaults to
        1e-12.
        **stark: The polarizabilities passed on to 'expand' (alpha_s, alpha_v, M, F).

    Raises:
        ValueError: If 'mode' is unknown, if a generated name is already taken, or if mode="stacked"
        is used with beams carrying parameter dimensions.

    Returns:
        Expansion: The expansion that was registered, useful for inspection.
    """
    if mode not in ("shapes", "stacked"):
        raise ValueError(f"'mode' must be 'shapes' or 'stacked', got {mode!r}")

    expansion = expand(lattice, drive, **cartesian_axes(potT), rtol=rtol, **stark)
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
        if not hasattr(potT, "add_context_func"):
            raise ValueError(
                "mode='stacked' needs the potential to support add_context_func, which this "
                "class does not. Use mode='shapes' instead."
            )
        extra = [dim for dim in expansion.param_dims if dim not in potT.spatial_dims]
        if extra:
            raise ValueError(
                f"mode='stacked' cannot keep the parameter dimension(s) {extra} carried by the "
                "gratings. Select them on the beams first, or use mode='shapes'."
            )

        # All the gratings in one array, contracted with the coefficient vector in a single term
        pair = f"_{name}_pair"
        spatial = tuple(potT.spatial_dims)
        grid_coords = dict(zero.coords)
        stack = np.stack(
            [
                np.asarray((grating.shape + zero).transpose(*spatial))
                for grating in expansion.gratings
            ]
        )
        coeffs = [grating.coeff for grating in expansion.gratings]

        def coefficient_vector(t, **params):
            """The coefficient of every grating at time t, as a vector."""
            values = [coeff(t, **params) for coeff in coeffs]
            if all(np.ndim(value) == 0 for value in values):
                return np.asarray(values, dtype=float)  # a scalar time: the solvers' path
            # to_potential and plot_t hand over a whole time coordinate instead, and the
            # coefficients then carry dimensions of their own that have to stay labelled
            return xr.concat(
                [xr.DataArray(value) for value in values], dim=pair, coords="minimal"
            )

        def combine(c):
            """Contract the gratings with their coefficients, labelled or not."""
            if isinstance(c, np.ndarray):
                return xr.DataArray(
                    np.tensordot(c, stack, axes=(0, 0)), dims=spatial, coords=grid_coords
                )
            gratings = xr.DataArray(stack, dims=(pair,) + spatial, coords=grid_coords)
            return (gratings * c).sum(pair)

        fname, cname = f"{name}_cvec", f"{name}_sum"
        _check_free(fname, potT.funcs, "time function")
        _check_free(cname, potT.context_funcs, "context function")

        potT.add_function(fname, coefficient_vector, parameters=expansion.parameters)
        potT.add_context_func(cname, combine)
        terms = [f"{cname}({fname})"]

    potT.add_term(" + ".join(terms), name=name)
    return expansion


def add_to_analytic(
    anpot,
    lattice,
    drive: LatticeDrive | None = None,
    name: str = "lat",
    **stark,
) -> None:
    """Add a driven optical lattice to an AnalyticPotential object, as a function of time and space.

    Contrarily to the PotentialT route, nothing is precomputed on a grid: the beam superposition is
    evaluated in numpy on whichever coordinates the solver hands over, which is what the rescaling
    solver needs, since its grid changes at every time step.

    Args:
        anpot: An AnalyticPotential-like object, used through its add_function and add_term methods.
        lattice (OptiLat): The lattice to add. Its beams must be free of parameter dimensions.
        drive (LatticeDrive, optional): The time dependence of each beam. Defaults to a static
        lattice.
        name (str, optional): The name of the generated function and term. Defaults to "lat".
        **stark: The polarizabilities alpha_s, alpha_v, M and F, as in stark_potential.

    Raises:
        ValueError: If the generated name is already taken, if the beams carry parameter dimensions,
        or if a polarizability is given as a parameter dimension.
    """
    _check_free(name, anpot.funcs, "function")

    drive = LatticeDrive(lattice) if drive is None else drive
    k, A, frames = _numpy_beams(lattice)
    layers = np.array([layer for layer, _ in lattice.beams])
    profiles = [beam.profile for _, beam in lattice.beams]
    kls = np.linalg.norm(k, axis=1)

    swept = [key for key, val in stark.items() if isinstance(val, xr.DataArray)]
    if swept:
        raise ValueError(
            f"This route evaluates the potential in numpy, so the polarizability argument(s) "
            f"{swept} cannot be parameter dimensions. Pass them as scalars, or use "
            "add_to_potentialT with mode='shapes'."
        )

    alpha_s = stark.get("alpha_s", 1.0)
    alpha_v, M, F = stark.get("alpha_v", 0.0), stark.get("M", 0.0), stark.get("F", 1.0)
    with_vector = not (is_zero(alpha_v) or is_zero(M))

    def lattice_potential(t, *coords, **params):
        """The Stark potential of the driven lattice at time t, over the coordinates given."""
        grid = np.zeros(np.broadcast_shapes(*(np.shape(c) for c in coords)))
        V = np.zeros(grid.shape)

        for layer in np.unique(layers):
            E = np.zeros((3,) + grid.shape, dtype=complex)
            for j in np.flatnonzero(layers == layer):
                # k has 3 components, but only the axes the solver provides are resolved
                kdr = sum(k[j, i] * np.asarray(c) for i, c in enumerate(coords))
                factor = drive[int(j)].factor(t, **params) * np.exp(1j * kdr)
                if profiles[j] is not None:
                    # The axes the solver does not provide sit at 0, as in cartesian_axes
                    rel = [
                        (np.asarray(coords[i]) if i < len(coords) else 0) - frames[j, 0, i]
                        for i in range(3)
                    ]
                    s, t1, t2 = (sum(frames[j, a, i] * rel[i] for i in range(3)) for a in (1, 2, 3))
                    factor = factor * profiles[j].envelope(s, t1, t2, kls[j])
                for comp in range(3):
                    E[comp] = E[comp] + A[j, comp] * factor

            V = V + alpha_s * (abs(E) ** 2).sum(0)
            if with_vector:
                V = V + alpha_v * M / (2 * F) * 2 * (E[0].conj() * E[1]).imag

        return -0.25 * V

    anpot.add_function(name, lattice_potential, parameters=drive.parameters)
    anpot.add_term(name)
