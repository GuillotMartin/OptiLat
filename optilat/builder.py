import numpy as np
import xarray as xr
from typing import Union, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ipywidgets import VBox, interactive_output
from IPython.display import display

if TYPE_CHECKING:
    from optilat.profiles import Profile

type treal = Union[float, xr.DataArray]
type tcomplex = Union[complex, xr.DataArray]
type tpolar = Union[list[tcomplex, tcomplex], xr.DataArray]
type vec3d = Union[list[treal, treal, treal], xr.DataArray]

# Below this in-plane norm, a direction is treated as purely vertical (+-z) and the
# TE/TM basis is chosen by convention instead of from the (undefined) in-plane part.
_INPLANE_EPS = 1e-12


def format_polar(polar: Union[list, np.ndarray, xr.DataArray]) -> xr.DataArray:
    """format a polarization in the Jones formalism as a dedicated xr.DataArray with a 'Jones' dimension.

    Args:
        polar (Union[list, np.ndarray, xr.DataArray]): The Jones vector, as a size-2 sequence of
        complex scalars or DataArrays, or as a DataArray with a size-2 "Jones" dimension.
    Returns:
        xr.DataArray: The Jones vector with a size-2 "Jones" dimension.
    """

    if isinstance(polar, xr.DataArray):
        if "Jones" not in polar.dims:
            raise ValueError(
                "The polarization object is a DataArray without a Jones dimension"
            )
        else:
            return polar
    elif isinstance(polar, (list, tuple, np.ndarray)):
        if len(polar) != 2:
            raise ValueError("polarization is list of length != 2")
        new_polar = xr.concat(
            [xr.DataArray(polar[0]), xr.DataArray(polar[1])],
            dim="Jones",
            coords="minimal",
        )
        return new_polar.assign_coords({"Jones": np.arange(2)})
    else:
        raise TypeError()


def format_3dvec(vec: Union[list, np.ndarray, xr.DataArray]) -> xr.DataArray:
    """format a 3d vector in the cartesian basis as a dedicated xr.DataArray with a 'component' dimension.

    Args:
        vec (Union[list, np.ndarray, xr.DataArray]): The vector, as a size-3 sequence of scalars or
        DataArrays, or as a DataArray with a size-3 "component" dimension.
    Returns:
        xr.DataArray: The vector with a size-3 "component" dimension.
    """

    if isinstance(vec, xr.DataArray):
        if "component" not in vec.dims:
            raise ValueError(
                "The vector object is a DataArray without a component dimension"
            )
        else:
            return vec
    elif isinstance(vec, (list, tuple, np.ndarray)):
        if len(vec) != 3:
            raise ValueError("Vector is list of length != 3")
        new_vec = xr.concat(
            [xr.DataArray(vec[0]), xr.DataArray(vec[1]), xr.DataArray(vec[2])],
            dim="component",
            coords="minimal",
        )
        return new_vec.assign_coords({"component": np.arange(3)})
    else:
        raise TypeError()


class Beam:
    def __init__(
        self,
        amplitude: tcomplex = 1,
        wavelength: treal = None,
        k: Union[treal, vec3d] = None,
        direction: vec3d = None,
        polar: tpolar = None,
        profile: "Profile" = None,
    ):
        """The Beam class contains simple functions to handle laser-generated plane waves ofr optical lattice construction.
        It supports xarray broadcast rules on every input. A beam's k-vector can be given as a wavelength + direction,
        k-modulus + direction, or k-vector directly.

        Args:
            amplitude (tcomplex, optional): The overall beam complex amplitude. Defaults to 1.
            wavelength (treal, optional): The beam's wavelength. Defaults to None.
            k (Union[treal, vec3d], optional): The beam's k vector. If given as a float, it is interpreted as the modulus
            and a direction must be given as well. If given as a list of 3 floats, each element is interpreted as a
            k-vector component. If given as an xarray.DataArray with a "component" dimension of length 3, it is
            interpreted as a k-vector, and as a modulus otherwise. Defaults to None.
            direction (vec3d, optional): The direction of propagation. Can be given as a list of 3 floats or as a
            DataArray with a size-3 "component" dimension. It is normalized before use. Defaults to None.
            polar (tpolar, optional): The Jones vector of the beam in the beam's frame.
            The first component is always in the xy plane. It is used as given, without renormalization, so that
            the norm of the resulting complex amplitude is |amplitude| times the norm of the Jones vector.
            Defaults to [1,0].
            profile (Profile, optional): The lateral profile of the beam, e.g. a GaussianProfile. The
            field of the beam is then A u(r) exp(i k.r), with u the profile's envelope. Defaults to
            None, an infinite plane wave.
        """

        self.amplitude = amplitude  # Amplitude of the beam
        self.profile = profile  # Lateral profile of the beam, None for a plane wave
        self.polar = format_polar(
            [1, 0] if polar is None else polar
        )  # The polarization is formatted as a complex DataArray with a size-2 "Jones" dimension

        if direction is not None:
            direction = format_3dvec(
                direction
            )  # The direction is formatted as a DataArray with a size-3 "component" dimension
            direction = (
                direction / (direction**2).sum("component") ** 0.5
            )  # Normalization

        if k is None and wavelength is None:
            raise ValueError("either wavelength or k vector must be specified.")

        if k is None:
            if direction is None:
                raise ValueError("Direction must be specified in wavelength mode.")

            self.kl = 2 * np.pi / wavelength  # k-vector modulus
            self.k: xr.DataArray = (
                self.kl * direction
            )  # The k-vector is formatted as a DataArray with a size-3 "component" dimension

        else:
            # A k given without a "component" dimension is a modulus, and needs a direction
            if isinstance(k, xr.DataArray):
                is_modulus = "component" not in k.dims
            else:
                is_modulus = np.ndim(k) == 0

            if is_modulus:
                if direction is None:
                    raise ValueError("Direction must be specified in k-modulus mode.")
                self.kl = k
                self.k: xr.DataArray = k * direction
            else:
                self.k: xr.DataArray = format_3dvec(k)
                self.kl = (self.k**2).sum("component") ** 0.5

        self.direction: xr.DataArray = self.k / self.kl
        self.TE, self.TM = self.compute_3d_Polar()
        self.A = self.compute_Camplitude()

    def __repr__(self):
        profile = "plane wave" if self.profile is None else self.profile
        return (
            f"A beam with k-vector: {self.kl}, \ndirection {self.direction} "
            f"\npolarization {self.polar} \nand profile {profile}"
        )

    def compute_3d_Polar(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute the TE and TM unit vectors' components in the cartesian basis.

        Both vectors are normalized and orthogonal to the direction of propagation, and
        (TE, TM, direction) always forms a right-handed orthonormal triad, so that a Jones
        vector of unit norm produces a complex amplitude of unit norm whatever the direction.

        Returns:
            tuple[xr.DataArray, xr.DataArray]: The TE and TM unit vectors.
        """
        TE = xr.zeros_like(self.direction, dtype=float)  # First vector orthogonal to k
        TM = xr.zeros_like(self.direction, dtype=float)  # Second vector orthogonal to k

        dx = self.direction[{"component": 0}]
        dy = self.direction[{"component": 1}]
        dz = self.direction[{"component": 2}]

        # In-plane norm of the direction, i.e. sin(theta). The TE/TM basis is degenerate when
        # it vanishes, so vertical beams get their basis from a separate convention below.
        inplane = (dx**2 + dy**2) ** 0.5
        vertical = inplane < _INPLANE_EPS
        # Only used where the beam is not vertical, but must stay finite everywhere so that
        # no division by zero happens on the discarded branch.
        norm = xr.where(vertical, 1.0, inplane)

        # The TE vector is contained in the xy-plane. For a vertical beam we pick TE along
        # +-x, its sign following the propagation direction to keep the triad right-handed.
        TE[{"component": 0}] = xr.where(vertical, dz, -dy / norm)
        TE[{"component": 1}] = xr.where(vertical, 0.0, dx / norm)

        # The second vector is determined by the cross-product of the direction and TE
        TM[{"component": 0}] = xr.where(vertical, 0.0, -dx * dz / norm)
        TM[{"component": 1}] = xr.where(vertical, 1.0, -dy * dz / norm)
        TM[{"component": 2}] = xr.where(vertical, 0.0, inplane)
        return TE, TM

    def compute_Camplitude(self) -> xr.DataArray:
        """Returns the complex amplitude (Ax, Ay, Az) of the beam. The full EM-field produced can then be written as Ei = Re[Ai exp(1j * (k.r - w.t))]

        Returns:
            xr.DataArray: Complex amplitude (Ax, Ay, Az)
        """

        A = xr.zeros_like(self.direction, dtype=complex)
        A = A + self.TE * self.polar[{"Jones": 0}] * self.amplitude
        A = A + self.TM * self.polar[{"Jones": 1}] * self.amplitude

        return A

    def beam_frame(self, x: treal = 0, y: treal = 0, z: treal = 0) -> tuple:
        """The coordinates of the points (x, y, z) in the beam frame, centred on the profile's focus.

        Args:
            x (treal, optional): The x-coordinate. Defaults to 0.
            y (treal, optional): The y-coordinate. Defaults to 0.
            z (treal, optional): The z-coordinate. Defaults to 0.

        Returns:
            tuple: (s, t1, t2), the coordinates along the direction of propagation, the TE and the
            TM unit vectors. The focus is the origin, or the origin of space for a plane wave.
        """
        focus = [0, 0, 0] if self.profile is None else self.profile.focus
        rel = [r - _comp(focus, i) for i, r in enumerate((x, y, z))]

        def project(vec):
            return sum(_comp(vec, i) * rel[i] for i in range(3))

        return project(self.direction), project(self.TE), project(self.TM)

    def envelope(self, x: treal = 0, y: treal = 0, z: treal = 0):
        """The complex envelope u(r) of the beam, whose field is A u(r) exp(i k.r).

        Args:
            x (treal, optional): The x-coordinate. Defaults to 0.
            y (treal, optional): The y-coordinate. Defaults to 0.
            z (treal, optional): The z-coordinate. Defaults to 0.

        Returns:
            The envelope, or 1 for a plane wave.
        """
        if self.profile is None:
            return 1
        return self.profile.envelope(*self.beam_frame(x, y, z), self.kl)


def _comp(vec, i: int):
    """The i-th cartesian component of a vector, without a leftover "component" coordinate."""
    if isinstance(vec, xr.DataArray):
        return vec.isel(component=i, drop=True)
    return vec[i]


class OptiLat:
    def __init__(self):
        """An optical lattice is made of the superposition of multiple laser beams.
        This superposition can be coherent, incoherent or a combination of both.
        """
        self.beams: list[
            tuple[int, Beam]
        ] = []  # List of beams objects and their respective fields.
        self.Coherence: dict[int, list[Beam]] = {}  # the different coherent fields indexes
        self.maxIndex = 0

    def add_beam(self, beam: Union[list[Beam], Beam], index: Union[int, list[int]] = 0):
        """Add a beam or a list of beam object to the lattice. Each beam must be assigned a field index.
        All beams with the same field index are considered coherent for the final
        computation of complex amplitudes.


        Args:
            beam (Union[list[Beam], Beam]): The beams to add
            index (Union[int, list[int]], optional): Index of the beam. A single index is shared by every
            beam passed, otherwise the list of indexes must have exactly one entry per beam. A None index
            means "a new field of its own". Defaults to 0.
        """
        if isinstance(beam, Beam):
            beams = [beam]
        else:
            beams = beam
        if index is None or isinstance(index, int):
            indexes = [index] * len(beams)
        else:
            indexes = index

        if len(indexes) != len(beams):
            raise ValueError(
                f"{len(beams)} beams were passed but {len(indexes)} indexes: "
                "each beam must be given exactly one field index."
            )

        for index, beam in zip(indexes, beams):
            if index is None:
                index = self.maxIndex
            if index >= self.maxIndex:
                self.maxIndex = index + 1

            self.beams.append((index, beam))
            if index in self.Coherence.keys():
                self.Coherence[index] = self.Coherence[index] + [beam]
            else:
                self.Coherence[index] = [beam]

    def compute_fields(self, x: treal = 0, y: treal = 0, z: treal = 0) -> xr.DataArray:
        """The main function of the class, evaluate the complex-amplitude of each coherent
        field in the optical lattice over a specified region of space.

        Args:
            x (Union[float, xr.DataArray], optional): The x-coordinate where to evaluate the field. Fully compatible with xarray
            broadcast rules and compatible with the Potential class from the bloch_schrodinger package. Defaults to 0.
            y (Union[float, xr.DataArray], optional): Same for the y-coordinate. Defaults to 0.
            z (Union[float, xr.DataArray], optional): Same for the z-coordinate. Defaults to 0.

        Returns:
            xr.DataArray: A DataArray "Fields" with a "field" dimension and a size-3 "component" dimension.
            Each field represents the coherent superposition of the beams with the same field index. The component dimension
            represents the 3 components of the complex, spatially dependant, amplitude A(r) = (Ax(r), Ay(r), Az(r))
        """
        coherent_layers = list(self.Coherence.keys())

        if not coherent_layers:
            raise ValueError("The lattice is empty, add beams with add_beam first.")

        # Each coherent layer is summed separately, so that no beam is ever broadcast over the
        # whole "field" dimension. The layers are only stacked together at the very end.
        layers: dict[int, xr.DataArray] = {}

        for co, beam in self.beams:
            kdr = (
                beam.k[{"component": 0}] * x
                + beam.k[{"component": 1}] * y
                + beam.k[{"component": 2}] * z
            )
            ToAdd = beam.A * beam.envelope(x, y, z) * xr.ufuncs.exp(1j * kdr)
            layers[co] = ToAdd if co not in layers else layers[co] + ToAdd

        Fields = xr.concat(
            [layers[co] for co in coherent_layers], dim="field", coords="minimal"
        ).assign_coords({"field": coherent_layers})

        return Fields.rename("Fields")

    def plot(
        self,
        box: Union[float, list[float], xr.DataArray] = 10,
        laser_style: Union[dict, list[dict]] = None,
        slider_start: str = "left",
    ) -> tuple[Figure, Axes]:
        """An interactive plotting function for an optical lattice. Represents each laser beam by an arrow, with its polarization ellipse. 
        The lenth of each laser arrow is equal to the wavelength of the corresponding laser beam.

        Args:
            box (Union[float, list[float,float,float], xr.DataArray], optional): The size of the box's sides in arbitrary units.
            Can be given as a single scalar for a cubic box, a list of 3 scalars for a custom rectangular box,
            or a DataArray with a size-3 component dimension for a dynamically sized box. Defaults to 10.
            laser_style (Union[dict, list[dict]], optional): The styles of the laser arrow and polarization ellipse. 
            If None is given, a simple style will be used.
            If a list of dict is given, then the style of each laser beam will be looped over this list. 
            Each style can contain a "direction" key linked to a dictionnary that will be passed as kwargs
            for matplotlib's quiver function and a "polar"key that will be passed likewise to the
            plot function for the polarization ellipse. A missing key falls back to the default style.
            Defaults to None.
            slider_start (str, optional): The default starting position of the sliders, can be "left" or "mid". Defaults to "left".

        Returns:
            tuple[Figure,Axes]
        """
        # Imported lazily: everything but this interactive plot works without bloch_schrodinger
        try:
            from bloch_schrodinger.utils import create_sliders_from_dims
        except ImportError as err:
            raise ImportError(
                "OptiLat.plot requires the optional bloch_schrodinger package, see the README."
            ) from err

        if isinstance(box, (int, float)):
            box = xr.DataArray([box, box, box], coords={"component": [0, 1, 2]})
        elif isinstance(box, (list, tuple, np.ndarray)):
            box = xr.DataArray(list(box), coords={"component": [0, 1, 2]})

        if laser_style is None:
            laser_styles = [
                {
                    "direction": {"colors": "k", "linewidths": 2},
                    "polar": {
                        "color": "r",
                        "linewidth": 2,
                    },
                }
            ]
        elif isinstance(laser_style, dict):
            laser_styles = [laser_style]
        else:
            laser_styles = laser_style
        l_s = len(laser_styles)

        # Creating the sliders objects, one per parameter dimension of the box and of the beams
        slider_dims = []
        dict_coords = {}

        def register_dims(obj, skip):
            """Register every parameter dimension of obj that has no slider yet."""
            if not isinstance(obj, xr.DataArray):
                return
            for dim in obj.dims:
                if dim in skip or dim in slider_dims:
                    continue
                if dim not in obj.coords:
                    raise ValueError(
                        f"The '{dim}' dimension has no coordinate values, so no slider can be "
                        "built for it. Give it coordinates, e.g. with "
                        "bloch_schrodinger.potential.create_parameter."
                    )
                dict_coords[dim] = obj.coords[dim]
                slider_dims.append(dim)

        register_dims(box, ["component"])
        for ind, beam in self.beams:
            register_dims(beam.k, ["component"])
            register_dims(beam.polar, ["Jones"])
            register_dims(beam.amplitude, [])
            if beam.profile is not None:
                for param in beam.profile.parameters():
                    register_dims(param, ["component"])

        sliders = create_sliders_from_dims(
            {dim: dict_coords[dim] for dim in slider_dims}, start=slider_start
        )

        # Initial parameter selections
        initial_sel = {dim: sliders[dim].value for dim in sliders}

        # Functions

        def select(obj: xr.DataArray, sel: dict) -> xr.DataArray:
            """Reduce obj to the currently selected slider values."""
            subsel = {dim: val for dim, val in sel.items() if dim in obj.dims}
            return obj.sel(subsel, method="nearest")

        def components(obj: xr.DataArray) -> list[float]:
            """The 3 cartesian components of obj as plain floats."""
            return [float(obj.isel(component=i)) for i in range(3)]

        def set_box(ax, box, sel):
            lx, ly, lz = components(select(box, sel))
            ax.set_xlim(-lx / 2, lx / 2)
            ax.set_ylim(-ly / 2, ly / 2)
            ax.set_zlim(-lz / 2, lz / 2)

        def place_beam(
            ax: Axes, beam: Beam, box: xr.DataArray, sel: dict, laser_style: dict
        ):

            k_sel = select(beam.k, sel)
            polar_sel = select(beam.polar, sel)
            TE_sel = select(beam.TE, sel)
            TM_sel = select(beam.TM, sel)

            k_length = float((abs(k_sel) ** 2).sum() ** 0.5)
            k_dir = components(k_sel / k_length)
            size = components(select(box, sel))

            position = [-d * s / 2 for d, s in zip(k_dir, size)]
            dir = ax.quiver(
                *position,  # base position
                *k_dir,  # direction
                length=2 * np.pi / k_length,
                **laser_style.get("direction", {"colors": "b"}),
            )

            # The polarization ellipse, drawn over one optical period at the arrow's base
            t = np.linspace(0, 1, 100)
            TE_osc = np.real(np.exp(1j * 2 * np.pi * t) * complex(polar_sel.isel(Jones=0)))
            TM_osc = np.real(np.exp(1j * 2 * np.pi * t) * complex(polar_sel.isel(Jones=1)))
            comps = [
                (te * TE_osc + tm * TM_osc) * np.pi / k_length + p
                for te, tm, p in zip(components(TE_sel), components(TM_sel), position)
            ]
            pol = ax.plot(*comps, **laser_style.get("polar", {"color": "r", "linewidth": 2}))

            return dir, pol

        # Initial data selection

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        set_box(ax, box, initial_sel)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        list_dir, list_pol = [], []

        for i, (ind, beam) in enumerate(self.beams):
            dir, pol = place_beam(ax, beam, box, initial_sel, laser_styles[i%l_s])
            list_dir += [dir]
            list_pol += [pol]

        def update(**kwargs):
            sel = {dim: kwargs[dim] for dim in sliders}
            for line in ax.lines:
                line.remove()
            for i, (ind, beam) in enumerate(self.beams):
                list_dir[i].remove()
                list_dir[i], list_pol[i] = place_beam(ax, beam, box, sel, laser_styles[i%l_s])
            set_box(ax, box, sel)  # The box itself can depend on the sliders

            fig.canvas.draw_idle()

        out = interactive_output(update, sliders)
        # Display everything
        display(VBox(list(sliders.values()) + [out]))
        return fig, ax


if __name__ == "__main__":
    from bloch_schrodinger.potential import create_parameter

    lamb = 0.83  # Laser wavelength
    laser_angles = [np.pi / 2 + np.pi * 2 / 3 * i for i in range(3)]  # 120deg lasers

    theta = create_parameter("theta", np.linspace(0, np.pi / 2, 50))
    phi = create_parameter("phi", np.linspace(0, np.pi, 50))
    dirtst = create_parameter("dirtest", np.linspace(0, 1, 20))

    beams = [
        Beam(
            wavelength=lamb,
            direction=[np.cos(ang), np.sin(ang) + dirtst, 0],
            polar=[np.cos(theta), np.sin(theta) * np.exp(1j * phi)],
        )
        for ang in laser_angles
    ]

    lattice = OptiLat()
    lattice.add_beam(beams, [0] * 3)
    lattice.plot(box=[7, 7, 2])
    plt.show()
