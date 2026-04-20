import numpy as np
import xarray as xr
from typing import Union
from bloch_schrodinger.utils import create_sliders_from_dims
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ipywidgets import FloatSlider, HBox, VBox, interactive_output
from IPython.display import display
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

type treal = Union[float, xr.DataArray]
type tcomplex = Union[complex, xr.DataArray]
type tpolar = Union[list[tcomplex, tcomplex], xr.DataArray]
type vec3d = Union[list[treal, treal], xr.DataArray]


def format_polar(polar: Union[list, np.ndarray, xr.DataArray]) -> xr.DataArray:
    """format a polarization in the Jones formalism as a dedicated xr.DataArray with a 'Jones' dimension.

    Args:
        polar (Union[list, np.ndarray, xr.DataArray]):
    Returns:
        xr.DataArray:
    """

    if isinstance(polar, xr.DataArray):
        if "Jones" not in polar.dims:
            raise ValueError(
                "The polarization object is a DataArray without a Jones dimension"
            )
        else:
            return polar
    elif isinstance(polar, Union[list, np.ndarray]):
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
        polar (Union[list, np.ndarray, xr.DataArray]):
    Returns:
        xr.DataArray:
    """

    if isinstance(vec, xr.DataArray):
        if "component" not in vec.dims:
            raise ValueError(
                "The vector object is a DataArray without a component dimension"
            )
        else:
            return vec
    elif isinstance(vec, Union[list, np.ndarray]):
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
        k: Union[treal, list[float, float, float], vec3d] = None,
        direction: Union[list[float, float, float], vec3d] = None,
        polar: Union[list[float, float, float], vec3d] = [1, 0],
    ):
        """The Beam class contains simple functions to handle laser-generated plane waves ofr optical lattice construction.
        It supports xarray broadcast rules on every input. A beam's k-vector can be given as a wavelength + direction,
        k-modulus + direction, or k-vector directly.

        Args:
            amplitude (tcomplex, optional): The overall beam complex amplitude. Defaults to 1.
            wavelength (treal, optional): The beam's wavelength. Defaults to None.
            k (Union[treal, list[float, float, float], vec3d], optional): The beam's k vector. if given as a float, it is interpreted as the modulus.
            If given as a list of 3 floats, each element is interpreted as a k-vector component. If given as an xarray.DataArray,
            it must have a ¨component"dimension of length 3. Defaults to None.
            direction (Union[list[float, float, float], vec3d], optional): The direction of propagation. Can be given using the same options as k.
            It is normalized before use. Defaults to None.
            polar (Union[list[float, float, float], vec3d], optional): The Jones vector of the beam in the beam's frame.
            The first component is always in the xy plane. Defaults to [1,0].
        """

        self.amplitude = amplitude  # Amplitude of the beam
        self.polar = format_polar(
            polar
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
            if wavelength is not None and direction is None:
                raise ValueError("Direction must be specified in wavelength mode.")

            self.kl = 2 * np.pi / wavelength  # k-vector modulus
            self.k: xr.DataArray = (
                self.kl * direction
            )  # The k-vector is formatted as a DataArray with a size-3 "component" dimension

        else:
            if isinstance(k, Union[int, float]):
                self.kl = k
                self.k: xr.DataArray = k * direction
            else:
                self.k: xr.DataArray = format_3dvec(k)
                self.kl = (self.k**2).sum("component") ** 0.5

        self.direction: xr.DataArray = self.k / self.kl
        self.TE, self.TM = self.compute_3d_Polar()
        self.A = self.compute_Camplitude()

    def __repr__(self):
        return f"A beam with k-vector: {self.kl}, \ndirection {self.direction} \nand polarization {self.polar}"

    def compute_3d_Polar(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute the TE and TM vector's component in the cartesian basis

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        TE = xr.zeros_like(self.direction)  # First vector orthogonal to k
        TM = xr.zeros_like(self.direction)  # Second vector orthogonal to k

        dx = self.direction[{"component": 0}]
        dy = self.direction[{"component": 1}]
        dz = self.direction[{"component": 2}]

        # The TE vector is contained in the xy-plane
        TE[{"component": 0}] = xr.where(
            (xr.ufuncs.equal(dx, 0) * xr.ufuncs.equal(dy, 0)), dz, -dy
        )
        TE[{"component": 1}] = xr.where(
            (xr.ufuncs.equal(dx, 0) * xr.ufuncs.equal(dy, 0)), 0, dx
        )

        # The second vector is determined by the cross-product of k and TE
        TM[{"component": 0}] = xr.where(
            (xr.ufuncs.equal(dx, 0) * xr.ufuncs.equal(dy, 0)), 0, -dx * dz
        )
        TM[{"component": 1}] = xr.where(
            (xr.ufuncs.equal(dx, 0) * xr.ufuncs.equal(dy, 0)), dz, -dy * dz
        )
        TM[{"component": 2}] = dy**2 + dx**2
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


class OptiLat:
    def __init__(self):
        """An optical lattice is made of the superposition of multiple laser beams.
        This superposition can be coherent, incoherent or a combination of both.
        """
        self.beams: list[
            list[int, Beam]
        ] = []  # List of beams objects and their respective fields.
        self.Coherence: dict[list[Beam]] = {}  # the different coherent fields indexes
        self.maxIndex = 0

    def add_beam(self, beam: Union[list[Beam], Beam], index: Union[int, list[int]] = 0):
        """Add a beam or a list of beam object to the lattice. Each beam must be assigned a field index.
        All beams with the same field index are considered coherent for the final
        computation of complex amplitudes.


        Args:
            beam (Union[list[Beam], Beam]): The beams to add
            index (Union[int, list[int]], optional): Index of the beam, if a list of beams is passed,
            then a list of indexes must be passed too. Defaults to 0.
        """
        if isinstance(beam, Beam):
            beams = [beam]
        else:
            beams = beam
        if isinstance(index, int):
            indexes = [index]
        else:
            indexes = index

        for index, beam in zip(indexes, beams):
            if index is None:
                index = self.maxIndex
            if index >= self.maxIndex:
                self.maxIndex = index + 1

            self.beams.append([index, beam])
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

        coherence = xr.DataArray(coherent_layers, coords={"field": coherent_layers})

        Fields = xr.zeros_like(coherence)

        for co, beam in self.beams:
            kdr = (
                beam.k[{"component": 0}] * x
                + beam.k[{"component": 1}] * y
                + beam.k[{"component": 2}] * z
            )
            ToAdd = (beam.A * xr.ufuncs.exp(1j * kdr)).expand_dims({"field": coherent_layers})
            Fields = Fields + xr.where(ToAdd.field == co, ToAdd, 0)

        return Fields

    def plot(
        self,
        box: Union[float, list[float, float, float], xr.DataArray] = 10,
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
            Each style must contain a "direction" key linked to a dictionnary that will be passed as kwargs 
            for matplotlib's quiver function and a "polar"key that will be passed likewise to the 
            plot function for the polarization ellipse. Defaults to None.
            slider_start (str, optional): The default starting position of the sliders, can be "left" or "center". Defaults to "left".

        Returns:
            tuple[Figure,Axes]
        """
        if isinstance(box, Union[int, float]):
            box = xr.DataArray([box, box, box], coords={"component": [0, 1, 2]})
        elif isinstance(box, list):
            box = xr.DataArray(box, coords={"component": [0, 1, 2]})

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

        # Creating the sliders objects
        slider_dims = []
        dict_coords = {}
        for ind, beam in self.beams:
            k_dims = [
                dim for dim in beam.k.dims if dim not in slider_dims + ["component"]
            ]
            p_dims = [
                dim
                for dim in beam.polar.dims
                if dim not in slider_dims + k_dims + ["Jones", "Component"]
            ]
            dict_coords.update({dim: beam.k.coords[dim] for dim in k_dims})
            dict_coords.update({dim: beam.polar.coords[dim] for dim in p_dims})
            slider_dims += k_dims + p_dims
        sliders = create_sliders_from_dims(
            {dim: dict_coords[dim] for dim in slider_dims}, start=slider_start
        )

        # Initial parameter selections
        initial_sel = {dim: sliders[dim].value for dim in sliders}

        # Functions

        def set_box(ax, box, sel):
            subsel = {dim: val for dim, val in sel.items() if dim in box.dims}
            size_sel = box.sel(subsel, method="nearest")
            ax.set_xlim(-size_sel[0] / 2, size_sel[0] / 2)
            ax.set_ylim(-size_sel[1] / 2, size_sel[1] / 2)
            ax.set_zlim(-size_sel[2] / 2, size_sel[2] / 2)

        def place_beam(
            ax: Axes, beam: Beam, box: xr.DataArray, sel: dict, laser_style: dict
        ):

            k_subsel = {dim: val for dim, val in sel.items() if dim in beam.k.dims}
            p_subsel = {dim: val for dim, val in sel.items() if dim in beam.polar.dims}
            s_subsel = {dim: val for dim, val in sel.items() if dim in box.dims}

            te_subsel = {dim: val for dim, val in sel.items() if dim in beam.TE.dims}
            tm_subsel = {dim: val for dim, val in sel.items() if dim in beam.TM.dims}

            k_sel = beam.k.sel(k_subsel, method="nearest")
            polar_sel = beam.polar.sel(p_subsel, method="nearest")
            size_sel = box.sel(s_subsel, method="nearest")
            TE_sel = beam.TE.sel(te_subsel, method="nearest")
            TM_sel = beam.TM.sel(tm_subsel, method="nearest")

            k_length = (abs(k_sel) ** 2).sum() ** 0.5
            k_dir = k_sel / k_length

            position = -k_dir * size_sel / 2
            dir = ax.quiver(
                position[0],
                position[1],
                position[2],  # base position
                k_dir[0],
                k_dir[1],
                k_dir[2],  # direction
                length=2 * np.pi / float(k_length),
                **laser_style.get("direction", {"colors": "b"}),
            )

            t = np.linspace(0, 1, 100)
            comps = []
            for i in range(3):
                comps += [
                    TE_sel.sel(component=i).item()
                    * np.real(np.exp(1j * 2 * np.pi * t) * polar_sel[0].item())
                    * np.pi
                    / float(k_length)
                    + TM_sel.sel(component=i).item()
                    * np.real(np.exp(1j * 2 * np.pi * t) * polar_sel[1].item())
                    * np.pi
                    / float(k_length)
                    + position[i].item()
                ]
            pol = ax.plot(*comps, **laser_style["polar"])

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
    lattice.plot(size=[7, 7, 2])
    plt.show()
