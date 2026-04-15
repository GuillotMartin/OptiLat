import numpy as np
import xarray as xr
from typing import Union

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
            self.k = (
                self.kl * direction
            )  # The k-vector is formatted as a DataArray with a size-3 "component" dimension

        else:
            if isinstance(k, Union[int, float]):
                self.kl = k
                self.k = k * direction
            else:
                self.k = format_3dvec(k)
                self.kl = (self.k**2).sum("component") ** 0.5

        self.direction = self.k / self.kl

    def __repr__(self):
        return f"A beam with k-vector: {self.kl}, \ndirection {self.direction} \nand polarization {self.polar}"

    def compute_Camplitude(self) -> xr.DataArray:
        """Returns the complex amplitude (Ax, Ay, Az) of the beam. The full EM-field produced can then be written as Ei = Re[Ai exp(1j * (k.r - w.t))]

        Returns:
            xr.DataArray: Complex amplitude (Ax, Ay, Az)
        """

        TE = xr.zeros_like(self.direction)  # First vector orthogonal to k
        TM = xr.zeros_like(self.direction)  # Second vector orthogonal to k

        dx = self.direction[{"component": 0}]
        dy = self.direction[{"component": 1}]
        dz = self.direction[{"component": 2}]

        # The TE vector is contained in the xy-plane
        TE[{"component": 0}] = xr.where((xr.ufuncs.equal(dx, 0) + xr.ufuncs.equal(dy, 0)), dz, -dy)
        TE[{"component": 1}] = xr.where((xr.ufuncs.equal(dx, 0) + xr.ufuncs.equal(dy, 0)), 0, dx)

        # The second vector is determined by the cross-product of k and TE
        TM[{"component": 0}] = xr.where((xr.ufuncs.equal(dx, 0) + xr.ufuncs.equal(dy, 0)), 0, -dx * dz)
        TM[{"component": 1}] = xr.where((xr.ufuncs.equal(dx, 0) + xr.ufuncs.equal(dy, 0)), dz, -dy * dz)
        TM[{"component": 2}] = dy**2 + dx**2

            

        # Complex amplitude A
        A = xr.zeros_like(self.direction, dtype=complex)
        A = A + TE * self.polar[{"Jones": 0}] * self.amplitude
        A = A + TM * self.polar[{"Jones": 1}] * self.amplitude

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
            beams = list(beam)
        else:
            beams = beam
        if isinstance(index, int):
            indexes = list(index)
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

            Camp = beam.compute_Camplitude()

            Fields = Fields + (Camp * xr.ufuncs.exp(1j * kdr)).assign_coords(
                {"field": co}
            )

        return Fields
