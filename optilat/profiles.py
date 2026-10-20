"""Lateral profiles of laser beams, turning a plane wave into a finite-size beam.

A beam with a profile has the field A u(r) exp(i k.r), where u is a complex scalar envelope that
varies slowly on the scale of the wavelength. Its polarization stays transverse, which is the
paraxial approximation. The envelope is written in the beam's own frame:

    s  = (r - focus) . direction   the distance from the focus along the propagation direction
    t1 = (r - focus) . TE          the transverse coordinate along the TE unit vector
    t2 = (r - focus) . TM          the transverse coordinate along the TM unit vector

The envelopes only use numpy operations, so the same code evaluates them on xarray DataArrays (in
OptiLat.compute_fields and dynamics.expand) and on plain numpy or cupy arrays (in the adapters
module). Every DataArray attribute of a profile is one of its inputs, which is what lets the base
class find and select the parameter dimensions of any subclass without it having to declare them.
"""

import numpy as np
import xarray as xr
from typing import Union

from optilat.builder import format_3dvec, treal, vec3d
from optilat.parameters import collect_dims, select_attributes


class Profile:
    """The base class of every lateral profile.

    A profile is centred on its focus, and subclasses implement 'envelope' in the beam frame.
    """

    def __init__(self, focus: vec3d = None):
        """Initialize the profile's focus.

        Args:
            focus (vec3d, optional): The position of the focus, as a list of 3 scalars or a
            DataArray with a size-3 "component" dimension. Defaults to the origin.
        """
        self.focus: xr.DataArray = format_3dvec([0, 0, 0] if focus is None else focus)

    def envelope(self, s, t1, t2, kl):
        """The complex envelope u in the beam frame.

        Args:
            s: The distance from the focus along the propagation direction.
            t1: The transverse coordinate along the TE unit vector.
            t2: The transverse coordinate along the TM unit vector.
            kl: The beam's wavenumber.

        Returns:
            The complex envelope, with the type of the inputs (DataArray or numpy array).
        """
        raise NotImplementedError

    @property
    def param_coords(self) -> dict[str, xr.DataArray]:
        """The parameter dimensions carried by the profile's inputs, with their coordinates."""
        return collect_dims(*vars(self).values())

    def sel(self, selection: dict, method: str = "nearest") -> "Profile":
        """The profile at a single point of its parameter space.

        Args:
            selection (dict): The selected label of each parameter dimension. The dimensions the
            profile does not carry are ignored.
            method (str, optional): The xarray selection method. Defaults to "nearest".

        Returns:
            Profile: A copy of the profile, reduced to the selected point.
        """
        return select_attributes(self, selection, method)


class GaussianProfile(Profile):
    def __init__(
        self,
        waist: Union[treal, list[treal]],
        focus: vec3d = None,
        diffraction: bool = True,
        power: treal = None,
    ):
        """A gaussian beam (TEM00 mode) in the paraxial approximation.

        With the exp(+i k.r) convention of the package, the envelope is the product over the two
        transverse axes a = TE, TM of

            (1 + i s/zR_a)^(-1/2) exp(-t_a^2 / (w_a^2 (1 + i s/zR_a))),    zR_a = kl w_a^2 / 2

        which contains the beam width w(s), the wavefront curvature and the Gouy phase. For a round
        beam it reduces to 1/(1 + i s/zR) exp(-rho^2 / (w0^2 (1 + i s/zR))).

        Args:
            waist (Union[treal, list[treal]]): The 1/e^2 intensity radius at the focus. Either a
            single value for a round beam, or a pair [w_TE, w_TM] for an elliptical beam, with the
            waists along the beam's TE and TM unit vectors.
            focus (vec3d, optional): The position of the focus. Defaults to the origin.
            diffraction (bool, optional): Whether to include the propagation of the beam (width,
            curvature and Gouy phase). If False, the envelope is only the transverse gaussian
            exp(-t1^2/w_TE^2 - t2^2/w_TM^2), which is cheaper and valid well within the Rayleigh
            range. Defaults to True.
            power (treal, optional): If given, the envelope is scaled by sqrt(2 P / (pi w_TE w_TM)),
            so that the beam carries the power P (the integral of |E|^2 over a transverse plane, in
            the arbitrary units of the package). The beam's amplitude then only multiplies this, so
            keep it at 1. If None, the envelope is 1 at the focus and the beam's amplitude is the
            peak field. Defaults to None.
        """
        super().__init__(focus)
        if isinstance(waist, (list, tuple)):
            if len(waist) != 2:
                raise ValueError("An elliptical waist must be a pair [w_TE, w_TM].")
            self.waist_TE, self.waist_TM = waist
        else:
            self.waist_TE = self.waist_TM = waist
        self.diffraction = diffraction
        self.power = power

    def __repr__(self):
        return (
            f"GaussianProfile(waist=({self.waist_TE}, {self.waist_TM}), "
            f"diffraction={self.diffraction}, power={self.power})"
        )

    def rayleigh_range(self, kl: treal) -> tuple[treal, treal]:
        """The Rayleigh ranges zR = kl w^2 / 2 along the TE and TM axes.

        Args:
            kl (treal): The beam's wavenumber.
        """
        return kl * self.waist_TE**2 / 2, kl * self.waist_TM**2 / 2

    def width(self, s, kl: treal) -> tuple:
        """The 1/e^2 intensity radii w(s) = w0 sqrt(1 + (s/zR)^2) along the TE and TM axes.

        Args:
            s: The distance from the focus along the propagation direction.
            kl (treal): The beam's wavenumber.
        """
        if not self.diffraction:
            return self.waist_TE + 0 * s, self.waist_TM + 0 * s
        zR_TE, zR_TM = self.rayleigh_range(kl)
        return (
            self.waist_TE * np.sqrt(1 + (s / zR_TE) ** 2),
            self.waist_TM * np.sqrt(1 + (s / zR_TM) ** 2),
        )

    def envelope(self, s, t1, t2, kl):
        if self.diffraction:
            u = 1
            for t, w in ((t1, self.waist_TE), (t2, self.waist_TM)):
                q = 1 + 1j * s / (kl * w**2 / 2)  # the reduced complex beam parameter
                u = u * q**-0.5 * np.exp(-(t**2) / (w**2 * q))
        else:
            u = np.exp(-(t1**2) / self.waist_TE**2 - t2**2 / self.waist_TM**2) + 0j
        if self.power is not None:
            u = u * np.sqrt(2 * self.power / (np.pi * self.waist_TE * self.waist_TM))
        return u
