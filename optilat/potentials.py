"""Turning the complex field amplitudes of an optical lattice into a trapping potential.

The formulas here are the AC-Stark shift of an atom in a light field, written in the linear form

    V(r) = -1/4 [ alpha_s |E(r)|^2 + alpha_v M/(2F) * 2 Im(E_x*(r) E_y(r)) ]

(the quantization axis being z). Writing the vector part as 2 Im(Ex* Ey) rather than as
C = 2 Im(Ex* Ey)/|E|^2 times |E|^2 keeps the whole potential *linear* in the quadratic forms of the
field, which is what allows the time-dependent machinery of the dynamics module to expand it into
static shapes times pure time functions.
"""

import xarray as xr


def is_zero(value) -> bool:
    """Whether a polarizability-like argument is a plain, exactly-zero scalar.

    Args:
        value: The value to test, either a scalar or a DataArray.

    Returns:
        bool: True only for a scalar equal to 0, so that a DataArray is never reduced to a boolean.
    """
    return not isinstance(value, xr.DataArray) and value == 0


def cartesian_axes(pot) -> dict[str, xr.DataArray]:
    """The cartesian coordinates of a Potential-like object, ready to be passed to compute_fields.

    Args:
        pot: A Potential (or PotentialT, AnalyticPotential, ...) object. Only its 'n_dims',
        'coord_names' and 'coords' attributes are used, so any object exposing them works.

    Returns:
        dict[str, xr.DataArray]: The "x", "y" and "z" coordinates of the potential's grid. The axes
        the potential does not have are set to 0.
    """
    axes = {name: 0 for name in ("x", "y", "z")}
    axes.update({pot.coord_names[i]: pot.coords[i] for i in range(pot.n_dims)})
    return axes


def field_on(lattice, pot) -> xr.DataArray:
    """Evaluate the complex field amplitudes of a lattice over a potential's own grid.

    Args:
        lattice (OptiLat): The optical lattice to evaluate.
        pot: A Potential-like object, whose cartesian grid is used. A 1D or 2D potential simply gets
        the missing coordinates set to 0.

    Returns:
        xr.DataArray: The "Fields" array returned by OptiLat.compute_fields, over the grid of 'pot'.
    """
    return lattice.compute_fields(**cartesian_axes(pot))


def intensity(fields: xr.DataArray, sum_fields: bool = True) -> xr.DataArray:
    """The light intensity |E|^2 of a field array, in the same arbitrary units as the amplitudes.

    Args:
        fields (xr.DataArray): The complex amplitudes, with a "component" dimension and, usually, a
        "field" dimension of incoherent contributions.
        sum_fields (bool, optional): Whether to sum the incoherent contributions of the "field"
        dimension. Defaults to True.

    Returns:
        xr.DataArray: The intensity.
    """
    I = (abs(fields) ** 2).sum("component")
    if sum_fields and "field" in I.dims:
        I = I.sum("field")
    return I


def vector_numerator(fields: xr.DataArray, sum_fields: bool = True) -> xr.DataArray:
    """The quantity 2 Im(E_x* E_y) driving the polarization-dependent part of the potential.

    Args:
        fields (xr.DataArray): The complex amplitudes, with a "component" dimension.
        sum_fields (bool, optional): Whether to sum the incoherent contributions of the "field"
        dimension. Defaults to True.

    Returns:
        xr.DataArray: 2 Im(E_x* E_y), which is the local degree of circular polarization around z
        times the intensity.
    """
    vec = 2 * (fields.isel(component=0).conj() * fields.isel(component=1)).imag
    if sum_fields and "field" in vec.dims:
        vec = vec.sum("field")
    return vec


def vector_factor(fields: xr.DataArray, sum_fields: bool = True) -> xr.DataArray:
    """The local degree of circular polarization C = 2 Im(E_x* E_y) / |E|^2 around the z axis.

    C runs from -1 (sigma-) through 0 (linear) to +1 (sigma+), and is the quantity to plot to see
    the polarization pattern of a lattice on its own.

    Args:
        fields (xr.DataArray): The complex amplitudes, with a "component" dimension.
        sum_fields (bool, optional): Whether to sum the incoherent contributions of the "field"
        dimension. Defaults to True.

    Returns:
        xr.DataArray: The degree of circular polarization.
    """
    return vector_numerator(fields, sum_fields) / intensity(fields, sum_fields)


def stark_potential(
    fields: xr.DataArray,
    alpha_s: float | xr.DataArray = 1.0,
    alpha_v: float | xr.DataArray = 0.0,
    M: float | xr.DataArray = 0.0,
    F: float | xr.DataArray = 1.0,
    sum_fields: bool = True,
) -> xr.DataArray:
    """The AC-Stark potential seen by an atom in the light field of an optical lattice.

    V = -1/4 [ alpha_s |E|^2 + alpha_v M/(2F) * 2 Im(E_x* E_y) ], with z as quantization axis. The
    polarizabilities are taken in arbitrary units, consistent with those of the beam amplitudes.

    Args:
        fields (xr.DataArray): The complex amplitudes, as returned by OptiLat.compute_fields.
        alpha_s (Union[float,xr.DataArray], optional): The scalar polarizability. Defaults to 1.
        alpha_v (Union[float,xr.DataArray], optional): The vector polarizability. Defaults to 0,
        which leaves out the polarization-dependent part entirely.
        M (Union[float,xr.DataArray], optional): The spin projection on the quantization axis.
        Defaults to 0.
        F (Union[float,xr.DataArray], optional): The total angular momentum. Defaults to 1.
        sum_fields (bool, optional): Whether to sum the incoherent contributions of the "field"
        dimension. Defaults to True.

    Returns:
        xr.DataArray: The potential.
    """
    V = alpha_s * intensity(fields, sum_fields)
    if not (is_zero(alpha_v) or is_zero(M)):
        V = V + alpha_v * M / (2 * F) * vector_numerator(fields, sum_fields)
    return -0.25 * V
