"""Parameter dimensions: the extra dimensions the inputs of a lattice may carry, besides space.

Any input of a Beam (amplitude, wavelength, direction, polarization), of its lateral profile, or a
polarizability can be given as a DataArray carrying one or several parameter dimensions, typically
built with bloch_schrodinger.potential.create_parameter. The xarray routes simply broadcast them.
The routes that evaluate the lattice in plain numpy (or cupy) for the BECs solvers instead reduce the
lattice to a single point of the parameter space at a time. The helpers below are shared by both:
they collect the parameter dimensions of an object, select a point, and turn the selected values
into plain numbers.

The numpy routes hand the BECs potentials a function that carries the lattice's parameter
dimensions itself, in 'param_coords', and reduces itself to a point of them with 'sel'. The potential
hands it the *label* of the point a run sits at, and the function selects the lattice there itself.
This works whatever the arrays are made of: a derived, unnamed or multi-dimensional input such as
[cos(theta), sin(theta) exp(i phi)] is reduced exactly like a bare parameter.
"""

import copy

import numpy as np
import xarray as xr

# The dimensions that describe the inputs themselves rather than a parameter space
INTERNAL_DIMS = ("component", "Jones")


def _same_coord(a: xr.DataArray, b: xr.DataArray) -> bool:
    """Whether two coordinates hold the same values."""
    return a.shape == b.shape and bool(np.array_equal(a.values, b.values))


def merge_dims(*dicts: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Merge several dictionaries of parameter coordinates, keyed by dimension name.

    Raises:
        ValueError: If two dictionaries give the same dimension different coordinates.

    Returns:
        dict[str, xr.DataArray]: The merged coordinates, in order of first appearance.
    """
    merged: dict[str, xr.DataArray] = {}
    for dims in dicts:
        for dim, coord in dims.items():
            if dim in merged and not _same_coord(merged[dim], coord):
                raise ValueError(
                    f"The parameter dimension '{dim}' is given two different coordinates. "
                    "Parameters are matched by name, so give one of them a different name."
                )
            merged[dim] = coord
    return merged


def collect_dims(*values, skip: tuple[str, ...] = INTERNAL_DIMS) -> dict[str, xr.DataArray]:
    """The parameter dimensions carried by some values, with their coordinates.

    Args:
        *values: The values to inspect. Only DataArrays carry dimensions, anything else is ignored.
        skip (tuple[str, ...], optional): The dimensions that are not parameters. Defaults to the
        "component" and "Jones" dimensions of the vectors.

    Raises:
        ValueError: If a parameter dimension has no coordinate values, or if two values give the
        same dimension different coordinates.

    Returns:
        dict[str, xr.DataArray]: One plain 1-D coordinate per parameter dimension, named after it.
    """
    dims: dict[str, xr.DataArray] = {}
    for value in values:
        if not isinstance(value, xr.DataArray):
            continue
        for dim in value.dims:
            if dim in skip:
                continue
            if dim not in value.coords:
                raise ValueError(
                    f"The '{dim}' dimension has no coordinate values, so no point of it can be "
                    "selected. Give it coordinates, e.g. with "
                    "bloch_schrodinger.potential.create_parameter."
                )
            data = value.coords[dim].values
            coord = xr.DataArray(data, coords={dim: data}, dims=dim, name=dim)
            dims = merge_dims(dims, {dim: coord})
    return dims


def select(value, selection: dict, method: str = "nearest"):
    """Reduce a value to a point of the parameter space.

    Args:
        value: The value, a DataArray or anything else, which is returned as it is.
        selection (dict): The selected label of each dimension. The dimensions the value does not
        carry are ignored.
        method (str, optional): The xarray selection method. Defaults to "nearest".
    """
    if not isinstance(value, xr.DataArray):
        return value
    sub = {dim: label for dim, label in selection.items() if dim in value.dims}
    return value.sel(sub, method=method) if sub else value


def select_attributes(obj, selection: dict, method: str = "nearest"):
    """A shallow copy of an object, with every DataArray attribute reduced to the selected point."""
    new = copy.copy(obj)
    for attr, value in vars(obj).items():
        if isinstance(value, xr.DataArray):
            setattr(new, attr, select(value, selection, method))
    return new


def to_number(value):
    """A value reduced to a single point, as a plain number.

    Raises:
        ValueError: If the value still carries a dimension.
    """
    if isinstance(value, (xr.DataArray, np.ndarray)):
        if value.ndim:
            raise ValueError(f"Expected a single value, got an array of shape {value.shape}.")
        return value.item()
    return value


def scalarize(obj):
    """A shallow copy of an object whose DataArray attributes are replaced by plain numbers, or by
    numpy arrays for those that are vectors, so that numpy and cupy code can evaluate it."""
    new = copy.copy(obj)
    for attr, value in vars(obj).items():
        if isinstance(value, xr.DataArray):
            setattr(new, attr, value.item() if value.ndim == 0 else value.values)
    return new


def array_module(*arrays):
    """The array module the arrays live in: cupy if any of them is a cupy array, numpy otherwise.

    cupy is only imported when one of its arrays is found, so it stays an optional dependency.
    """
    for array in arrays:
        if type(array).__module__.split(".")[0] == "cupy":
            import cupy

            return cupy
    return np
