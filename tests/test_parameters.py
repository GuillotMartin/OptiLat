"""Parameter dimensions carried by beams, profiles and lattices."""

import numpy as np
import pytest
import xarray as xr

from optilat import Beam, GaussianProfile, OptiLat


def parameter(name, values):
    """A parameter dimension, as bloch_schrodinger.potential.create_parameter builds it."""
    values = np.asarray(values)
    return xr.DataArray(values, coords={name: values}, name=name)


def assert_same_beam(a: Beam, b: Beam):
    for attr in ("k", "A", "TE", "TM", "direction"):
        np.testing.assert_allclose(getattr(a, attr).values, getattr(b, attr).values, atol=1e-14)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: Beam(wavelength=0.83, direction=[1, 0, 0], polar=[np.cos(p), np.sin(p)]),
        lambda p: Beam(wavelength=0.5 + p, direction=[1, 1, 0]),
        lambda p: Beam(wavelength=1.0, direction=[1, p, 0.2], amplitude=1 + 1j * p),
        lambda p: Beam(wavelength=1.0, direction=[0, 0, 1], polar=[1, np.exp(1j * p)]),
    ],
    ids=["polarization", "wavelength", "direction-amplitude", "vertical"],
)
def test_beam_sel_matches_rebuilt(build):
    p = parameter("p", [0.0, 0.3, 0.9])
    swept = build(p)
    assert list(swept.param_coords) == ["p"]
    for value in p.values:
        assert_same_beam(swept.sel({"p": value}), build(value))


def test_profile_sel_matches_rebuilt():
    waist = parameter("waist", [2.0, 4.0])
    x = xr.DataArray(np.linspace(-5, 5, 11), dims="x")
    swept = Beam(wavelength=1.0, direction=[0, 0, 1], profile=GaussianProfile(waist, power=1.0))
    assert list(swept.param_coords) == ["waist"]

    for w in waist.values:
        rebuilt = Beam(wavelength=1.0, direction=[0, 0, 1], profile=GaussianProfile(w, power=1.0))
        np.testing.assert_allclose(
            swept.sel({"waist": w}).envelope(x=x, z=3.0).values,
            rebuilt.envelope(x=x, z=3.0).values,
        )


def test_param_coords_merges_every_input():
    theta = parameter("theta", [0.0, 0.5])
    waist = parameter("waist", [2.0, 3.0])
    focus = parameter("dz", [0.0, 1.0])
    lattice = OptiLat()
    lattice.add_beam(
        [
            Beam(wavelength=1.0, direction=[1, 0, 0], polar=[np.cos(theta), np.sin(theta)]),
            Beam(
                wavelength=1.0,
                direction=[-1, 0, 0],
                profile=GaussianProfile(waist, focus=[0, 0, focus]),
            ),
        ],
        0,
    )
    assert set(lattice.param_coords) == {"theta", "waist", "dz"}
    np.testing.assert_array_equal(lattice.param_coords["waist"].values, waist.values)


def test_param_coords_needs_coordinates():
    lattice = OptiLat()
    lattice.add_beam(
        Beam(wavelength=1.0, direction=[1, 0, 0], polar=[xr.DataArray([1.0, 0.5], dims="q"), 0])
    )
    with pytest.raises(ValueError, match="no coordinate"):
        _ = lattice.param_coords


def test_param_coords_rejects_conflicting_coordinates():
    lattice = OptiLat()
    lattice.add_beam(
        [
            Beam(wavelength=parameter("lamb", [0.8, 0.9]), direction=[1, 0, 0]),
            Beam(wavelength=parameter("lamb", [0.8, 1.0]), direction=[-1, 0, 0]),
        ]
    )
    with pytest.raises(ValueError, match="lamb"):
        _ = lattice.param_coords


def test_lattice_sel_keeps_field_indexes_and_ignores_unknown_dims():
    theta = parameter("theta", [0.0, 0.5])
    lattice = OptiLat()
    lattice.add_beam(Beam(wavelength=1.0, direction=[1, 0, 0], polar=[np.cos(theta), 0]), 0)
    lattice.add_beam(Beam(wavelength=1.0, direction=[0, 1, 0]), 3)

    selected = lattice.sel({"theta": 0.5, "unrelated": 2.0})
    assert [index for index, _ in selected.beams] == [0, 3]
    assert selected.maxIndex == lattice.maxIndex
    assert selected.param_coords == {}
    # The original lattice is left untouched
    assert set(lattice.param_coords) == {"theta"}
