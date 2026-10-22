"""The three routes handing a driven lattice to the BECs potentials, with parameter dimensions."""

import itertools
import warnings

import numpy as np
import pytest
import xarray as xr

potentialT = pytest.importorskip("BECs.potentialT")
create_parameter = pytest.importorskip("bloch_schrodinger.potential").create_parameter
frames = pytest.importorskip("bloch_schrodinger.frames")

from optilat import (
    Beam,
    BeamDrive,
    GaussianProfile,
    LatticeDrive,
    LatticeFunction,
    OptiLat,
    add_to_analytic,
    add_to_potentialT,
    field_on,
    stark_potential,
)

AnalyticPotential, PotentialT = potentialT.AnalyticPotential, potentialT.PotentialT

BOX = [[2, 0], [0, 2]]
RES = (24, 24)
ANGLES = [np.pi / 2 + 2 * np.pi / 3 * i for i in range(3)]

theta = create_parameter("theta", np.array([0.0, 0.6]))
waist = create_parameter("waist", np.array([1.5, 3.0]))
nu = create_parameter("nu", np.array([1.0, 2.5]))
alpha = create_parameter("alpha", np.array([1.0, 2.0]))

STARK = dict(alpha_s=alpha, alpha_v=0.4, M=3, F=3)
SWEEP = {"theta": theta, "waist": waist, "nu": nu, "alpha": alpha}


def honeycomb(polar_angle=theta, w=waist):
    lattice = OptiLat()
    lattice.add_beam(
        [
            Beam(
                wavelength=0.83,
                direction=[np.cos(a), np.sin(a), 0],
                polar=[np.cos(polar_angle), np.sin(polar_angle)],
                profile=GaussianProfile(w),
            )
            for a in ANGLES
        ],
        0,
    )
    return lattice


def shaken(lattice):
    drive = LatticeDrive(lattice)
    drive[[1, 2]] = BeamDrive.shake(amp=0.6, freq=nu)
    return drive


@pytest.fixture(scope="module")
def routes():
    """The same driven lattice, through the analytic, shapes and stacked routes."""
    warnings.simplefilter("ignore")
    lattice = honeycomb()
    drive = shaken(lattice)

    analytic = AnalyticPotential(BOX, RES)
    add_to_analytic(analytic, lattice, drive, **STARK)
    shapes = PotentialT(BOX, RES, v0=0)
    add_to_potentialT(shapes, lattice, drive, **STARK)
    stacked = PotentialT(BOX, RES, v0=0)
    add_to_potentialT(stacked, lattice, drive, mode="stacked", **STARK)
    return analytic, shapes, stacked


def selections():
    for values in itertools.product(*(p.values for p in SWEEP.values())):
        yield dict(zip(SWEEP, values))


def test_every_dimension_is_registered(routes):
    for pot in routes:
        assert set(pot.param_coords) == set(SWEEP)


def test_the_fast_routes_register_functions_that_select_themselves(routes):
    """The lattice's own dims reach the function through 'sel', not as fake keyword parameters."""
    analytic, _, stacked = routes
    for pot, name in ((analytic, "lat"), (stacked, "lat_sum")):
        entry = pot.funcs[name]
        assert potentialT.selects_itself(entry["func"])
        assert set(entry["selects"]) == {"theta", "waist", "alpha"}
        # only the drive's own parameter is handed over as a keyword
        assert set(entry["parameters"]) | set(entry["dims"]) == {"nu"}


def test_solver_path_agrees_across_routes(routes):
    analytic, shapes, stacked = routes
    for sel in selections():
        reference = np.asarray(shapes.make_Vt(sel)(0.31))
        np.testing.assert_allclose(np.asarray(analytic.make_V(sel)(0.31)), reference, atol=1e-10)
        np.testing.assert_allclose(np.asarray(stacked.make_Vt(sel)(0.31)), reference, atol=1e-10)


@pytest.mark.parametrize("when", [dict(t=0.2), dict(t_coord=(0.0, 1.0, 3))])
def test_labelled_path_agrees_across_routes(routes, when):
    analytic, shapes, stacked = routes
    reference = shapes.to_potential(**when).V
    for pot in (analytic, stacked):
        V = pot.to_potential(**when).V
        assert set(V.dims) == set(reference.dims)
        assert float(abs(V - reference).max()) < 1e-10


def test_labelled_matches_solver_path(routes):
    analytic = routes[0]
    V = analytic.to_potential(t=0.2).V
    for sel in itertools.islice(selections(), 4):
        np.testing.assert_allclose(
            V.sel(sel).transpose("a1", "a2").values,
            np.asarray(analytic.make_V(sel)(0.2)),
            atol=1e-10,
        )


def test_static_lattice_matches_compute_fields():
    """The analytic route against the plain xarray evaluation, which shares none of its code."""
    lattice = honeycomb()
    analytic = AnalyticPotential(BOX, RES)
    add_to_analytic(analytic, lattice, **STARK)

    for sel in itertools.islice(selections(), 8):
        stark = dict(STARK, alpha_s=sel["alpha"])
        reference = stark_potential(field_on(lattice.sel(sel), analytic), **stark)
        np.testing.assert_allclose(
            np.asarray(analytic.make_V(sel)(0.0)),
            reference.transpose("a1", "a2").values,
            atol=1e-12,
        )


def test_broadcast_coordinates_match_the_dense_grid():
    """A solver may hand over 1-D coordinates shaped to broadcast against each other."""
    lattice = honeycomb()
    function = LatticeFunction(lattice, shaken(lattice), **STARK)
    analytic = AnalyticPotential(BOX, RES)
    at = function.sel({"theta": 0.6, "waist": 3.0, "alpha": 2.0})

    dense = at(0.4, *(np.asarray(c) for c in analytic.coords), nu=2.5)
    rank1 = at(0.4, *analytic.coords_1d(), nu=2.5)
    labelled = at(0.4, *analytic.coords, nu=2.5)

    np.testing.assert_allclose(rank1, dense, atol=1e-12)
    assert isinstance(labelled, xr.DataArray) and labelled.dims == ("a1", "a2")
    np.testing.assert_allclose(labelled.values, dense, atol=1e-12)


def test_plain_lattice_without_parameters():
    lattice = honeycomb(polar_angle=0.4, w=2.0)
    drive = LatticeDrive(lattice)
    drive[:] = BeamDrive.ramp(vi=0.0, vf=1.0, ti=0.0, tf=1.0)

    analytic = AnalyticPotential(BOX, RES)
    function = add_to_analytic(analytic, lattice, drive, alpha_s=1.0)
    assert function.param_coords == {} and analytic.param_coords == {}

    shapes = PotentialT(BOX, RES, v0=0)
    add_to_potentialT(shapes, lattice, drive, alpha_s=1.0)
    for when in (dict(t=0.5), dict(t_coord=(0.0, 1.0, 3))):
        assert float(abs(analytic.to_potential(**when).V - shapes.to_potential(**when).V).max()) < 1e-12


def test_composes_with_other_parametrized_terms():
    """A parameter-free lattice next to a swept trap: their results must combine by dimension name."""
    lattice = honeycomb(polar_angle=0.4, w=2.0)
    omega = create_parameter("omega", np.array([1.0, 2.0, 3.0]))
    analytic = AnalyticPotential(BOX, RES)
    add_to_analytic(analytic, lattice, term=False)
    analytic.add_function(
        "trap", lambda t, x, y, omega: omega**2 * (x**2 + y**2), parameters={"omega": omega}
    )
    analytic.add_term("lat + trap")

    V = analytic.to_potential(t=0.0).V
    lat = stark_potential(field_on(lattice, analytic))
    for w in omega.values:
        trap = w**2 * (analytic.coords[0] ** 2 + analytic.coords[1] ** 2)
        assert float(abs(V.sel(omega=w) - (lat + trap)).max()) < 1e-12
    np.testing.assert_allclose(
        np.asarray(analytic.make_V({"omega": 2.0})(0.0)),
        (lat + 4.0 * (analytic.coords[0] ** 2 + analytic.coords[1] ** 2)).values,
        atol=1e-12,
    )


def test_term_false_only_registers_the_function():
    analytic = AnalyticPotential(BOX, RES)
    add_to_analytic(analytic, honeycomb(), term=False)
    assert "lat" in analytic.funcs and "lat" not in analytic.terms.values()


def test_missing_dimension_raises():
    function = LatticeFunction(honeycomb())
    with pytest.raises(ValueError, match="theta"):
        function.sel({"waist": 1.5})


def test_plain_coordinates_need_a_reduced_function():
    """Unreduced, the result keeps the lattice's dims, which plain arrays cannot be kept apart from."""
    function = LatticeFunction(honeycomb())
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="sel"):
        function(0.0, x, x)
    # a lattice without dims has nothing to keep apart, and is simply summed
    plain = LatticeFunction(honeycomb(polar_angle=0.4, w=2.0))
    np.testing.assert_allclose(plain(0.0, x, x), plain.sel({})(0.0, x, x), atol=1e-12)


def test_conflicting_drive_coordinate_raises():
    lattice = honeycomb()
    drive = LatticeDrive(lattice)
    drive[0] = BeamDrive.move(detuning=create_parameter("theta", np.array([0.0, 1.0, 2.0])))
    with pytest.raises(ValueError, match="theta"):
        LatticeFunction(lattice, drive)


def test_cupy_coordinates():
    cp = pytest.importorskip("cupy")
    lattice = honeycomb()
    function = LatticeFunction(lattice, shaken(lattice), **STARK)
    x, y = AnalyticPotential(BOX, RES).coords_1d()
    at = function.sel({"theta": 0.0, "waist": 1.5, "alpha": 1.0})

    on_device = at(0.2, cp.asarray(x), cp.asarray(y), nu=1.0)
    assert isinstance(on_device, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(on_device), at(0.2, x, y, nu=1.0), atol=1e-12)


def test_ssfm_sweeps_beam_parameters():
    ssfm = pytest.importorskip("BECs.ssfm")
    small = (16, 16)
    lattice = honeycomb(w=2.0)
    analytic = AnalyticPotential(BOX, small)
    add_to_analytic(analytic, lattice, shaken(lattice), alpha_s=5.0)

    psi0 = xr.ones_like(analytic.V, dtype=complex) / (analytic.n_elems * analytic.get_dS()) ** 0.5
    solver = ssfm.SSFM(analytic, psi0, g=0.0)
    t_samples = create_parameter("t", np.linspace(0, 0.02, 3))
    psi = solver.solve(0.0, 0.02, t_samples, dt0=1e-3, tol=1e-6, verbose=False, device="cpu")

    assert {"theta", "nu"} <= set(psi.dims)
    norm = (abs(psi) ** 2).sum(["a1", "a2"]) * analytic.get_dS()
    # The adaptive step controller lets the norm drift by about 'tol', whatever the potential
    assert float(abs(norm - 1).max()) < 1e-5
    # The polarization angle changes the potential, so the runs must differ
    assert float(abs(psi.isel(theta=0) - psi.isel(theta=1)).isel(t=-1).max()) > 1e-6


def test_sample_on_follows_a_moving_grid():
    """The whole chain: a swept lattice drawn over a field whose grid stretches with time.

    The potential is evaluated frame by frame on the field's own, factored, moving coordinates, at
    every point of the lattice's own sweep, exactly as the reduced function sums the beams there.
    """
    lattice = honeycomb()
    analytic = AnalyticPotential(BOX, RES)
    function = add_to_analytic(analytic, lattice, shaken(lattice), **STARK)

    tv = np.linspace(0, 1, 3)
    lam = xr.DataArray(1 + 0.5 * tv, dims="t", coords={"t": tv})
    rho_x, rho_y = xr.broadcast(analytic.x, analytic.y)
    field = (xr.ones_like(rho_x) * (1 + 0 * lam)).transpose("t", "a1", "a2")
    field = field.drop_vars(["x", "y"], errors="ignore").assign_coords(
        rho_x=rho_x.drop_vars(["x", "y"], errors="ignore"),
        rho_y=rho_y.drop_vars(["x", "y"], errors="ignore"),
        lambda_x=lam,
        lambda_y=lam,
    )
    field = frames.set_cart_factors(field, {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")})

    V = analytic.sample_on(field)
    assert set(V.dims) == {"t", "theta", "waist", "nu", "alpha", "a1", "a2"}
    for t in (0.0, 1.0):
        point = {"theta": 0.6, "waist": 3.0, "alpha": 2.0}
        x = (field.rho_x * lam.sel(t=t)).transpose("a1", "a2").values
        y = (field.rho_y * lam.sel(t=t)).transpose("a1", "a2").values
        np.testing.assert_allclose(
            V.sel(t=t, nu=2.5, **point).transpose("a1", "a2").values,
            function.sel(point)(t, x, y, nu=2.5),
            atol=1e-12,
        )
