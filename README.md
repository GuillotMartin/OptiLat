# Optical lattice builder 

This package provides the tools to compute the EM field created by the interferences of multiple laser beams,
the trapping potential it produces, and the time-dependent potentials of a driven lattice.

## Installation

First download the repository and extract it where you want. Then, run in your python environment 

```bash
pip install path/to/package/OptiLat
```

or 

```bash
pip install -e path/to/package/OptiLat
```

if you want to be able to modify it in place.

**N.B** : This package requires the `bloch_schrodinger` package to make use of it's most powerful methods.
It is only imported when needed, so `Beam`, `OptiLat.compute_fields` and the formatting helpers work
without it; only the interactive `OptiLat.plot` method requires it.

## Getting started

Once you have installed the package, open in a jupyter viewer the [Getting Started](docs/GettingStarted.ipynb) notebook.

## Beam profiles

Beams are infinite plane waves by default. Pass a `profile` to give them a finite size, e.g.
`Beam(wavelength=1, direction=[1, 0, 0], profile=GaussianProfile(waist=10, focus=[0, 0, 0]))`.
`GaussianProfile` is a full paraxial gaussian beam (width, wavefront curvature and Gouy phase).
It accepts elliptical waists `[w_TE, w_TM]`, a `diffraction=False` switch that keeps only the
transverse gaussian, and a `power=` normalization. Profiles work everywhere a field is built,
including the time-dependent adapters below. See the [Gaussian beams](docs/GaussianBeams.ipynb) notebook.

## From beams to a trapping potential

`optilat.potentials` turns complex field amplitudes into the AC-Stark shift an atom feels,

$$V = -\frac14\left[\alpha_s|E|^2 + \frac{\alpha_v M}{2F}\,2\,\mathrm{Im}(E_x^*E_y)\right]$$

with `field_on` evaluating a lattice on a potential's own cartesian grid whatever its dimension,
`stark_potential` applying the formula above, and `vector_factor` returning the local degree of
circular polarization on its own.

## Time-dependent lattices

`optilat.dynamics` builds *time-dependent* potentials out of a lattice, for the `PotentialT` and
`AnalyticPotential` classes of the `BECs` package. Give each beam a complex time factor
`c_j(t) = s_j(t) exp(i phi_j(t))` with a `BeamDrive` -- `ramp`, `shake`, `move`, `modulate` and
`pulse` cover the usual experiments, and any plain callable covers the rest -- and the squared field
splits exactly into static pairwise gratings times pure functions of time. Since this is an
algebraic identity rather than a slow-envelope approximation, lattice loading, modulation
spectroscopy, Floquet shaking and moving lattices are all described without approximation. Any drive
argument passed as a `create_parameter` array becomes a parameter dimension the solvers sweep.

Three routes are available, all describing the same potential to round-off. Measured per time step
for three beams with the polarizability tensor's vector part switched on (15 gratings):

| route | 128x128 | 256x256 | when to use it |
|---|---|---|---|
| `add_to_potentialT(..., mode="shapes")` | 4.4 ms | 5.2 ms | the default: parameter dimensions carried by the beams stay sweepable, and the terms compose with the rest of the potential |
| `add_to_potentialT(..., mode="stacked")` | 0.8 ms | 1.1 ms | long runs: one term instead of fifteen, but the beams must carry no parameter dimension of their own |
| `add_to_analytic(...)` | 0.9 ms | 3.8 ms | the rescaling solver, whose grid moves at every step, and grids too large to hold the gratings |

The gratings cost `n_gratings x grid`, so on a large 3D grid the analytic route is the one to reach
for. See the [Time-dependent lattices](docs/TimeDependentLattices.ipynb) notebook.
