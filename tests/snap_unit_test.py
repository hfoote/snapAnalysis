# unit tests for base methods of snapshot objects.
# Requires the example snapshot in example_snaps/snap_000.hdf5

import numpy as np
from snapanalysis.snap import snapshot
import astropy.units as u
import pytest


@pytest.fixture
def dm_snap():
    s = snapshot("tests/example_snaps/snap_000.hdf5", 1)
    return s


@pytest.fixture
def star_snap():
    s = snapshot("tests/example_snaps/snap_000.hdf5", 4)
    return s


## tests for I/O
@pytest.mark.parametrize(
    "snap, expected",
    [
        ("dm_snap", {"fdm": False, "time": 0.0 * u.Gyr, "box_size": 0.0, "N": 10000}),
        ("star_snap", {"fdm": False, "time": 0.0 * u.Gyr, "box_size": 0.0, "N": 0}),
    ],
)
def test_can_read_metadata(snap, expected, request):
    s = request.getfixturevalue(snap)
    assert s.fdm == expected["fdm"], "DM type detection failed!"
    assert s.time == expected["time"], "Time detection failed!"
    assert s.box_size == expected["box_size"], "Box size detction failed!"
    assert s.N == expected["N"], "Particle number detection failed!"


def test_unit_detection(dm_snap):
    print(dm_snap)
     # define tolerance - 1 part in 1000 should account for rounding differences 
     # between unit specification in astropy vs gadget
    unit_tol = 1e-3 
    assert (dm_snap.field_units["Coordinates"] - 1.0 * u.kpc) / (
        1.0 * u.kpc
    ) < unit_tol, "Length unit detection failed!"
    assert (dm_snap.field_units["Velocities"] - 1.0 * u.km / u.s) / (
        1.0 * u.km / u.s
    ) < unit_tol, "Velocity unit detection failed!"
    assert (dm_snap.field_units["Masses"] - 1e10 * u.Msun) / (
        1e10 * u.Msun
    ) < unit_tol, "Mass unit detection failed!"


def test_can_read_field(dm_snap):
    assert np.allclose(
        dm_snap.read_field("Coordinates")[0],
        np.array([-34.63336945, -66.06946564, 172.63697815]),
    ), "Particle data reading failed!"


def test_can_read_masstable(dm_snap):
    assert dm_snap.read_masstable() == 0.001799996432347745, "Failed to read masstable!"


def test_can_select_particles_by_id_range(dm_snap):
    min_id = 1000
    max_id = 2000
    dm_snap.select_particles((min_id, max_id))
    assert (dm_snap.data_fields["ParticleIDs"] >= min_id).all(), (
        "Particle ID selection lower limit failed!"
    )
    assert (dm_snap.data_fields["ParticleIDs"] <= max_id).all(), (
        "Particle ID selection upper limit failed!"
    )
    assert len(dm_snap.data_fields["ParticleIDs"]) == (max_id - min_id + 1), (
        "Particle ID selection failed!"
    )


def test_can_select_particles_with_boolean_mask(dm_snap):
    dm_snap.load_particle_data(["ParticleIDs", "Coordinates"])
    mask = np.array([False] * dm_snap.N)
    mask[1000:2000] = True
    dm_snap.select_particles(mask)
    assert len(dm_snap.data_fields["ParticleIDs"]) == 1000, (
        "Mask selection failed!"
    )
    assert dm_snap.data_fields["Coordinates"].shape[0] == 1000, (
        "Mask selection failed!"
    )


def test_centering(dm_snap):
    test_kwargs = {"vol_dec": 2.0}
    com = dm_snap.find_position_center(**test_kwargs)
    assert np.allclose(com, np.array([0.11066814, -0.01628367, 0.42120691]) * u.kpc), (
        "Position center calculation failed!"
    )
    assert np.allclose(
        dm_snap.find_velocity_center(com),
        np.array([-2.337177, -0.104841, -1.221633]) * u.km / u.s,
    ), "Velocity center calculation failed!"


def test_density_points_returns_correct_central_density(dm_snap):
    k = 100
    central_density = dm_snap.density_points(
        np.zeros(3), k_max=k
    )[0].value

    particle_radii = np.sqrt(np.sum(dm_snap.data_fields['Coordinates']**2, axis=1))
    volume = 4.0 * np.pi / 3.0 * (np.partition(particle_radii, k-1)[k-1])**3
    emp_density = (dm_snap.data_fields['Masses'][0] * k / volume).value

    assert np.abs(central_density/emp_density - 1) < 1e-6


def test_halo_axis_ratios_are_similar(dm_snap):
    axis_ratios, principal_axes = dm_snap.principal_axes()

    assert np.all(np.abs(axis_ratios/axis_ratios.max() - 1) < 0.2)
