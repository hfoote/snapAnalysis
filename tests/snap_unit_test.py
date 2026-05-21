# unit tests for base methods of snapshot objects.
# Requires the example snapshot in example_snaps/snap_000.hdf5

import numpy as np
from snapanalysis.snap import snapshot
import astropy.units as u
import pytest

CDM_TEST_SNAP_PATH = "tests/example_snaps/cdm_snaps/"
FDM_TEST_SNAP_PATH = "tests/example_snaps/fdm_snaps/"


@pytest.fixture
def dm_snap():
    s = snapshot(CDM_TEST_SNAP_PATH + "snap_000.hdf5", 1)
    return s


@pytest.fixture
def fdm_snap():
    s = snapshot(FDM_TEST_SNAP_PATH + "snap_000.hdf5", 1)
    return s


@pytest.fixture
def star_snap():
    s = snapshot(CDM_TEST_SNAP_PATH + "snap_000.hdf5", 4)
    return s


## tests for I/O
@pytest.mark.parametrize(
    "snap, expected",
    [
        (
            "dm_snap",
            {"fdm": False, "time": 0.0 * u.Gyr, "box_size": 0.0 * u.kpc, "N": 10000},
        ),
        (
            "fdm_snap",
            {
                "fdm": True,
                "time": 0.0 * u.Gyr,
                "box_size": 299.99994169 * u.kpc,
                "N": 128**3,
            },
        ),
        (
            "star_snap",
            {"fdm": False, "time": 0.0 * u.Gyr, "box_size": 0.0 * u.kpc, "N": 0},
        ),
    ],
)
def test_can_read_metadata(snap, expected, request):
    s = request.getfixturevalue(snap)
    assert s.fdm == expected["fdm"], "DM type detection failed!"
    assert s.time == expected["time"], "Time detection failed!"
    assert np.abs(s.box_size - expected["box_size"]) < 1e-6 * u.kpc, (
        "Box size detction failed!"
    )
    assert s.N == expected["N"], "Particle number detection failed!"


def test_can_read_fdm_metadata(fdm_snap):
    assert fdm_snap.fdm, "DM type detection failed!"
    assert fdm_snap.N_cells == 128, "Number of cells detection failed!"
    assert fdm_snap.m_axion_ev == 1e-23 * u.eV, "FDM particle mass detection failed!"


def test_can_load_fdm_velocities(fdm_snap):
    # This is a windtunnel FDM snapshot - the y-velocity should be 32 km/s and 
    # the x and z velocities should be zero,
    # with some noise from the random phases during IC generation.
    tol = 1.0 * u.km / u.s  # tolerance for mean FDM velocity in each dimension
    assert fdm_snap.data_fields["Velocities"] is None
    fdm_snap.load_particle_data(["Velocities"])

    assert fdm_snap.data_fields["Velocities"].shape == (128**3, 3), (
        "FDM velocity loading failed!"
    )
    assert np.abs(np.mean(fdm_snap.data_fields["Velocities"][:, 0])) < tol, (
        "Mean FDM velocity in x direction is nonzero!"
    )
    assert (
        np.abs(np.mean(fdm_snap.data_fields["Velocities"][:, 1]) - 32 * u.km / u.s)
        < tol
    ), "Mean FDM velocity in y direction is incorrect!"
    assert np.abs(np.mean(fdm_snap.data_fields["Velocities"][:, 2])) < tol, (
        "Mean FDM velocity in z direction is nonzero!"
    )


def test_fdm_masses_are_cell_densities(fdm_snap):
    expected_density = 6.25e10 * u.Msun / fdm_snap.box_size**3
    fdm_snap.load_particle_data(["Masses"])

    assert (np.mean(fdm_snap.data_fields["Masses"]) / expected_density - 1.0) < 1e-3, (
        "FDM mass loading failed!"
    )


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


def test_load_particle_data_throws_error_if_no_wavefunction(dm_snap):
    with pytest.raises(
        RuntimeError, match="Non-FDM particle types do not contain wavefunction data!"
    ):
        dm_snap.load_particle_data(["PsiRe"])


def test_load_particle_data_can_load_wavefunction_data(fdm_snap):
    fdm_snap.load_particle_data(["PsiRe", "PsiIm"])

    assert fdm_snap.data_fields["PsiRe"].shape == (fdm_snap.N_cells**3,), (
        "Failed to load wavefunction real part!"
    )
    assert fdm_snap.data_fields["PsiIm"].shape == (fdm_snap.N_cells**3,), (
        "Failed to load wavefunction imaginary part!"
    )


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
    assert len(dm_snap.data_fields["ParticleIDs"]) == 1000, "Mask selection failed!"
    assert dm_snap.data_fields["Coordinates"].shape[0] == 1000, "Mask selection failed!"


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


def test_angular_momentum_alignment_is_consistent_with_manual_rotation(dm_snap):
    from snapanalysis.utils import find_alignment_rotation

    dm_snap.align_angular_momentum()
    assert np.allclose(
        dm_snap.find_angular_momentum_direction(), np.array([0.0, 0.0, 1.0])
    )

    snap_2 = snapshot(CDM_TEST_SNAP_PATH + "snap_000.hdf5", 1)
    J_vec = snap_2.find_angular_momentum_direction()
    mat = find_alignment_rotation(J_vec.value)
    snap_2.apply_rotation(mat)

    assert np.allclose(
        snap_2.find_angular_momentum_direction(), np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(
        dm_snap.data_fields["Coordinates"], snap_2.data_fields["Coordinates"]
    )


def test_density_points_returns_correct_central_density(dm_snap):
    k = 100
    central_density = dm_snap.density_points(np.zeros(3), k_max=k)[0].value

    particle_radii = np.sqrt(np.sum(dm_snap.data_fields["Coordinates"] ** 2, axis=1))
    volume = 4.0 * np.pi / 3.0 * (np.partition(particle_radii, k - 1)[k - 1]) ** 3
    emp_density = (dm_snap.data_fields["Masses"][0] * k / volume).value

    assert np.abs(central_density / emp_density - 1) < 1e-6


def test_halo_axis_ratios_are_similar(dm_snap):
    axis_ratios, principal_axes = dm_snap.principal_axes()

    assert np.all(np.abs(axis_ratios / axis_ratios.max() - 1) < 0.2)
