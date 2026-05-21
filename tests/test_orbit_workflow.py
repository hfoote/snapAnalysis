# E2E test for a workflow for extracting individual particle orbits from a simulation
import numpy as np
from astropy import units as u
import pytest
from tests.snap_unit_test import CDM_TEST_SNAP_PATH


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_orbit_workflow(temp_dir):
    from snapanalysis.orbit import Orbit, read_orbit_from_file
    from snapanalysis.sim import get_particle_orbit

    # a user wants to extract the orbit of two particles from an entire simulation
    sim_dir = CDM_TEST_SNAP_PATH
    orb = get_particle_orbit(sim_dir, 1, ids=[1, 2])

    # The orbit object stores the simulation directory that it refers to
    orb.source_dir == sim_dir

    # They verify the orbit contains the correct particles,
    # (i.e. ID 1 is the first particle, and ID 2 is the second)
    assert orb.ids["1"] == 0
    assert orb.ids["2"] == 1

    # The time array contains the snapshot times for the entire simulation
    assert np.allclose(orb.t.value, np.array([0.0, 0.19718176]))

    # The position array contains the xyz coordinates at each time,
    # i.e. has the shape (N_times, 3, N_particles)
    assert orb.pos.shape == (2, 3, 2)

    # The velocity array contains the xyz coordinates at each time,
    # i.e. has the shape (N_times, 3, N_particles)
    assert orb.vel.shape == (2, 3, 2)

    # The units are stored correctly
    assert orb.t.unit == u.Gyr
    assert orb.pos.unit == u.kpc
    assert orb.vel.unit == u.km / u.s

    # Finally, they save the orbit to a file
    orb.save(temp_dir / "test_particle_orbits.pkl")

    # Later, they want to load the stored orbit file
    orb_from_file = read_orbit_from_file(temp_dir / "test_particle_orbits.pkl")

    assert isinstance(orb_from_file, Orbit)
    assert orb == orb_from_file
