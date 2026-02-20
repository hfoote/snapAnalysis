# Unit tests for the Orbit dataclass
import numpy as np
from snapanalysis.orbit import Orbit, read_orbit_from_file
import astropy.units as u
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_orbit_object_initialized_correctly():
    t = np.arange(10) * u.Myr
    ids = {"1": 0}
    pos = np.zeros([10, 3]) * u.kpc
    vel = np.ones([10, 3]) * u.km / u.s
    orb = Orbit(t=t, ids=ids, pos=pos, vel=vel)

    assert np.array_equal(orb.t, t)
    assert orb.ids == ids
    assert np.array_equal(orb.pos, pos)
    assert np.array_equal(orb.vel, vel)
    assert orb.source_dir is None


def test_read_orbit_from_file(temp_dir):
    t = np.arange(10) * u.Myr
    ids = {"1": 0}
    pos = np.zeros([10, 3]) * u.kpc
    vel = np.ones([10, 3]) * u.km / u.s
    orb = Orbit(t=t, ids=ids, pos=pos, vel=vel)
    orb.save(temp_dir / "temp_orb.pkl")

    orb_read = read_orbit_from_file(temp_dir / "temp_orb.pkl")

    assert orb == orb_read
