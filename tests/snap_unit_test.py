# unit tests for base methods of snapshot objects. 
# Requires the example snapshot in example_snaps/snap_000.hdf5

import numpy as np
from snapAnalysis.snap import snapshot
import astropy.units as u
import pytest

@pytest.fixture
def dm_snap():
	s = snapshot('tests/example_snaps/snap_000.hdf5', 1)
	return s

@pytest.fixture
def star_snap():
	s = snapshot('tests/example_snaps/snap_000.hdf5', 4)
	return s

@pytest.fixture
def binary_snap():
	s = snapshot('tests/example_snaps/snap_binary_000', 1, file_format=2)
	return s

## tests for I/O
@pytest.mark.parametrize("snap, expected", [("dm_snap", {'fdm':False, 'time':0.*u.Gyr, 'box_size':0., 'N':10000}),
											("binary_snap", {'fdm':False, 'time':0.*u.Gyr, 'box_size':0., 'N':10000}),
									        ("star_snap", {'fdm':False, 'time':0.*u.Gyr, 'box_size':0., 'N':0})])
def test_can_read_metadata(snap, expected, request):
	s = request.getfixturevalue(snap)
	assert s.fdm == expected['fdm'], "DM type detection failed!"
	assert s.time == expected['time'], "Time detection failed!"
	assert s.box_size == expected['box_size'], "Box size detction failed!"
	assert s.N == expected['N'], "Particle number detection failed!"

def test_unit_detection(dm_snap):
	print(dm_snap)
	unit_tol = 1e-3 # 1 part in 1000 should account for rounding differences between unit specification in astropy vs gadget
	assert (dm_snap.field_units['Coordinates'] - 1.0*u.kpc)/(1.0*u.kpc) < unit_tol, "Length unit detection failed!"
	assert (dm_snap.field_units['Velocities'] - 1.0*u.km/u.s)/(1.0*u.km/u.s) < unit_tol, "Velocity unit detection failed!"
	assert (dm_snap.field_units['Masses'] - 1e10*u.Msun)/(1e10*u.Msun) < unit_tol, "Mass unit detection failed!"

@pytest.mark.parametrize("snap, expected", [("dm_snap", np.array([-34.63336945, -66.06946564, 172.63697815])),
											("binary_snap", np.array([-34.63336945, -66.06946564, 172.63697815]))])
def test_can_read_field(snap, expected, request):
	s = request.getfixturevalue(snap)
	assert np.allclose(s.read_field('Coordinates')[0], expected), "Particle data reading failed!"

def test_can_read_masstable(dm_snap):
	assert (dm_snap.read_masstable() == 0.001799996432347745), "Failed to read masstable!"

def test_can_select_particles(dm_snap):

	min_id = 1000
	max_id = 2000
	dm_snap.select_particles((min_id, max_id))
	assert (dm_snap.data_fields['ParticleIDs'] >= min_id).all(), "Particle ID selection lower limit failed!"
	assert (dm_snap.data_fields['ParticleIDs'] <= max_id).all(), "Particle ID selection upper limit failed!"
	assert (len(dm_snap.data_fields['ParticleIDs']) == (max_id - min_id + 1)), "Particle ID selection failed!"

def test_centering(dm_snap):
	test_kwargs = {'vol_dec':2.}
	com = dm_snap.find_position_center(**test_kwargs)
	assert np.allclose(com, np.array([0.11066814, -0.01628367,  0.42120691])*u.kpc), \
		"Position center calculation failed!"
	assert np.allclose(dm_snap.find_velocity_center(com), np.array([-2.337177, -0.104841, -1.221633])*u.km/u.s), \
		"Velocity center calculation failed!"
