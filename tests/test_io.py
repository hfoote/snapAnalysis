# tests for the I/O functionality of snap objects
# requires the example snapshot test_snap.hdf5

from snapAnalysis.snap import snap
import astropy.units as u
import numpy as np
snap_obj = snap('tests/test_snap.hdf5', 1)

def test_snap_creation() -> None:

	assert snap_obj.fdm == False, "DM type detection failed!"
	assert snap_obj.time == 0.*u.Gyr, "Time detection failed!"
	assert snap_obj.box_size == 0., "Box size detction failed!"
	assert snap_obj.N == 10000, "Particle number detection failed!"

	unit_tol = 1e-3 # 1 part in 1000 should account for rounding differences between unit specification in astropy vs gadget
	assert (snap_obj.field_units['Coordinates'] - 1.0*u.kpc)/(1.0*u.kpc) < unit_tol, "Length unit detection failed!"
	assert (snap_obj.field_units['Velocities'] - 1.0*u.km/u.s)/(1.0*u.km/u.s) < unit_tol, "Velocity unit detection failed!"
	assert (snap_obj.field_units['Masses'] - 1e10*u.Msun)/(1e10*u.Msun) < unit_tol, "Mass unit detection failed!"

	return None

def test_read_field() -> None:

	assert np.allclose(snap_obj.read_field('Coordinates')[0], np.array([-34.63336945, -66.06946564, 172.63697815])), "Particle data reading failed!"

	return None

def test_read_masstable() -> None:

	assert (snap_obj.read_masstable() == 0.001799996432347745), "Failed to read masstable!"

	return None

def test_load_particle_data() -> None:

	snap_obj.load_particle_data(['Coordinates'])
	assert np.allclose(snap_obj.data_fields['Coordinates'][0], np.array([-34.63336945, -66.06946564, 172.63697815])*u.kpc), \
	"Failed to load particle data!"

	return None

def test_check_if_field_read() -> None:
	
	assert snap_obj.check_if_field_read('Coordinates'), "Coordinates have already been loaded, test failed!"
	assert not snap_obj.check_if_field_read('Velocities'), "Velocities have not yet been loaded, test failed!"

	return None