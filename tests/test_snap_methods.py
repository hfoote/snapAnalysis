# tests for the functionality of snap objects
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

def test_centering() -> None:
	
	# separate test for each centering routine
	test_kwargs = {'vol_dec':2.}
	com = snap_obj.find_position_center(**test_kwargs)
	assert np.allclose(com, np.array([0.11066814, -0.01628367,  0.42120691])*u.kpc), \
		"Position center calculation failed!"
	assert np.allclose(snap_obj.find_velocity_center(com), np.array([-2.337177, -0.104841, -1.221633])*u.km/u.s), \
		"Velocity center calculation failed!"

	snap_obj.find_and_apply_center(com_kwargs=test_kwargs)
	
	assert np.allclose(snap_obj.data_fields['Coordinates'][0], 
					   np.sum(np.array([[-34.63336945, -66.06946564, 172.63697815], [-0.110668, 0.016284, -0.421207]]), axis=0)*u.kpc), \
					   "Position centering failed!"
	assert np.allclose(snap_obj.data_fields['Velocities'][0], 
					   np.sum(np.array([[1.14670885, -28.27835464, -46.99738312], [2.337177, 0.104841, 1.221633]]), axis=0)*u.km/u.s), \
					   "Velocity centering failed!"
	
	return None

def test_select_particles() -> None:

	min_id = 1000
	max_id = 2000
	snap_obj.select_particles((min_id, max_id))
	assert (snap_obj.data_fields['ParticleIDs'] >= min_id).all(), "Particle ID selection lower limit failed!"
	assert (snap_obj.data_fields['ParticleIDs'] <= max_id).all(), "Particle ID selection upper limit failed!"
	assert (len(snap_obj.data_fields['ParticleIDs']) == (max_id - min_id +1)), "Particle ID selection failed!"