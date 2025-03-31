# E2E tests for the functionality of snap objects
# requires the example snapshots in tests/example_snaps/

from snapAnalysis.snap import snapshot
import astropy.units as u
import numpy as np
import os
import pytest

# A user wants to analyze the CDM particles in the first snapshot of the simulaton
# stored in tests/example/snaps. They create a snapshot object using the DM particles in 
# this snapshot.
@pytest.fixture
def dm_snap():
	dm_snap = snapshot('tests/example_snaps/snap_000.hdf5', 1)
	return dm_snap

def test_CDM_workflow(dm_snap):

	# They load the positions, velocities, and masses of the particles,
	dm_snap.read_all()

	# and verify that these are loaded, while the potential is still empty. 
	assert dm_snap.check_if_field_read('Coordinates')
	assert dm_snap.check_if_field_read('Velocities')
	assert dm_snap.check_if_field_read('Masses')
	assert not dm_snap.check_if_field_read('Potential')

	# They center the positions and velocities on the snapshot's center of mass
	dm_snap.find_and_apply_center()

	# and check that the velocity and position centers are now zero.
	com = dm_snap.find_position_center()
	assert np.allclose(com, np.zeros(3))
	assert np.allclose(dm_snap.find_velocity_center(com), np.zeros(3))

	# They rotate the positions and velocities to align the total angular momentum
	# vector with the z-axis. 
	dm_snap.align_angular_momentum()
	assert np.allclose(dm_snap.find_angular_momentum_direction(), np.array([0., 0., 1.]))

	# and save a plot of the dm density within 10 kpc of the xy plane,
	dm_snap.density_projection(slice_width=10.*u.kpc, plot_name='tests/density_plot_test.png')
	# as well as the halo's density profile
	dm_snap.density_profile(plot_name='tests/density_profile_test.png')
	# and anisotropy profile
	dm_snap.anisotropy_profile(plot_name='tests/beta_profile_test.png')

	# later, they decide to load the potential at the particle locations, 
	dm_snap.load_particle_data(['Potential'])
	assert dm_snap.check_if_field_read('Potential')

	# and save a plot of the potential near the x-y plane as well. 
	dm_snap.potential_projection(slice_width=10.*u.kpc, plot_name='tests/potential_plot_test.png')

	# cleanup
	os.remove('tests/density_plot_test.png')
	os.remove('tests/density_profile_test.png')
	os.remove('tests/beta_profile_test.png')
	os.remove('tests/potential_plot_test.png')
	
