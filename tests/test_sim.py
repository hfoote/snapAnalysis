# E2E tests for functions in snapAnalysis.sim

def test_orbit_com():
	import numpy as np
	from snapAnalysis.sim import orbit_com
	import os

	TEST_ARR = np.array([[0.000000,  0.110668, -0.016284,  0.421207, -2.337177, -0.104841, -1.221633],
						 [0.197182, -0.401837, -0.480167, -0.274596, -0.160502, -0.603877, -0.087050]])

	# A user wants to determine the COM position and velocity of the dark matter halo in 
	# the test simulation contained in tests/example_snaps.
	# They specify a shrinking factor of 2, and save the COM data in a file called test_centers.txt
	test_file = 'tests/test_centers.txt'
	orbit = orbit_com('tests/example_snaps/', 1, com_kwargs={'vol_dec':2.}, out_file=test_file)
	
	assert np.allclose(orbit, TEST_ARR), "Orbit determination failed!"

	# Later, they want to use the stored centers, so they load the orbit file they saved earlier, 
	# and compare it to the output from the center finder. 
	orbit_from_file = np.loadtxt(test_file)

	assert np.allclose(orbit_from_file, TEST_ARR), "Orbit file failed!"

	# cleanup
	os.remove('tests/test_centers.txt')