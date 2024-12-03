# tests for functions in snapAnalysis.sim

def test_orbit_com() -> None:
	import numpy as np
	from snapAnalysis.sim import orbit_com
	import os

	test_kwargs = {'vol_dec':2.}
	test_arr = np.array([[0.000000,  0.110668, -0.016284,  0.421207, -2.337177, -0.104841, -1.221633],
						 [0.197182, -0.401837, -0.480167, -0.274596, -0.160502, -0.603877, -0.087050]])
	test_file = 'tests/test_centers.txt'
	orbit = orbit_com('tests/example_snaps/', 1, com_kwargs=test_kwargs, out_file=test_file)
	
	assert np.allclose(orbit, test_arr), "Orbit determination failed!"

	orbit_from_file = np.loadtxt(test_file)

	assert np.allclose(orbit_from_file, test_arr), "Orbit file failed!"

	os.remove('tests/test_centers.txt')

	return None