# tests for functions in snapAnalysis.sim

def test_orbit_com() -> None:
	import numpy as np
	from snapAnalysis.sim import orbit_com

	test_kwargs = {'vol_dec':2.}
	orbit = orbit_com('tests/example_snaps/', 1, com_kwargs=test_kwargs)
	
	assert np.allclose(orbit, np.array([[0.000000,  0.110668, -0.016284,  0.421207, -2.337177, -0.104841, -1.221633],
										[0.197182, -0.401837, -0.480167, -0.274596, -0.160502, -0.603877, -0.087050]])), \
			"Orbit determination failed!"
	return None