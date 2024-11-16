# tests for the utility functions of snapAnalysis

def test_com_define() -> None:
	import numpy as np
	from snapAnalysis.utils import com_define

	pos = np.array([[1., 1., 1.],
				 	[-1., 1., 1.],
					[1., -1., 1.],
					[1., 1., -1.],
					[1., -1., -1.],
					[-1., 1., -1.],
					[-1., -1., 1.],
					[-1., -1., -1.]])
	m = np.ones(pos.shape[0])

	assert np.array_equal(com_define(m, pos), np.array([0., 0., 0.])), "COM defintion incorrect!"

	return None