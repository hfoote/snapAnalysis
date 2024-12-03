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

def test_get_vslice_indices() -> None:
	import numpy as np
	from snapAnalysis.utils import get_vslice_indices

	pos = np.array([[1., 1., 1.],
				 	[-1., 1., 1.],
					[1., -1., 1.],
					[1., 1., -1.],
					[1., -1., -1.],
					[-1., 1., -1.],
					[-1., -1., 1.],
					[-1., -1., -1.], 
					[0., 0., 0.]])
	
	assert np.array_equal(get_vslice_indices(pos, slice=0.5, axis=2), (np.array([8]),)), "Vertical slicing incorrect!"

	return None
	
def test_get_snaps() -> None:
	import numpy as np
	from snapAnalysis.utils import get_snaps

	expected = np.array(['tests/example_snaps/snap_000.hdf5', 'tests/example_snaps/snap_001.hdf5'])

	assert(np.array_equal(get_snaps('tests/example_snaps/'), expected)), "Snapshot collection failed!"

	return None
