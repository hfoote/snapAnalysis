# utility functions for snapAnalysis

import numpy as np
import astropy.units as u
import astropy.constants as const

def com_define(m:np.ndarray, pos:np.ndarray) -> np.ndarray:
	'''com_define basic center-of-mass calculation

	Parameters
	----------
	m : np.ndarray
		particle masses
	pos : np.ndarray
		particle positions

	Returns
	-------
	np.ndarray
		[x, y, z] center of mass
	'''

	tot_m = np.sum(m)

	return np.sum(pos * m[:, None] / tot_m, axis=0)