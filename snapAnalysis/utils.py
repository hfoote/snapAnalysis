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

	M = np.sum(m)

	a_com = np.sum(pos[0]*m)/M
	b_com = np.sum(pos[1]*m)/M
	c_com = np.sum(pos[2]*m)/M

	return a_com, b_com, c_com