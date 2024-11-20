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

def set_axes(ax:int) -> tuple[int, int]:
	'''set_axes returns x and y axes for a plot based on the projection axis

	Parameters
	----------
	ax : int
		projection axis index

	Returns
	-------
	int :
		plot x-axis index
	int :
		plot y-axis index
	'''
	if ax == 0 : # y-z plane
		i = 1
		j = 2
	elif ax == 1 : # x-z plane
		i = 0
		j = 2
	elif ax == 2 : # x-y plane
		i = 0
		j = 1
	else :
		raise ValueError("Invalid axis index")
	
	return i, j

def get_vslice_indices(pos:np.ndarray, slice:float, axis:int) -> np.ndarray:
	'''get_vslice_indices returns particle indices within a vertical slice
	about the box midplane

	Parameters
	----------
	pos : np.ndarray
		paticle positions
	slice : float
		slice width in simulation units
	axis : int
		axis index along which the slice is taken

	Returns
	-------
	np.ndarray
		array of indices to pos that specify which particles are in the slice
	'''

	return np.argwhere((np.abs(pos[:,axis]) <= (slice/2.)))[0]
