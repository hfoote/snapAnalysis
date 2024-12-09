# utility functions for snapAnalysis

import numpy as np
import astropy.units as u
import astropy.constants as const
from glob import glob
import re

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

	return np.where((np.abs(pos[:,axis]) <= (slice/2.)))

def get_snaps(dir:str, ext:str='.hdf5', prefix:str='snap_') -> np.ndarray:
	'''get_snaps returns an ordered list of all snapshots in a directory. 
	Original code by Himansh Rathore

	Parameters
	----------
	dir : str
		directory where snapshots are stored
	ext : str, optional
		snapshot file extension, by default '.hdf5'
	prefix : str, optional
		snapshot name prefix, by default 'snap_'

	Returns
	-------
	np.ndarray
		Ordered list of snapshots
	'''

	snap_list = np.array(glob(dir + prefix + '*' + ext))
	nsnaps = len(snap_list)

	if (nsnaps == 0):
		raise RuntimeError('No files found !')

	current_order = np.zeros(nsnaps)

	for i in range(nsnaps):
		snap = snap_list[i]
		result = re.search(prefix + '(.*)' + ext, snap)
		current_order[i] += int(result.group(1))
	
	snap_list_ordered = snap_list[np.argsort(current_order)]
	
	return snap_list_ordered    

def cartesian_to_spherical(coords:np.ndarray) -> np.ndarray:
	'''cartesian_to_spherical transforms a set of Cartesian coordinates
	to spherical coordinates. 

	Parameters
	----------
	coords : np.array
		Nx3 array of x,y,z coordinates

	Returns
	-------
	np.array
		(r, theta (polar), phi (azimuth)) coordinates
	'''

	# make input 2D if required
	if coords.ndim == 1:
		coords = coords[np.newaxis, :]
		remove_axis = True
	else:
		remove_axis = False

	r = np.sqrt(np.sum(coords.value**2, axis=1))
	theta = np.arccos(coords[:,2].value/r.value)
	phi = np.arctan2(coords[:,1].value, coords[:,0].value)

	if remove_axis:
		return np.hstack([r, theta, phi])
	
	return np.array([r, theta, phi]).T

def cartesian_to_cylindrical(coords:np.ndarray) -> np.ndarray:
	'''cartesian_to_cylindrical transforms a set of Cartesian coordinates
	to cylindrical coordinates. 

	Parameters
	----------
	coords : np.array
		Nx3 array of x,y,z coordinates

	Returns
	-------
	np.array
		(rho, phi (azimuth), z) coordinates
	'''
	
	# make input 2D if required
	if coords.ndim == 1:
		coords = coords[np.newaxis, :]
		remove_axis = True
	else:
		remove_axis = False

	rho = np.sqrt(np.sum(coords.value[:,:2]**2, axis=1))
	phi = np.arctan2(coords[:,1].value, coords[:,0].value)

	if remove_axis:
		return np.hstack([rho, phi, coords[:,2]])[0]
	
	return np.array([rho, phi, coords[:,2]]).T

def rotation_matrix(alpha:float, beta:float, gamma:float) -> np.ndarray:
	'''rotation_matrix returns the rotation matrix for a general intrinsic rotation
	of yaw, pitch, roll (Tait-Bryan angles about z,y,x) alpha, beta, gamma, respectively. 

	Parameters
	----------
	alpha : float
		yaw angle (about z axis)
	beta : float
		pitch angle (about y axis)
	gamma : float
		roll angle (about x axis)

	Returns
	-------
	np.ndarray
		rotation matrix
	'''

	Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], 
				   [np.sin(alpha), np.cos(alpha), 0],
				   [0, 0, 1]])
	
	Ry = np.array([[np.cos(beta), 0, np.sin(beta)], 
				   [0, 1, 0],
				   [-np.sin(beta), 0, np.cos(beta)]])
	
	Rx = np.array([[1, 0, 0], 
				   [0, np.cos(gamma), -np.sin(gamma)],
				   [0, np.sin(gamma), np.cos(gamma)]])
	
	return np.matmul(Rz, np.matmul(Ry, Rx))

def find_alignment_rotation(vec:np.ndarray) -> np.ndarray:
	'''find_alignment_rotation returns the rotation matrix needed to 
	align the input vector with the positive z-axis

	Parameters
	----------
	vec : np.ndarray
		[x,y,z] vector

	Returns
	-------
	np.ndarray
		rotation matrix
	'''

	# get rotation angles first
	_, theta, phi = cartesian_to_spherical(vec)

	return rotation_matrix(-phi, -theta, 0.)
