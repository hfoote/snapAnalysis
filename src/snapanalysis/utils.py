# utility functions for snapAnalysis

import numpy as np
from glob import glob
import re


def com_define(m: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """com_define basic center-of-mass calculation

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
    """

    tot_m = np.sum(m)

    return np.sum(pos * m[:, None] / tot_m, axis=0)


def set_axes(ax: int) -> tuple[int, int]:
    """set_axes returns x and y axes for a plot based on the projection axis

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
    """
    if ax == 0:  # y-z plane
        i = 1
        j = 2
    elif ax == 1:  # x-z plane
        i = 0
        j = 2
    elif ax == 2:  # x-y plane
        i = 0
        j = 1
    else:
        raise ValueError("Invalid axis index")

    return i, j


def get_vslice_indices(pos: np.ndarray, slice: float, axis: int) -> np.ndarray:
    """get_vslice_indices returns particle indices within a vertical slice
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
    """

    return np.where((np.abs(pos[:, axis]) <= (slice / 2.0)))


def get_snaps(dir: str, ext: str = ".hdf5", prefix: str = "snap_") -> np.ndarray:
    """get_snaps returns an ordered list of all snapshots in a directory.
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
    """

    snap_list = np.array(glob(dir + prefix + "*" + ext))
    nsnaps = len(snap_list)

    if nsnaps == 0:
        raise RuntimeError("No files found !")

    current_order = np.zeros(nsnaps)

    for i in range(nsnaps):
        snap = snap_list[i]
        result = re.search(prefix + "(.*)" + ext, snap)
        current_order[i] += int(result.group(1))

    snap_list_ordered = snap_list[np.argsort(current_order)]

    return snap_list_ordered


def cartesian_to_spherical(coords: np.ndarray) -> np.ndarray:
    """cartesian_to_spherical transforms a set of Cartesian coordinates
    to spherical coordinates, while preserving the shape of the input.

    Parameters
    ----------
    coords : np.array
        Nx3 array of x,y,z coordinates

    Returns
    -------
    np.array
        (r, theta (polar), phi (azimuth)) coordinates
    """

    # make input 2D if required
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
        remove_axis = True
    else:
        remove_axis = False

    r = np.sqrt(np.sum(coords**2, axis=1))
    theta = np.arccos(coords[:, 2] / r)
    phi = np.arctan2(coords[:, 1], coords[:, 0])

    if remove_axis:
        return np.hstack([r, theta, phi])

    return np.array([r, theta, phi]).T


def vector_cartesian_to_spherical(coords: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    '''vector_cartesian_to_spherical transforms a vector field defined in 
    cartesian coordinates at coords with vectors vecs into spherical coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Points at which the vector field is defined in Cartesian coordinates
    vecs : np.ndarray
        Cartesian vector values of the field

    Returns
    -------
    np.ndarray
        (r, theta (polar), phi (azimuth)) vectors
    '''

    # make inputs 2D if required
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
        vecs = vecs[np.newaxis, :]
        remove_axis = True
    else:
        remove_axis = False

    coords_spherical = cartesian_to_spherical(coords)
    theta = coords_spherical[:,1]
    phi = coords_spherical[:,2]

    vx = vecs[:,0]
    vy = vecs[:,1]
    vz = vecs[:,2]

    v_r = (
        np.sin(theta)*np.cos(phi)*vx
        + np.sin(theta)*np.sin(phi)*vy
        +  np.cos(theta)*vz
    )
    v_theta = (
        np.cos(theta)*np.cos(phi)*vx 
        + np.cos(theta)*np.sin(phi)*vy 
        - np.sin(theta)*vz
    )
    v_phi = (-np.sin(phi)*vx + np.cos(phi)*vy)

    if remove_axis:
        return np.hstack([v_r, v_theta, v_phi])

    return np.array([v_r, v_theta, v_phi]).T


def cartesian_to_cylindrical(coords: np.ndarray) -> np.ndarray:
    """cartesian_to_cylindrical transforms a set of Cartesian coordinates
    to cylindrical coordinates, while preserving the input shape.

    Parameters
    ----------
    coords : np.array
            Nx3 array of x,y,z coordinates

    Returns
    -------
    np.array
        (rho, phi (azimuth), z) coordinates
    """

    # make input 2D if required
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
        remove_axis = True
    else:
        remove_axis = False

    rho = np.sqrt(np.sum(coords[:, :2] ** 2, axis=1))
    phi = np.arctan2(coords[:, 1], coords[:, 0])

    if remove_axis:
        return np.hstack([rho, phi, coords[:, 2]])

    return np.array([rho, phi, coords[:, 2]]).T


def vector_cartesian_to_cylindrical(coords: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    '''vector_cartesian_to_cylindrical transforms a vector field defined in 
    cartesian coordinates at coords with vectors vecs into cylindrical coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Points at which the vector field is defined in Cartesian coordinates
    vecs : np.ndarray
        Cartesian vector values of the field

    Returns
    -------
    np.ndarray
        (rho, phi (azimuth), z) vectors
    '''

    # make inputs 2D if required
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
        vecs = vecs[np.newaxis, :]
        remove_axis = True
    else:
        remove_axis = False

    coords_cylindrical = cartesian_to_cylindrical(coords)
    phi = coords_cylindrical[:,1]

    vx = vecs[:,0]
    vy = vecs[:,1]
    vz = vecs[:,2]

    v_rho = (np.cos(phi)*vx + np.sin(phi)*vy)
    v_phi = (-np.sin(phi)*vx + np.cos(phi)*vy)

    if remove_axis:
        return np.hstack([v_rho, v_phi, vz])

    return np.array([v_rho, v_phi, vz]).T


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """rotation_matrix returns the rotation matrix for a general intrinsic rotation
    of yaw, pitch, roll (Tait-Bryan angles about z,y,x) alpha, beta, gamma, 
    respectively.

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
    """

    Rz = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )

    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ]
    )

    return np.matmul(Rx, np.matmul(Ry, Rz))


def find_alignment_rotation(vec: np.ndarray) -> np.ndarray:
    """find_alignment_rotation returns the rotation matrix needed to
    align the input vector with the positive z-axis

    Parameters
    ----------
    vec : np.ndarray
            [x,y,z] vector

    Returns
    -------
    np.ndarray
            rotation matrix
    """

    # get rotation angles first
    _, theta, phi = cartesian_to_spherical(vec)

    return rotation_matrix(-phi, -theta, 0.0)


def inertia_tensor(m: np.ndarray, pos: np.ndarray) -> np.ndarray:
    '''inertia_tensor Calculates the inertia tensor for a collection
    of point masses of mass m at positions pos

    Parameters
    ----------
    m : np.ndarray
        Nx1 array of masses
    pos : np.ndarray
        Nx3 array of positions

    Returns
    -------
    np.ndarray
        3x3 symmetric inertia tensor
    '''

    I = np.zeros([3, 3])

    I[0,0] = np.sum(m * (pos[:,1]**2 + pos[:,2]**2))
    I[1,1] = np.sum(m * (pos[:,0]**2 + pos[:,2]**2))
    I[2,2] = np.sum(m * (pos[:,0]**2 + pos[:,1]**2))

    I[0,1] = I[1,0] = - np.sum(m * pos[:,0] * pos[:,1])
    I[0,2] = I[2,0] = - np.sum(m * pos[:,0] * pos[:,2])
    I[1,2] = I[2,1] = - np.sum(m * pos[:,1] * pos[:,2])

    return I