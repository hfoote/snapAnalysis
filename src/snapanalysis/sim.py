# Routines for analyzing an entire simulation at once
# TODO: wrapper routine for multiprocessing
# TODO: function for storing rotation matrices

import numpy as np
import astropy.units as u
from .snap import snapshot
from .utils import get_snaps, find_alignment_rotation
from .orbit import Orbit


def orbit_com(
    sim_dir: str,
    part_type: int,
    out_file: None | str = None,
    select_IDs: None | tuple = None,
    use_guess: None | str = None,
    com_kwargs: dict = {},
    vel_kwargs: dict = {},
    verbose: bool = False,
) -> np.ndarray:
    """orbit_com stores the center-of-mass position and velocity of the specified

    Parameters
    ----------
    sim_dir : str
            Directory containing simulation snapshots
    part_type : int
            particle type to use for center-finding
    out_file : None or str, optional
            if not None, saves the orbit file under this name
    select_IDs : None or tuple, optional
            min. and max. particle IDs for selection if desired, otherwise uses all
            particles of the specified type.
    use_guess : None or str ('previous')
            specifies where the intial sphere is centered. None uses the com of
            every particle, 'previous' uses the previous snapshot's com.
    com_kwargs : dict, optional
            kwargs for snap.find_position_center, by default {}
    vel_kwargs : dict, optional
            kwargs for snap.find_velocity_center, by default {}
    verbose : bool, optional
            print progress, by default False

    Returns
    -------
    np.ndarray
            Nx7 array with [t,x,y,z,vx,vy,vz] at each timestep
    """

    snap_names = get_snaps(sim_dir)

    N_snaps = len(snap_names)
    orbit = np.zeros([N_snaps, 7])

    for i, snap in enumerate(snap_names):
        if verbose:
            print(f"Finding center of snapshot {i} of {N_snaps - 1}")

        s = snapshot(snap, part_type)
        s.read_all()

        if select_IDs:
            s.select_particles(select_IDs)

        if use_guess == "previous":
            if i == 0:
                # first snapshot uses no guess
                com_p = s.find_position_center(verbose=verbose, **com_kwargs)
            else:
                # use the com from the previous iteration
                com_p = s.find_position_center(
                    guess=com_p, verbose=verbose, **com_kwargs
                )
        else:  # calculate without initial guess
            com_p = s.find_position_center(verbose=verbose, **com_kwargs)
        com_v = s.find_velocity_center(com_p, **vel_kwargs)

        orbit[i] = s.time.value, *tuple(com_p.value), *tuple(com_v.value)

    if out_file is not None:
        np.savetxt(
            out_file,
            orbit,
            fmt="%13.6f" * 7,
            comments="#",
            header="{:>10s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}".format(
                f"t [{s.time.unit}]",
                f"x [{com_p.unit}]",
                f"y [{com_p.unit}]",
                f"z [{com_p.unit}]",
                f"vx [{com_v.unit}]",
                f"vy [{com_v.unit}]",
                f"vz [{com_v.unit}]",
            ),
        )

    return np.round(orbit, 6)


def get_alignment_rotations(
    sim_dir: str,
    part_type: int,
    r_max: u.Quantity | None = None,
    out_file: None | str = None,
    use_centers: None | str = None,
) -> np.ndarray:
    """
    Generates the rotation matrices needed to align the angular momentum vector of
    the specified particle type to the z-axis for every snapshot in a simulation.

    Parameters
    ----------
    sim_dir : str
        Directory in which the simualtion snapshots are stored
    part_type : int
        Particle type with which to compute angular momentum vectors
    r_max : u.Quantity | None, optional
        If not None, computes angular momenta using only particles inside
        this radius, by default None
    out_file : None | str, optional
        If given as a string, saves the rotation matrices to a numpy
        binary file of this name, by default None
    use_centers : None or str, optional
        Uses COM positions and velocities stored in the specified text file. If None,
        snapshots will be auto-centered before rotations are calculated

    Returns
    -------
    np.ndarray
        N_snapsx3x3 array containting the rotation matrices needed
        to align each snapshot's angular momentum to the z-axis
    """
    snap_names = get_snaps(sim_dir)

    if use_centers is not None:
        centers = np.loadtxt(use_centers)

    N_snaps = len(snap_names)
    matrices = np.zeros([N_snaps, 3, 3])

    for i, snap_name in enumerate(snap_names):
        s = snapshot(snap_name, part_type)
        s.load_particle_data(["Coordinates", "Velocities"])

        if use_centers is not None:
            s.apply_center(
                pos_center=centers[i, 1:4] * s.length_unit,
                vel_center=centers[i, 4:7] * s.vel_unit,
            )
        else:
            s.find_and_apply_center()

        J_vec = s.find_angular_momentum_direction(r_max)
        matrices[i, :, :] = find_alignment_rotation(J_vec.value)

    if out_file is not None:
        np.save(out_file, matrices)

    return matrices


def get_particle_orbit(
    sim_dir: str,
    part_type: int,
    ids: list,
    use_centers: None | str = None,
    use_rotations: None | str = None,
) -> Orbit:
    """
    Stores the orbit of a particle or particles throughout a simulation.
    TODO: support loading precomputed centers and rotations

    Parameters
    ----------
    sim_dir : str
        Directory containing simulation snapshots
    part_type : int
        Type of particle
    ids : list
        List of IDs of desired particles
    use_centers : None or str, optional
        Uses COM positions and velocities stored in the specified text file. If None,
        snapshots will be auto-centered
    use_rotations : None or str, optional
        Uses rotation matrices stored in the specified numpy binary file. If None,
        snapshots will be auto-aligned

    Returns
    -------
    Orbit : snapanalysis.orbit.Orbit
        Orbit object containing the orbits of the specified particles
    """

    snap_names = get_snaps(sim_dir)
    N_snaps = len(snap_names)

    if use_centers is not None:
        centers = np.loadtxt(use_centers)
    if use_rotations is not None:
        rotations = np.load(use_rotations)

    times = np.zeros(N_snaps)
    pos = np.zeros([N_snaps, 3, len(ids)])
    vel = np.zeros([N_snaps, 3, len(ids)])

    for i, snap_name in enumerate(snap_names):
        s = snapshot(snap_name, part_type)
        s.load_particle_data(["ParticleIDs", "Coordinates", "Velocities"])
        if use_centers is not None:
            s.apply_center(
                pos_center=centers[i, 1:4] * s.length_unit,
                vel_center=centers[i, 4:7] * s.vel_unit,
            )
        else:
            s.find_and_apply_center()
        if use_rotations is not None:
            s.apply_rotation(rotations[i, :, :])
        else:
            s.align_angular_momentum()

        mask = np.isin(s.data_fields["ParticleIDs"], ids)
        s.select_particles(mask)
        s.arrange_fields(np.argsort(s.data_fields["ParticleIDs"]))

        times[i] = s.time.value
        pos[i] = s.data_fields["Coordinates"].T.value
        vel[i] = s.data_fields["Velocities"].T.value

        if i == 0:
            id_map = dict(
                zip(
                    s.data_fields["ParticleIDs"].value.astype(int).astype(str),
                    range(len(ids)),
                )
            )

    return Orbit(
        times * s.time.unit, pos * s.length_unit, vel * s.vel_unit, id_map, sim_dir
    )
