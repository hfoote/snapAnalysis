# Routines for analyzing an entire simulation at once

import numpy as np
from snapAnalysis.snap import snapshot
from snapAnalysis.utils import get_snaps

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

    if out_file:
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
