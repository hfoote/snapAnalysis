# Defines snapAnalysis' Orbit dataclass for storing orbits of individual particles

from dataclasses import dataclass
import numpy as np
from astropy.units import Quantity
import pickle


@dataclass(eq=False)
class Orbit:
    """Stores orbits of particles throughout a simulation.
    TODO: allow adding of orbits

    Attributes
    ----------
    t : np.array of astropy.Quantity
            Times the orbit is recorded at
    pos : np.array of astropy.Quantity
            Orbit xyz positions, shape is (N_times, 3, N_particles)
    vel : np.array of astropy.Quantity
            Orbit xyz positions, shape is (N_times, 3, N_particles)
    ids : dict
            Keys are particle IDs, values are the index of the particles in the
            last dimension of the position and velocity arrays
    source : str, optional
            The directory of the simulation the orbit was extracted from
    """

    t: np.ndarray[Quantity]
    pos: np.ndarray[Quantity]
    vel: np.ndarray[Quantity]
    ids: dict
    source_dir: str | None = None

    def __eq__(self, other_obj: object) -> bool:
        """
        Returns True if the contents of the other Orbit object are identical
        """
        if isinstance(other_obj, Orbit):
            return (
                np.array_equal(self.t, other_obj.t)
                and np.array_equal(self.pos, other_obj.pos)
                and np.array_equal(self.vel, other_obj.vel)
                and self.ids == other_obj.ids
                and self.source_dir == other_obj.source_dir
            )
        return False

    def save(self, filename: str):
        """
        Saves the Orbit object to a pickle binary file

        Parameters
        ----------
        filename : str
                name of the file
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def read_orbit_from_file(filename: str) -> Orbit:
    """
    Reads an Orbit object from a pickle binary file

    Parameters
    ----------
    filename : str
            name of the file

    Returns
    -------
    snapanalysis.orbit.Orbit
            Loaded Orbit object
    """

    with open(filename, "rb") as f:
        orb = pickle.load(f)

    return orb
