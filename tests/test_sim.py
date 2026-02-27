# E2E tests for functions in snapAnalysis.sim
import numpy as np
from snapanalysis.sim import orbit_com, get_particle_orbit, get_alignment_rotations
import pytest

# expected particle orbit data
PARTICLE_ORBIT_POS = np.array(
    [
        [
            [-6.37767716, -67.28720821],
            [-1.31997945, 275.07242887],
            [-3.76567291, 11.24781486],
        ],
        [
            [-9.43083674, -58.93297116],
            [-15.3050804, 264.30005654],
            [6.45821875, 90.12003454],
        ],
    ]
)

PARTICLE_ORBIT_VEL = np.array(
    [
        [
            [-101.31617369, 5.04129786],
            [-111.0747997, 14.59257077],
            [62.6801419, -9.44535906],
        ],
        [
            [44.51811798, 7.05256656],
            [-17.01287482, 14.23372224],
            [49.01288198, -3.34417125],
        ],
    ]
)

# expected snapshot orbit data
COM_ORBIT = np.array(
    [
        [0.0, 0.122077, -0.062773, 0.174305, -2.467153, -0.026791, -1.192435],
        [0.197182, -0.430234, -0.513655, -0.271056, -0.246454, -0.602834, -0.123797],
    ],
)

# expected alignment rotation data
ALIGN_ROT = np.array(
    [
        [
            [-0.08363217, 0.0202638, -0.99629064],
            [-0.23548298, -0.97187847, 0.0],
            [-0.96827343, 0.23460949, 0.08605209],
        ],
        [
            [-0.09203664, -0.00533282, -0.99574134],
            [0.0578453, -0.99832556, 0.0],
            [-0.99407403, -0.05759896, 0.09219101],
        ],
    ]
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_simulation_workflow(temp_dir):
    # A user wants to determine the COM position and velocity of the dark matter
    # halo in the test simulation contained in tests/example_snaps.
    # They save the COM data in a
    # file called test_centers.txt
    com_file = temp_dir / "test_centers.txt"
    rot_file = temp_dir / "test_rotations.npy"
    orbit = orbit_com("tests/example_snaps/", 1, out_file=com_file)

    assert np.allclose(orbit, COM_ORBIT), "Orbit determination failed!"

    # Next, they compute the rotation matrices required to align the
    # angular momentum of the halo with the z-axis
    rotations = get_alignment_rotations(
        "tests/example_snaps/", 1, out_file=rot_file, use_centers=com_file
    )

    assert np.allclose(rotations, ALIGN_ROT)

    # Later, they want to use the stored centers, so they load the orbit file
    # they saved earlier, and compare it to the output from the center finder.
    orbit_from_file = np.loadtxt(com_file)

    assert np.allclose(orbit_from_file, COM_ORBIT), "Orbit file failed!"

    # and they do the same with the rotations
    rot_from_file = np.load(rot_file)
    assert np.allclose(rot_from_file, ALIGN_ROT)

    # Next, they want to use those precomputed centers to find the orbits of
    # the first two particle IDs
    particle_orbits = get_particle_orbit(
        "tests/example_snaps/", 1, [1, 2], use_centers=com_file, use_rotations=rot_file
    )

    assert np.allclose(particle_orbits.pos.value, PARTICLE_ORBIT_POS)
    assert np.allclose(particle_orbits.vel.value, PARTICLE_ORBIT_VEL)


def test_particle_orbit_extraction_returns_correct_data_for_multiple_particles():
    # A user wants to extract the orbits of the first two particle IDs
    orbits = get_particle_orbit("tests/example_snaps/", 1, [1, 2])

    # they verify the orbits come from the correct simulation
    assert orbits.source_dir == "tests/example_snaps/"

    # contain the correct particles
    assert orbits.ids["1"] == 0
    assert orbits.ids["2"] == 1

    assert np.allclose(orbits.t.value, np.array([0.0, 0.19718176]))

    # and the correct data
    assert np.allclose(orbits.pos.value, PARTICLE_ORBIT_POS)
    assert np.allclose(orbits.vel.value, PARTICLE_ORBIT_VEL)

def test_particle_orbit_extraction_returns_correct_data_for_single_particle():
    # A user wants to extract the orbits of the first particle ID
    orbits = get_particle_orbit("tests/example_snaps/", 1, [1])

    # they verify the orbits come from the correct simulation
    assert orbits.source_dir == "tests/example_snaps/"

    # contain the correct particles
    assert orbits.ids["1"] == 0

    assert np.allclose(orbits.t.value, np.array([0.0, 0.19718176]))

    # and the correct data
    assert np.allclose(orbits.pos.value, PARTICLE_ORBIT_POS[:,:,:0])
    assert np.allclose(orbits.vel.value, PARTICLE_ORBIT_VEL[:,:,:0])