# E2E tests for functions in snapAnalysis.sim
import numpy as np


def test_orbit_com():
    from snapanalysis.sim import orbit_com
    import os

    TEST_ARR = np.array(
        [
            [0.000000, 0.110668, -0.016284, 0.421207, -2.337177, -0.104841, -1.221633],
            [
                0.197182,
                -0.401837,
                -0.480167,
                -0.274596,
                -0.160502,
                -0.603877,
                -0.087050,
            ],
        ]
    )

    # A user wants to determine the COM position and velocity of the dark matter
    # halo in the test simulation contained in tests/example_snaps.
    # They specify a shrinking factor of 2, and save the COM data in a
    # file called test_centers.txt
    test_file = "tests/test_centers.txt"
    orbit = orbit_com(
        "tests/example_snaps/", 1, com_kwargs={"vol_dec": 2.0}, out_file=test_file
    )

    assert np.allclose(orbit, TEST_ARR), "Orbit determination failed!"

    # Later, they want to use the stored centers, so they load the orbit file
    # they saved earlier, and compare it to the output from the center finder.
    orbit_from_file = np.loadtxt(test_file)

    assert np.allclose(orbit_from_file, TEST_ARR), "Orbit file failed!"

    # cleanup
    os.remove("tests/test_centers.txt")


def test_particle_orbit_extraction_returns_correct_data():
    from snapanalysis.sim import get_particle_orbit

    orbits = get_particle_orbit("tests/example_snaps/", 1, [1, 2])

    assert orbits.source_dir == "tests/example_snaps/"

    assert orbits.ids["1"] == 0
    assert orbits.ids["2"] == 1

    assert np.allclose(orbits.t.value, np.array([0.0, 0.19718176]))

    assert np.allclose(
        orbits.pos.value,
        np.array(
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
        ),
    )
    assert np.allclose(
        orbits.vel.value,
        np.array(
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
        ),
    )
