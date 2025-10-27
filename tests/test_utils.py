# unit tests for the utility functions of snapAnalysis
import numpy as np


def test_com_define() -> None:
    from snapAnalysis.utils import com_define

    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    m = np.ones(pos.shape[0])

    assert np.array_equal(com_define(m, pos), np.array([0.0, 0.0, 0.0])), (
        "COM defintion incorrect!"
    )

    return None


def test_get_vslice_indices() -> None:
    from snapAnalysis.utils import get_vslice_indices

    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert np.array_equal(
        get_vslice_indices(pos, slice=0.5, axis=2), (np.array([8]),)
    ), "Vertical slicing incorrect!"

    return None


def test_get_snaps() -> None:
    from snapAnalysis.utils import get_snaps

    expected = np.array(
        ["tests/example_snaps/snap_000.hdf5", "tests/example_snaps/snap_001.hdf5"]
    )

    assert np.array_equal(get_snaps("tests/example_snaps/"), expected), (
        "Snapshot collection failed!"
    )

    return None


def test_cartesian_to_spherical() -> None:
    from snapAnalysis.utils import cartesian_to_spherical

    test_vec_1d = np.ones(3)
    test_vec_2d = np.ones([2, 3])

    expected = np.array([np.sqrt(3), np.arccos(1.0 / np.sqrt(3)), np.pi / 4.0])

    assert np.allclose(cartesian_to_spherical(test_vec_1d), expected), (
        "Spherical coordinate conversion failed!"
    )
    assert np.allclose(
        cartesian_to_spherical(test_vec_2d), np.vstack([expected, expected])
    ), "Spherical coordinate conversion failed!"


def test_cartesian_to_cylindrical() -> None:
    from snapAnalysis.utils import cartesian_to_cylindrical

    test_vec_1d = np.ones(3)
    test_vec_2d = np.ones([2, 3])

    expected = np.array([np.sqrt(2), np.pi / 4.0, 1.0])

    assert np.allclose(cartesian_to_cylindrical(test_vec_1d), expected), (
        "Cylindrical coordinate conversion failed!"
    )
    assert np.allclose(
        cartesian_to_cylindrical(test_vec_2d), np.vstack([expected, expected])
    ), "Cylindrical coordinate conversion failed!"

    return None


def test_find_alignment_rotation() -> None:
    from snapAnalysis.utils import find_alignment_rotation

    input = np.array([[23.0, 5.0, -64.0]])
    expected = np.array([[0.0, 0.0, 1.0]]) * np.linalg.norm(input[0])

    rotation = np.matmul(find_alignment_rotation(input[0]), input.T).T

    assert np.allclose(rotation, expected), "Rotations failed!"

    return None
