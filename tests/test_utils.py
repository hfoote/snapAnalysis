# unit tests for the utility functions of snapAnalysis
import numpy as np
import pytest

def test_com_define() -> None:
    from snapanalysis.utils import com_define

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
    from snapanalysis.utils import get_vslice_indices

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
    from snapanalysis.utils import get_snaps

    expected = np.array(
        ["tests/example_snaps/snap_000.hdf5", "tests/example_snaps/snap_001.hdf5"]
    )

    assert np.array_equal(get_snaps("tests/example_snaps/"), expected), (
        "Snapshot collection failed!"
    )

    return None


@pytest.mark.parametrize(
    "input, expected",
    [
        (np.ones(3), np.array([np.sqrt(3), np.arccos(1.0 / np.sqrt(3)), np.pi / 4.0])),
        (
            np.ones([2, 3]), 
            np.vstack([
                np.array([np.sqrt(3), np.arccos(1.0 / np.sqrt(3)), np.pi / 4.0]), 
                np.array([np.sqrt(3), np.arccos(1.0 / np.sqrt(3)), np.pi / 4.0])
            ])
        )
    ]
)
def test_cartesian_to_spherical(input, expected) -> None:
    from snapanalysis.utils import cartesian_to_spherical

    assert np.allclose(cartesian_to_spherical(input), expected), (
        "Spherical coordinate conversion failed!"
    )


@pytest.mark.parametrize(
    "input, expected",
    [
        (np.ones(3), np.array([np.sqrt(2), np.pi / 4.0, 1.0])),
        (
            np.ones([2, 3]), 
            np.vstack([
                np.array([np.sqrt(2), np.pi / 4.0, 1.0]), 
                np.array([np.sqrt(2), np.pi / 4.0, 1.0])
            ])
        )
    ]
)
def test_cartesian_to_cylindrical(input, expected) -> None:
    from snapanalysis.utils import cartesian_to_cylindrical

    assert np.allclose(cartesian_to_cylindrical(input), expected), (
        "Cylindrical coordinate conversion failed!"
    )

    return None


def test_find_alignment_rotation() -> None:
    from snapanalysis.utils import find_alignment_rotation

    input = np.array([[23.0, 5.0, -64.0]])
    expected = np.array([[0.0, 0.0, 1.0]]) * np.linalg.norm(input[0])

    rotation = np.matmul(find_alignment_rotation(input[0]), input.T).T

    assert np.allclose(rotation, expected), "Rotations failed!"

    return None


@pytest.mark.parametrize(
    "input_points, input_vectors, expected", 
    [
        (
            np.array([0.5, 0.5, 1./np.sqrt(2)]), 
            np.array([0., 0., 1.]), 
            np.array([1./np.sqrt(2), -1./np.sqrt(2), 0])
        ),
        (
            np.array([[0.5, 0.5, 1./np.sqrt(2)],
                     [1., 0., 0.]]), 
            np.array([[0., 0., 1.],
                     [0., 0., 1.]]), 
            np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0],
                     [0., -1., 0.]])
        )
    ]
)
def test_vector_cartesian_to_spherical(input_points, input_vectors, expected) -> None:
    from snapanalysis.utils import vector_cartesian_to_spherical

    assert np.allclose(
        vector_cartesian_to_spherical(input_points, input_vectors), expected
    ), "Spherical vector transform failed!"

    return None

@pytest.mark.parametrize(
    "input_points, input_vectors, expected", 
    [
        (
            np.array([0.5, 0.5, 1.]), 
            np.array([0., 0., 1.]), 
            np.array([0., 0., 1.])
        ),
        (
            np.array([[0.5, 0.5, 1.],
                     [1., 0., 0.]]), 
            np.array([[0., 0., 1.],
                     [1., 0., 0.]]), 
            np.array([[0., 0., 1.],
                     [1., 0., 0.]])
        )
    ]
)
def test_vector_cartesian_to_cylindrical(input_points, input_vectors, expected) -> None:
    from snapanalysis.utils import vector_cartesian_to_cylindrical

    assert np.allclose(
        vector_cartesian_to_cylindrical(input_points, input_vectors), expected
    ), "Cylindrical vector transform failed!"

    return None