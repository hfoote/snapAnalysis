import pytest
from snapanalysis.snap import snapshot

CDM_TEST_SNAP_PATH = "tests/example_snaps/cdm_snaps/"
FDM_TEST_SNAP_PATH = "tests/example_snaps/fdm_snaps/"

@pytest.fixture
def dm_snap():
    s = snapshot(CDM_TEST_SNAP_PATH + "snap_000.hdf5", 1)
    return s

@pytest.fixture
def fdm_snap():
    s = snapshot(FDM_TEST_SNAP_PATH + "snap_000.hdf5", 1)
    return s

@pytest.fixture
def star_snap():
    s = snapshot(CDM_TEST_SNAP_PATH + "snap_000.hdf5", 4)
    return s