# unit tests for snapanalysis.vis
from snapanalysis import vis
import astropy.units as u


def test_density_projection(mocker, dm_snap):
	mock_imshow = mocker.patch("matplotlib.pyplot.imshow")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	dens, xbins, ybins = vis.density_projection(
		dm_snap, axis=0, bins=100, slice_width=10.0*u.kpc, normalization="overdensity"
	)
	mock_imshow.assert_called_once_with(
		dens, origin='lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
	)
	mock_show.assert_called_once()
	assert dens.shape == (100, 100)
	assert xbins.shape == (101,)
	assert ybins.shape == (101,)

	
def test_density_profile(mocker, dm_snap):
	mock_plot = mocker.patch("matplotlib.pyplot.plot")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	r, rho = vis.density_profile(dm_snap, rmin=1.0, rmax=100.0, nbins=200)
	mock_plot.assert_called_with(r, rho)
	mock_show.assert_called_once()
	assert r.shape == (200,)
	assert rho.shape == (200,)

	
def test_anisotropy_profile(mocker, dm_snap):
	mock_plot = mocker.patch("matplotlib.pyplot.plot")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	r, beta = vis.anisotropy_profile(
		dm_snap, rmin=1.0, rmax=100.0, log_bins=True
	)
	mock_plot.assert_called_with(r, beta)
	mock_show.assert_called_once()
	assert r.shape == (100,)
	assert beta.shape == (100,)


def test_potential_projection(mocker, dm_snap):
	mock_imshow = mocker.patch("matplotlib.pyplot.imshow")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	pot, xbins, ybins = vis.potential_projection(
		dm_snap, axis=1, bins=100, slice_width=20.0*u.kpc
	)
	mock_imshow.assert_called_once_with(
		pot, origin='lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
	)
	mock_show.assert_called_once()
	assert pot.shape == (100, 100)
	assert xbins.shape == (101,)
	assert ybins.shape == (101,)

