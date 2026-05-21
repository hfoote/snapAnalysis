# unit tests for snapanalysis.vis
from snapanalysis import vis
import astropy.units as u


def test_density_projection(mocker, dm_snap):
	mock_imshow = mocker.patch("matplotlib.pyplot.imshow")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	dens, xbins, ybins = vis.density_projection(
		dm_snap, axis=0, bins=100, slice_width=10.0*u.kpc, normalization="overdensity"
	)
	assert mock_imshow.called_with(
		dens, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
	)

	assert dens.shape == (100, 100)
	assert xbins.shape == (101,)
	assert ybins.shape == (101,)

	assert mock_show.called_once()

def test_density_profile(mocker, dm_snap):
	mock_plot = mocker.patch("matplotlib.pyplot.plot")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	r, rho = vis.density_profile(dm_snap, rmin=1.0, rmax=100.0, nbins=200)
	assert mock_plot.called_with(r, rho)

	assert r.shape == (200,)
	assert rho.shape == (200,)

	assert mock_show.called_once()


def test_anisotropy_profile(mocker, dm_snap):
	mock_plot = mocker.patch("matplotlib.pyplot.plot")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	r, beta = vis.anisotropy_profile(
		dm_snap, rmin=1.0, rmax=100.0, log_bins=True
	)
	assert mock_plot.called_with(r, beta)

	assert r.shape == (100,)
	assert beta.shape == (100,)

	assert mock_show.called_once()


def test_potential_projection(mocker, dm_snap):
	mock_imshow = mocker.patch("matplotlib.pyplot.imshow")
	mock_show = mocker.patch("matplotlib.pyplot.show")

	pot, xbins, ybins = vis.potential_projection(
		dm_snap, axis=1, bins=100, slice_width=20.0*u.kpc
	)
	assert mock_imshow.called_with(
		pot, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
	)

	assert pot.shape == (100, 100)
	assert xbins.shape == (101,)
	assert ybins.shape == (101,)

	assert mock_show.called_once()

