# Visualization functions for making plots of snapshot objects
import numpy as np
import matplotlib.pyplot as plt
import snapanalysis
from snapanalysis import utils
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
from astropy import units as u
import warnings

def density_projection(
	snap: snapanalysis.snap.snapshot,
	axis: int = 2,
	bins: int | list = [200, 200],
	mass_weight: bool = True,
	plot: bool = True,
	plot_name: bool | str = False,
	slice_width: None | float = None,
	normalization: bool = "surface",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""density_projection generates a density histogram projected along
	the specified axis.

	Parameters
	----------
	axis : int, optional
			Axis to project along (x=0, y=1, z=2), by default 2
	bins : int or list, optional
			bin specification passed to np.histogram, by default [200,200]
	mass_weight : bool, optional
			weight particles by their mass, by default True
	plot : bool, optional
			create a plot of the density histogram, by default True
	plot_name : bool or str, optional
			if not False, saves the plot under this file name, by default False
	slice_width : None | float, optional
			A distance in the length units of the snapshot
	from the midplane along the specified axis to include, by default None
	normalization : str, optional
			"surface" (default) returns surface density,
			"overdensity" returns the density contrast,
			"volume" returns volume density, requires slice_width to be nonzero.

	Returns
	-------
	np.ndarray :
			Density histogram
	np.ndarray :
			x bin edges
	np.ndarray :
			y bin edges
	"""

	snap.load_particle_data(["Coordinates", "Masses"])
	pos = snap.data_fields["Coordinates"]
	m = snap.data_fields["Masses"]

	i, j = utils.set_axes(axis)

	if slice_width is not None:
		slice = utils.get_vslice_indices(pos, slice_width, axis)
		pos = pos[slice]
		m = m[slice]

	if mass_weight:
		weights = m.value
	else:
		weights = None

	dens, xbins, ybins = np.histogram2d(
		pos[:, j].value, pos[:, i].value, bins=bins, weights=weights
	)

	# normalization
	if normalization == "overdensity":
		dens = dens / np.mean(dens) - 1.0
	elif normalization == "surface":
		bin_volume = (xbins[1] - xbins[0]) * (ybins[1] - ybins[0])
		dens /= bin_volume
	elif (normalization == "volume") and (slice_width is not None):
		bin_volume = (
			(xbins[1] - xbins[0]) * (ybins[1] - ybins[0]) * 2.0 * slice_width.value
		)
		dens /= bin_volume
	else:
		raise ValueError("Invalid normalization!")

	if plot:
		plt.imshow(
			dens, origin="lower", extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
		)

		if plot_name:
			plt.savefig(plot_name)
			plt.close()
		else:
			plt.show()

	return dens, xbins, ybins

def density_profile(
	snap: snapanalysis.snap.snapshot,
	rmin: float = 0.0,
	rmax: float = 150.0,
	nbins: int = 100,
	log_bins: bool = False,
	plot: bool = True,
	plot_name: bool | str = False,
) -> tuple[np.ndarray, np.ndarray]:
	"""density_profile computes the density profile of a halo
	Note: snpshot must be centered (with e.g. snap.apply_center()) first!

	Parameters
	----------
	rmin : float, optional
			min. radius, by default 0.
	rmax : float, optional
			max. radius, by default 150.
	nbins : int, optional
			number of radial bins, by default 100
	log_bins : bool, optional
			use logarithmic bin spacing, by default False
	plot : bool, optional
			creates a plot of the density profile, by default True
	plotName : bool | str, optional
			if not False, saves the plot under this file name, by default False
	Returns
	-------
	np.ndarray :
			Radial points / bin centers
	np.ndarray :
			value of the density profile at the bin centers
	"""

	snap.load_particle_data(["Coordinates", "Masses"])
	pos = snap.data_fields["Coordinates"]
	m = snap.data_fields["Masses"]

	if log_bins:
		bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
	else:
		bins = np.linspace(rmin, rmax, nbins + 1)

	r = np.sqrt(np.sum(pos**2, axis=1))

	shell_volume = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

	hist, edges = np.histogram(
		r.to(u.kpc).value, bins=bins, weights=m.to(u.Msun).value
	)
	rho = hist / shell_volume

	bin_centers = (bins[1:] + bins[:-1]) / 2.0

	if plot:
		plt.plot(bin_centers, rho)
		plt.ylabel("$\\rho(r)$ $[M_{\\odot}/kpc^{3}]$")
		plt.xlabel("r [kpc]")

		if plot_name:
			plt.savefig(plot_name)
			plt.close()
		else:
			plt.show()

	return bin_centers, rho

def anisotropy_profile(
	snap: snapanalysis.snap.snapshot,
	rmin: float = 0.0,
	rmax: float = 150.0,
	nbins: int = 100,
	log_bins: bool = False,
	plot: bool = True,
	plot_name: bool | str = False,
) -> tuple[np.ndarray, np.ndarray]:
	"""anisotropy_profile calculates the velocity anisotropy profile of a halo,
	given by Eqn 4.61 of Binney & Tremaine:

	:math:`\\beta(r) = 1 - \\frac{\\sigma_\\theta^2 + \\sigma_\\phi^2}{2 \\sigma_r^2}`

	Parameters
	----------
	rmin : float, optional
			min. radius, by default 0.
	rmax : float, optional
			max. radius, by default 150.
	nbins : int, optional
			number of radial bins, by default 100
	log_bins : bool, optional
			If True, make logarithmically spaced bins instead of linearly
			spaced bins, by default False
	plot : bool, optional
			show a plot of the computed profile, by default True
	plot_name : bool | str, optional
			if a str, saves the plot with this filename, by default False

	Returns
	-------
	np.ndarray :
			radial points / bin centers
	np.ndarray :
			value of the anisotropy profile at the bin centers
	"""

	if (not snap.pos_centered) | (not snap.vel_centered):
		warnings.warn(
			"Snapshot has not been centered! Calculating an anisotropy profile on \
				an un-centered snapshot may give garbage results."
		)

	snap.load_particle_data(["Coordinates", "Velocities"])
	pos = snap.data_fields["Coordinates"]
	pos_spherical = utils.cartesian_to_spherical(pos)

	r = pos_spherical[:, 0]
	theta = pos_spherical[:, 1]
	phi = pos_spherical[:, 2]

	vel = snap.data_fields["Velocities"]
	vx = vel[:, 0]
	vy = vel[:, 1]
	vz = vel[:, 2]

	v_r = (
		np.sin(theta) * np.cos(phi) * vx
		+ np.sin(theta) * np.sin(phi) * vy
		+ np.cos(theta) * vz
	)
	v_theta = (
		np.cos(theta) * np.cos(phi) * vx
		+ np.cos(theta) * np.sin(phi) * vy
		- np.sin(theta) * vz
	)
	v_phi = -np.sin(phi) * vx + np.cos(phi) * vy

	if log_bins:
		bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
	else:
		bins = np.linspace(rmin, rmax, nbins + 1)

	sigma_r, _, _ = binned_statistic(r, v_r, statistic="std", bins=bins)
	sigma_theta, _, _ = binned_statistic(r, v_theta, statistic="std", bins=bins)
	sigma_phi, _, _ = binned_statistic(r, v_phi, statistic="std", bins=bins)

	beta = 1.0 - ((sigma_phi**2.0 + sigma_theta**2.0) / (2.0 * sigma_r**2.0))

	bin_centers = (bins[1:] + bins[:-1]) / 2.0

	if plot:
		plt.plot(bin_centers, beta)
		plt.ylabel("$\\beta(r)$")
		plt.xlabel("r [kpc]")

		if plot_name:
			plt.savefig(plot_name)
			plt.close()
		else:
			plt.show()

	return bin_centers, beta

def potential_projection(
	snap: snapanalysis.snap.snapshot,
	axis: int = 2,
	bins: int | list = [200, 200],
	plot: bool = True,
	plot_name: bool | str = False,
	slice_width: None | float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""potential_projection bins the particles in the specified plane and finds
	the mean value of the potential within each bin.

	Parameters
	----------
	axis : int, optional
			Axis to project along (x=0, y=1, z=2), by default 2
	bins : int | list, optional
			bin specification passed to scipy's binned_statistic_2d,
			by default [200,200]
	plot : bool, optional
			create a plot of the potential, by default True
	plot_name : bool | str, optional
			if not False, saves the plot under this file name, by default False
	slice_width : None | float, optional
			Leave as None to use the whole box. Otherwise, provide a distance
			in the length units of the snapshot from the midplane along the
			specified axis to include, by default None

	Returns
	-------
	np.ndarray :
			Binned potential field
	np.ndarray :
			x bin edges
	np.ndarray :
			y bin edges
	"""

	snap.load_particle_data(["Coordinates", "Potential"])
	pos = snap.data_fields["Coordinates"]
	pot = snap.data_fields["Potential"]

	i, j = utils.set_axes(axis)

	if slice_width is not None:
		slice = utils.get_vslice_indices(pos, slice_width, axis)
		pos = pos[slice]
		pot = pot[slice]

	phi, xbins, ybins, _ = binned_statistic_2d(
		pos[:, j], pos[:, i], pot, bins=bins, statistic="mean"
	)

	if plot:
		plt.imshow(
			phi, origin="lower", extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]]
		)

		if plot_name:
			plt.savefig(plot_name)
			plt.close()
		else:
			plt.show()

	return phi, xbins, ybins
