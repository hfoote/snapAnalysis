# Contains the main snap class of snapAnalysis, which stores snapshots in a flexible format. 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import astropy
from astropy import units as u
from astropy import constants as const
from snapAnalysis import utils
import warnings

class snap:
	''' The main class of snapAnalysis, which reads and stores a single particle type from a single gagdet/arepo format hdf5 snapshot. 
	Methods provide common analysis routines such as computing density fields, centering, rotations, etc.
	'''

	def __init__(self, filename:str, ptype:int) -> None:
		'''__init__ creates a snap object and stores useful metadata.

		Parameters
		----------
		filename : str
			Name of the simulation snapshot to read
		ptype : int
			Integer particle type 
		'''

		self.filename = filename
		self.ptype = ptype
		self.fdm = False

		# keep data fields empty until populated later
		self.data_fields = {'Coordinates':None, 
							'Velocities':None, 
							'Masses':None, 
							'ParticleIDs':None, 
							'Potential':None, 
							'Accelerations':None, 
							'PsiRe':None, 
							'PsiIm':None
							}

		with h5py.File(self.filename, 'r') as f:

			if 'BECDM' in f['Config'].attrs.keys():
				self.fdm = True
				self.N_cells = int(f['Config'].attrs['PMGRID'])
				self.m_axion_Ev = f['Parameters'].attrs['AxionMassEv']*u.eV 

			self.vel_unit = (f['Parameters'].attrs['UnitVelocity_in_cm_per_s']*u.cm/u.s).to(u.km/u.s)
			self.length_unit = (f['Parameters'].attrs['UnitLength_in_cm']*u.cm).to(u.kpc)
			self.mass_unit = (f['Parameters'].attrs['UnitMass_in_g']*u.g).to(u.Msun)
			self.time_unit = (self.length_unit / self.vel_unit).to(u.Gyr)
			
			self.box_size = f['Header'].attrs['BoxSize']
			self.box_half = self.box_size/2.
			self.time = f['Header'].attrs['Time']*self.time_unit
			self.N = f['Header'].attrs['NumPart_ThisFile'][ptype]

		self.G = const.G.to(self.length_unit * self.vel_unit**2 / self.mass_unit)
		self.field_units = {'Coordinates':self.length_unit, 
							'Velocities':self.vel_unit, 
							'Masses':self.mass_unit, 
							'ParticleIDs':u.dimensionless_unscaled, 
							'Potential':self.vel_unit**2, 
							'Accelerations':self.vel_unit**2/self.length_unit, 
							'PsiRe':u.dimensionless_unscaled, 
							'PsiIm':u.dimensionless_unscaled
							}

	def print_structure(self):
		'''Displays the file structure of the snapshot.
		'''
		print('File structure of '+self.filename)
		with h5py.File(self.filename, 'r') as f:
			names = f.keys() # get group names
			for name in names:
				print('\nGROUP: '+ name) # print group name

				print('KEYS:')
				attList = f[name].attrs.keys()
				for atr in attList: # print all attributes in that group
					print('       '+atr)

				print('DATA SETS:')
				dataList = f[name]
				for data in dataList:
					print('       '+data)

	def print_header(self):
		'''Displays all info contained in the header of the snapshot.

		'''
		print('___________________________________________')
		print('Header of '+self.filename)
		with h5py.File(self.filename, 'r') as f:
			try: # open header, if it doesn't work, there is no header
				head = f['Header']
			except KeyError:
				print("Either the header does not have the name 'Header' or it does not exist.")
				raise
			# print each key and what it contains
			print('\nKEYS')
			print('--------')
			attList = head.attrs.keys()
			for atr in attList: # print all attributes in the header, with their contents
				print(atr+':',head.attrs[atr])

			print('\nDATA SETS')
			print('--------')
			for data in head: # print all data sets in the header, with their contents
				print(data+':',head[data])
				
	def read_field(self, field:str) -> np.ndarray:
		'''read_field returns one data field of one particle type 

		Parameters
		----------
		field : str
			data field

		Returns
		-------
		np.ndarray
			data field
		'''

		with h5py.File(self.filename, 'r') as f:
			data = np.array(f[f'PartType{self.ptype}/{field}'])

		return data
	
	def read_masstable(self) -> u.Quantity:
		'''read_masstable returns the masstable

		Returns
		-------
		astropy Quantity
			masstable
		'''

		with h5py.File(self.filename, 'r') as f:
			masstable = f['Header'].attrs['MassTable'][self.ptype]

		return masstable
	

	def check_if_field_read(self, field:str) -> bool:
		'''check_if_field_read checks to see whether a data field has already been stored

		Parameters
		----------
		field : str
			field name

		Returns
		-------
		bool
			True if field has already been read, False otherwise
		'''
		
		if self.data_fields[field] == None:
			return False
		return True

	def arrange_fields(self, indices:np.ndarray) -> None:
		'''arrange_fields re-arranges all existing data fields according to the input indices.
		Useful for sorting the simulation by particle ID, for instance. 

		Parameters
		----------
		indices : np.ndarray
			field name

		'''
		for field in self.data_fields.keys():
			if not self.check_if_field_read(field):
				continue
			self.data_fields[field] = self.data_fields[field][indices]

	def select_particles(self, ID_range:tuple) -> None:
		'''select_particles selects the desired ID range 

		Parameters
		----------
		ID_range : tuple
			min and max ID values to keep (inclusive)
		'''

		IDs = self.load_particle_data('ParticleIDs')

		for field in self.data_fields.keys():
			self.data_fields[field] = self.data_fields[field][np.where((IDs >= ID_range[0]) & (IDs <= ID_range[1]))]

	def load_particle_data(self, fields:list) -> None:
		'''load_particle_data reads particle data for a field if it doesn't already exist

		Parameters
		----------
		fields : list
			list containing desired data fields, e.g.
			['Coordinates', 'Velocities', 'ParticleIDs', 'Masses', 'Potential', 'Acceleration', 'PsiRe', 'PsiIm']
		'''

		for field in fields:
			if self.check_if_field_read(field):
				continue

			# special cases for FDM
			if ((field == 'PsiRe') or (field == 'PsiIm')):
				assert (self.fdm and self.ptype == 1), "Non-FDM particle types do not contain wavefunction data!"
			
			if (field == 'Velocity') and (self.ptype == 1) and self.fdm:
				self.data_fields[field] = self.get_FDM_velocities()

			# special case for masses, populate mass array from masstable or wavefunction if necessary
			if field == 'Masses':
				if (self.ptype == 1) and self.fdm:
					self.data_fields[field] = np.abs(self.read_field('PsiRe')**2 + self.read_field('PsiIm')**2) * self.mass_unit / self.length_unit**3
				else:
					try:
						self.data_fields[field] = self.read_field(field) * self.mass_unit
					except KeyError: 
						self.data_fields[field] = np.full(self.N, self.read_masstable()) * self.mass_unit

			else:
				self.data_fields[field] = self.read_field(field) * self.field_units[field]

	def get_FDM_velocities(self) -> None:
		'''Calculates the FDM velocity field based on the wavefunction outputs, via the phase gradient method:
		v = hbar/m gradient(phase). Based on code from Philip Mocz
		'''

		self.load_particle_data(1, ['ParticleIDs', 'Coordinates', 'PsiRe', 'PsiIm'])

		m = self.m_axion_ev	/ const.c**2

		# reshape and sort arrays 
		IDsort = np.argsort(IDs) # indices to sort the arrays in ID order
		self.arrange_fields(IDsort)

		# sort and reshape IDs and positions to their final forms
		IDs = self.data_fields['ParticleIDs']
		IDs = np.reshape(IDs, (self.N_cells, self.N_cells, self.N_cells), order='F')
		IDs = np.reshape(IDs, (self.N,), order='F')
		# do the same with positions
		pos = self.data_fields['Coordinates']
		pos = np.reshape(pos, (self.N_cells, self.N_cells, self.N_cells,3), order='F')
		dx = pos[0,1,0,0] - pos[0,0,0,0] # cell spacing
		
		pos_x = np.reshape(pos[:,:,:,0], (self.N, 1), order='F') # reshape back to snapshot shape
		pos_y = np.reshape(pos[:,:,:,1], (self.N, 1), order='F')
		pos_z = np.reshape(pos[:,:,:,2], (self.N, 1), order='F')
		pos = np.concatenate([pos_x, pos_y, pos_z], axis=1)

		# get psi into a single complex number and reshape it to look like the box
		psi = self.data_fields['psiRe'] + 1.j*self.data_fields['psiIm']
		psi = psi[IDsort]
		psi = np.reshape(psi, (self.N_cells, self.N_cells, self.N_cells), order='F')

		# phase gradient approach, following Philip Mocz's code
		phase = np.angle(psi)

		# roll arrays for finite differencing
		vx = np.roll(phase, -1, axis=1) - np.roll(phase, 1, axis=1)
		vy = np.roll(phase, -1, axis=0) - np.roll(phase, 1, axis=0)
		vz = np.roll(phase, -1, axis=2) - np.roll(phase, 1, axis=2)
		
		# roll over phases to be periodic if needed
		vx[vx > np.pi] = vx[vx > np.pi] - 2.*np.pi
		vx[vx <= -np.pi] = vx[vx <= -np.pi] + 2.*np.pi
		vy[vy > np.pi] = vy[vy > np.pi] - 2.*np.pi
		vy[vy <= -np.pi] = vy[vy <= -np.pi] + 2.*np.pi
		vz[vz > np.pi] = vz[vz > np.pi] - 2.*np.pi
		vz[vz <= -np.pi] = vz[vz <= -np.pi] + 2.*np.pi

		# calculate velocities
		vx = (vx / (2.*dx) / m * const.hbar).to(self.vel_unit)
		vy = (vy / (2.*dx) / m * const.hbar).to(self.vel_unit)
		vz = (vz / (2.*dx) / m * const.hbar).to(self.vel_unit)
	
		# get velocities back into the same shape as they're output
		vel_x = np.reshape(vx, (self.N,1), order='F') # reshape back to snapshot shape
		vel_y = np.reshape(vy, (self.N,1), order='F')
		vel_z = np.reshape(vz, (self.N,1), order='F')
		
		self.data_fields['Velocities'] = np.concatenate([vel_x, vel_y, vel_z], axis=1)

	def read_all(self) -> None:
		'''read_all wrapper to load all "standard" fields (IDs, pos, vel, mass)
		'''

		self.load_particle_data(['ParticleIDs', 'Coordinates', 'Velocities', 'Masses'])

	def apply_center(self, pos_center:np.ndarray, vel_center:np.ndarray=None) -> None:
		'''center_particles centers particles on a specified center in position and velocity

		Parameters
		----------
		pos_center : np.ndarray
			[x, y, z] Position center
		vel_center : None or np.ndarray, optional
			[vx, vy, vz] Velocity center
		'''

		assert self.check_if_field_read('Coordinates'), "Particle positions not loaded!"
		self.data_fields['Coordinates'] -= pos_center[None, :]

		if vel_center != None:
			assert self.check_if_field_read('Velocities'), "Particle velocities not loaded!"
			self.data_fields['Velocities'] -= vel_center[None, :]

	def find_position_center(self, vol_dec:float=3., delta:float=0.01, N_min:int=1000) -> np.ndarray:
		'''find_position_center finds the center of mass of the snapshot via the shrinking-spheres method.

		Parameters
		----------
		vol_dec : float, optional
			Factor by which the sphere radius is reduced during an iteration, by default 3.
		delta : float, optional
			Tolerance, stop when change in COM is less than delta, by default 0.01
		Nmin : int, optional
			Minimum number of particles allowed within the sphere, by default 1000

		Returns
		-------
		np.ndarray
			[x, y, z] center of mass
		'''
		self.load_particle_data(['Masses', 'Coordinates'])

		# initial guess using every particle
		com = utils.com_define(self.data_fields['Masses'], self.data_fields['Coordinates'])
		r_COM = np.sqrt(np.sum(com**2))

		# make temp coordinates with respect to COM
		pos_new = self.data_fields['Coordinates'] - com[None, :]
		r_new = np.sqrt(np.sum(pos_new**2, axis=1))

		# shrink sphere
		r_max = max(r_new)/vol_dec
		change = 1000.0 # initialize change

		while (change > delta):
			# all particles within the reduced radius, starting from original coordinates
			index2 = np.where(r_new <= r_max)
			pos2 = self.data_fields['Coordinates'][index2]
			m2 = self.data_fields['Masses'][index2]

			# recompute with particles in reduced radius
			com2 = utils.com_define(m2, pos2)
			r_COM2 = np.sqrt(np.sum(com2**2))

			if len(m2) < N_min:
				warnings.warn("COM: Minimum number of particles reached before COM converged.")
				return com2
                                                                                    
			change = np.abs(r_COM - r_COM2).value                                                                                                                                                                    

			# Before loop continues, reset rmax, particle separations, and COM                                                                                                     
			r_max /= vol_dec                                                                          
			
			pos_new = self.data_fields['Coordinates'] - com2[None, :]
			r_new = np.sqrt(np.sum(pos_new**2, axis=1))
                                                  
			com = com2
			r_COM = r_COM2

		return com
	
	def find_velocity_center(self, pos_center:np.ndarray, r_max:u.Quantity=15.0*u.kpc) -> np.ndarray:
		'''find_velocity_center finds the COM velocity using particles within r_max.

		Parameters
		----------
		pos_center : np.ndarray
			[x, y, z] center of mass position
		r_max : u.Quantity, optional
			max distance from COM to use particles for the calculation, by default 15.0*u.kpc

		Returns
		-------
		np.ndarray
			[vx, vy, vz] COM velocity
		'''

		self.load_particle_data(['Masses', 'Coordinates', 'Velocities'])

		pos_new = self.data_fields['Coordinates'] - pos_center[None, :]
		r_new = np.sqrt(np.sum(pos_new**2, axis=1))

		index = np.where(r_new <= r_max)
		vel = self.data_fields['Velocities'][index]
		m = self.data_fields['Masses'][index]
		
		return utils.com_define(m, vel)
	
	def find_and_apply_center(self, com_kwargs:dict={}, vel_kwargs:dict={}) -> None:
		'''find_and_apply_center Calculates the positions and velocity center
		of the snapshot and centers the particle data on the COM location in phase-space. 

		Parameters
		----------
		com_kwargs : dict, optional
			kwargs for snap.find_position_center
		vel_kwargs : dict, optional
			kwargs for snap.find_velocity_center
		'''

		com_pos = self.find_position_center(**com_kwargs)
		com_vel = self.find_velocity_center(com_pos, **vel_kwargs)

		self.apply_center(com_pos, com_vel)