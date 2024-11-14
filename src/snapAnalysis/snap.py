# Contains the main snap class of snapAnalysis, which stores snapshots in a flexible format. 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import astropy
from astropy import units as u
from astropy import constants as const

class snap:
	''' The main class of snapAnalysis, which reads and stores a single gagdet/arepo format hdf5 snapshot. 
	Methods provide common analysis routines such as computing density fields, centering, rotations, etc.
	'''

	def __init__(self, filename:str) -> None:
		'''__init__ creates a snap object and stores useful metadata.

		Parameters
		----------
		filename : str
			Name of the simulation snapshot to read
		'''

		self.filename = filename
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

			if 'BECDM' in f['Config'].attrs().keys():
				self.fdm = True
				self.N_cells = int(f['Config'].attrs['PMGRID'])
				self.m_axion_Ev = f['Parameters'].attrs['AxionMassEv']*u.eV 

			self.vel_unit = (f['Parameters'].attrs('UnitVelocity_in_cm_per_s')*u.cm/u.s).to(u.km/u.s)
			self.length_unit = (f['Parameters'].attrs('UnitLength_in_cm')*u.cm).to(u.kpc)
			self.mass_unit = (f['Parameters'].attrs('UnitMass_in_g')*u.g).to(u.Msun)
			self.time_unit = (self.length_unit / self.vel_unit).to(u.Gyr)
			
			self.box_size = f['Header'].attrs['BoxSize']
			self.box_half = self.box_size/2.
			self.time = f['Header'].attrs['Time']*self.time_unit
			self.N_types = f['Header'].attrs['NumPart_ThisFile']

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
				
	def read_field(self, ptype:int, field:str) -> np.ndarray:
		'''read_field returns one data field of one particle type 

		Parameters
		----------
		ptype : int
			particle type
		field : str
			data field

		Returns
		-------
		np.ndarray
			data field
		'''

		with h5py.File(self.filename, 'r') as f:
			data = np.array(f[f'PartType{ptype}/{field}'])

		return data
	
	def read_masstable(self) -> u.Quantity:
		'''read_masstable returns the masstable

		Returns
		-------
		astropy Quantity
			masstable
		'''

		with h5py.File(self.filename, 'r') as f:
			masstable = f['Header'].attrs['MassTable'] * self.mass_unit

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


	def get_particle_data(self, ptype:int, fields:list) -> None:
		'''get_particle_data reads particle data for a field if it doesn't already exist

		Parameters
		----------
		ptype : int
			particle type 
		fields : list
			list containing desired data fields, e.g.
			['Coordinates', 'Velocities', 'ParticleIDs', 'Masses', 'Potential', 'Acceleration', 'PsiRe', 'PsiIm']
		'''

		for field in fields:
			if self.check_if_field_read(field):
				continue

			# special cases for FDM
			if ((field == 'PsiRe') or (field == 'PsiIm')):
				assert (self.fdm and ptype == 1), "Non-FDM particle types do not contain wavefunction data!"
			
			if (field == 'Velocity') and (ptype == 1) and self.fdm:
				self.data_fields[field] = self.get_FDM_velocities()

			# special case for masses, populate mass array from masstable or wavefunction if necessary
			if field == 'Masses':
				if (ptype == 1) and self.fdm:
					self.data_fields[field] = np.abs(self.read_field(ptype, 'PsiRe')**2 + self.read_field(ptype, 'PsiIm')**2) * self.mass_unit / self.length_unit**3
				else:
					try:
						self.data_fields[field] = self.read_field(ptype, field) * self.mass_unit
					except KeyError: 
						self.data_fields[field] = np.full(self.N_types[ptype], self.read_masstable()[ptype]) * self.mass_unit

			else:
				self.data_fields[field] = self.read_field(ptype, field) * self.field_units[field]

	def get_FDM_velocities(self) -> None:
		'''Calculates the FDM velocity field based on the wavefunction outputs, via the phase gradient method:
		v = hbar/m gradient(phase). Based on code from Philip Mocz
		'''

		self.get_particle_data(1, ['ParticleIDs', 'Coordinates', 'PsiRe', 'PsiIm'])

		m = self.m_axion_ev	/ const.c**2

		# reshape and sort arrays 
		IDsort = np.argsort(IDs) # indices to sort the arrays in ID order
		self.arrange_fields(IDsort)

		# sort and reshape IDs and positions to their final forms
		IDs = self.data_fields['ParticleIDs']
		IDs = np.reshape(IDs, (self.N_cells, self.N_cells, self.N_cells), order='F')
		IDs = np.reshape(IDs, (self.N_types[1],), order='F')
		# do the same with positions
		pos = self.data_fields['Coordinates']
		pos = np.reshape(pos, (self.N_cells, self.N_cells, self.N_cells,3), order='F')
		dx = pos[0,1,0,0] - pos[0,0,0,0] # cell spacing
		
		pos_x = np.reshape(pos[:,:,:,0], (self.N_types[1], 1), order='F') # reshape back to snapshot shape
		pos_y = np.reshape(pos[:,:,:,1], (self.N_types[1], 1), order='F')
		pos_z = np.reshape(pos[:,:,:,2], (self.N_types[1], 1), order='F')
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
		vel_x = np.reshape(vx, (self.N_types[1],1), order='F') # reshape back to snapshot shape
		vel_y = np.reshape(vy, (self.N_types[1],1), order='F')
		vel_z = np.reshape(vz, (self.N_types[1],1), order='F')
		
		self.data_fields['Velocities'] = np.concatenate([vel_x, vel_y, vel_z], axis=1)








				