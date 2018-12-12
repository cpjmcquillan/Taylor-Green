import os
import re
from shutil import move, copyfile

import numpy as np
import matplotlib.pyplot as plt
# from mayavi.mlab import *
import h5py

# Order of finite difference scheme used.
ORDER = 4
HALO_POINTS = 2
RE = 800

# Domain extents (cube).
DOMAIN = (0.0, 2*np.pi)

# Grid points and respective timesteps
GRIDS = [32, 64, 128]
TIMESTEPS = [0.005, 0.005, 0.0025]

def grid_compare(grids, timesteps, domain, folder_path):
	"""
	Function to carry out post processing for a Taylor-Green grid refinement study.
	"""
	postpros = []

	for idx, grid in enumerate(grids):
		# Create postpro object.
		postpro = TaylorGreenPostPro(grid, timesteps[idx], domain, folder_path)
		postpros.append(postpro)
		# Run postpro.
		print("Running postprocessing for Taylor Green Vortex: n={}...".format(grid))
		postpro()

	# Plot KE comparison.
	for idx, postpro in enumerate(postpros):
		energies = [postpro.flow_data[time]['E_k'] for time in postpro.times]
		plt.plot(postpro.times, energies, label='N={}'.format(grids[idx]))
	plt.ylabel("Non-dimensional kinetic energy")
	plt.xlabel("Non-dimensional time")
	plt.xlim(0.0, 20.0)
	plt.legend()
	# plt.show()
	plt.savefig(os.path.join(folder_path, 'output_data', 'kinetic_energy_comparison.png'))
	plt.gcf().clear()

	# Plot DR comparison.
	for idx, postpro in enumerate(postpros):
		dissipations = [postpro.flow_data[time]['eps'] for time in postpro.times]
		plt.plot(postpro.times, dissipations, label='N={}'.format(grids[idx]))
	plt.ylabel("Non-dimensional dissipation rate")
	plt.xlabel("Non-dimensional time")
	plt.xlim(0.0, 20.0)
	plt.legend()
	# plt.show()
	plt.savefig(os.path.join(folder_path, 'output_data', 'dissipation_rate_comparison.png'))
	plt.gcf().clear()

class TaylorGreenPostPro(object):
	"""
	Object for a Taylor Green Vortex Postpro Run.
	1. Open files.
	2. Extract flow variables.
	3. Calculate derivatives.
	4. Calculate vorticities.
	5. Calculate kinetic energy.
	6. Calculate dissipation rate.
	7. Plot KE and DR over time.
	8. Plot vorticity contours.
	"""
	def __init__(self, grid_points, timestep, domain, folder_path):
		# Assign variables.
		self.order = ORDER
		self.halo_points = HALO_POINTS
		self.dt = timestep
		self.Re = RE
		self.n = [grid_points, grid_points, grid_points]
		self.domain = [domain, domain, domain]
		self.folder_path = folder_path

		# Output path of data.
		self.output_path = os.path.join(self.folder_path, 'output_data', str(grid_points))

		# Find files in output directory.
		fnames = [f for f in os.listdir(self.output_path) if (os.path.isfile(os.path.join(self.output_path,f)) and '.h5' in f)]

		# Calculate simulation time based on dt and file name.
		sim_times = []
		for fname in fnames:
			sim_times.append(self.dt*int(re.findall('\d+', fname)[0]))

		# Sort them to be ascending in time.
		self.file_names = [x for _,x in sorted(zip(sim_times, fnames))]
		self.times = np.array([x for x,_ in sorted(zip(sim_times, fnames))])

	def __call__(self, *args, **kwargs):
		# Create coordinate arrays.
		self._create_coords()

		# Dictionary containing data at each time.
		flow_data = {time: {} for time in self.times}

		# Loop over output files and extract flow variables.
		for file_no, file in enumerate(self.file_names):
			print("Processing simulation data at time t=%.2f..." % self.times[file_no])

			# Open file.
			f, group = self._read_file(os.path.join(self.output_path,file))
			
			# Extract flow variables.
			rho, u, v, w, p = self._extract_flow_variables(group)

			# Calculate derivatives.
			du_dx = self._derivative(u, 0)
			dv_dx = self._derivative(v, 0)
			dw_dx = self._derivative(w, 0)
			du_dy = self._derivative(u, 1)
			dv_dy = self._derivative(v, 1)
			dw_dy = self._derivative(w, 1)
			du_dz = self._derivative(u, 2)
			dv_dz = self._derivative(v, 2)
			dw_dz = self._derivative(w, 2)

			# Flow variables without halos.
			rho_nohalo = rho[self.halo_points:-self.halo_points,
							 self.halo_points:-self.halo_points,
							 self.halo_points:-self.halo_points]
			u_nohalo = u[self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points]
			v_nohalo = v[self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points]
			w_nohalo = w[self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points]
			p_nohalo = p[self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points,
						 self.halo_points:-self.halo_points]
			du_dx_nohalo = du_dx[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dv_dx_nohalo = dv_dx[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dw_dx_nohalo = dw_dx[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			du_dy_nohalo = du_dy[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dv_dy_nohalo = dv_dy[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dw_dy_nohalo = dw_dy[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			du_dz_nohalo = du_dz[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dv_dz_nohalo = dv_dz[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]
			dw_dz_nohalo = dw_dz[self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points,
								 self.halo_points:-self.halo_points]

			# Calculate vorticities.
			omega_x = dw_dy_nohalo - dv_dz_nohalo
			omega_y = du_dz_nohalo - dw_dx_nohalo
			omega_z = dv_dx_nohalo - du_dy_nohalo
			vorticities = [omega_x, omega_y, omega_z]

			# Calculate mean density.
			rho_mean = np.sum(rho_nohalo)/np.size(rho_nohalo)

			# Calculate KE field. E_k = 1/(rho_ref*V)int_V{0.5*rho*u_j*u_j}dV
			E = 0.5 * rho_nohalo * (u_nohalo**2 + v_nohalo**2 + w_nohalo**2) 

			# KE grid based.
			areas = []
			for k in range(self.n[2]):
				lines = []
				for j in range(self.n[1]):
					line = np.sum(E[k,j,:])
					lines.append(line)
				area = np.sum(lines)
				areas.append(area)
			int_E = np.sum(areas)

			E_k = int_E / (rho_mean * self.n[0]**3)

			# Calculate dissipation rate. eps = 1/(Re*V)int_V{omega_i*omega_i}dV
			enstrophy = omega_x**2 + omega_y**2 + omega_z**2

			# DR grid based.
			areas = []
			for k in range(self.n[2]):
				lines = []
				for j in range(self.n[1]):
					line = np.sum(enstrophy[k,j,:])
					lines.append(line)
				area = np.sum(lines)
				areas.append(area)
			int_enstrophy = np.sum(areas)

			epsilon = int_enstrophy / (self.Re * self.n[0]**3)

			# # KE trapz. Needs adjusting to work correctly for a non-cube domain. 
			# Integrate KE field over domain.
			# x_grid = self.coordinates[0][0,0,:]			# grid in x direction.
			# y_grid = self.coordinates[1][0,:,0]			# grid in y direction.
			# z_grid = self.coordinates[2][:,0,0]			# grid in z direction.
			# areas = []
			# for k in range(self.n[2]):					# calculate area integral at each z slice.
			# 	lines = []
			# 	for j in range(self.n[1]):				# calculate line integral at each y location.
			# 		line = np.trapz(E[k,j,:], x_grid)
			# 		lines.append(line)
			# 	area = np.trapz(lines, y_grid)
			# 	areas.append(area)
			# int_E = np.trapz(areas, z_grid)				# calculate volume integral

			# Calculate volume of domain.
			# volume = (self.domain[0][1] - self.domain[0][0]) * \
			# 		 (self.domain[1][1] - self.domain[1][0]) * \
			# 		 (self.domain[2][1] - self.domain[2][0])
			# volume = (2 * np.pi)**3

			# E_k = int_E / (rho_mean * volume)

			# Calculate dissipation rate. eps = 1/(Re*V)int_V{omega_i*omega_i}dV

			# # DR trapz. Needs adjusting to work correctly for a non-cube domain. 
			# areas = []
			# for k in range(self.n[2]):
			# 	lines = []
			# 	for j in range(self.n[1]):
			# 		line = np.trapz(enstrophy[k,j,:], x_grid)
			# 		lines.append(line)
			# 	area = np.trapz(lines, y_grid)
			# 	areas.append(area)
			# int_enstrophy = np.trapz(areas, z_grid)

			# epsilon = int_enstrophy / (Re * volume)

			# 3D iso-contour plot of vorticity magnitude. INCOMPLETE
			# vorticity_mag = np.sqrt(enstrophy)
			# iso = contour3d(self.coordinates[0][0,0,:]/(2*np.pi), self.coordinates[1][0,:,0]/(2*np.pi), self.coordinates[2][:,0,0]/(2*np.pi),
			# 				vorticity_mag)
			
			# Store data.
			flow_data[self.times[file_no]]['rho'] = rho_nohalo
			flow_data[self.times[file_no]]['u'] = u_nohalo
			flow_data[self.times[file_no]]['v'] = v_nohalo
			flow_data[self.times[file_no]]['w'] = w_nohalo
			flow_data[self.times[file_no]]['p'] = p_nohalo
			flow_data[self.times[file_no]]['E_k'] = E_k
			flow_data[self.times[file_no]]['du_dx'] = du_dx_nohalo
			flow_data[self.times[file_no]]['dv_dx'] = dv_dx_nohalo
			flow_data[self.times[file_no]]['dw_dx'] = dw_dx_nohalo
			flow_data[self.times[file_no]]['du_dy'] = du_dy_nohalo
			flow_data[self.times[file_no]]['dv_dy'] = dv_dy_nohalo
			flow_data[self.times[file_no]]['dw_dy'] = dw_dy_nohalo
			flow_data[self.times[file_no]]['du_dz'] = du_dz_nohalo
			flow_data[self.times[file_no]]['dv_dz'] = dv_dz_nohalo
			flow_data[self.times[file_no]]['dw_dz'] = dw_dz_nohalo
			flow_data[self.times[file_no]]['eps'] = epsilon
			flow_data[self.times[file_no]]['vorticities'] = vorticities

			self.flow_data = flow_data

			# Close file.
			f.close()

		# KE and DR Plots.
		self._plots()

		# Vorticity Plots.
		print("Plotting vorticity contours...")
		for time in self.times:
			# Omega_x plots through X.
			# self._vorticity_plot(time, 3, 0)
			# Omega_y plots through Y.
			# self._vorticity_plot(time, 3, 1)
			# Omega_z plots through Z.
			self._vorticity_plot(time, 3, 2)

		print("Postpro complete.")

	def _vorticity_plot(self, time, slices, direction):
		"""
		Function to create sweep of vorticity contour plots,
		overlayed with velocity vectors in a given direction.
		"""
		# Create directory.
		folder = os.path.join(self.output_path, 'plots', 'vorticity', str(time))
		if not os.path.exists(folder):
			os.makedirs(folder)

		# Given number of slices requested, determine indices gap. For evenly distributed cube domain. 
		idx_gap = int(np.ceil(self.n[0] / float(slices + 1)))		# calculate gap between indices for given slices.
		indices = np.arange(0, self.n[0])							# array of indices.
		indices = indices[0:self.n[0]:idx_gap]						# slice array to include valid indices.
		indices = indices[1:]										# remove 0 idx.

		# Coordinates.
		x = self.coordinates[0][0,0,:]/(2*np.pi)
		y = self.coordinates[1][0,:,0]/(2*np.pi)
		z = self.coordinates[2][:,0,0]/(2*np.pi)

		# Vorticity.
		vorticity = self.flow_data[time]['vorticities'][direction]

		# Pressure.
		# pressure = self.flow_data[time]['p']

		# Velocities velocities.
		u = self.flow_data[time]['u']
		v = self.flow_data[time]['v']
		w = self.flow_data[time]['w']

		# Velocity magnitude in plane.
		if direction == 0:
			speed = np.sqrt(v**2 + w**2)
		elif direction == 1:
			speed = np.sqrt(u**2 + w**2)
		elif direction == 2:
			speed = np.sqrt(u**2 + v**2)
		else:
			raise ValueError("Direction must be either 0, 1 or 2")

		# Reduce data density for quiver plot. INCOMPLETE. Required to improve visibility of vector plot.
		# grid = self.n[0]
		# sample = 1
		# if grid > 32:
		# 	sample = 2
		# if grid > 64:
		# 	sample = 4
		# skip = slice(None,None,sample)

		for idx in indices:
			# Plot.
			if direction == 0:
				# Slice location.
				location = x[idx]
				# Max speed.
				speed_max = speed[:,:,idx].max()
				# Normalise speed.
				speed_n = speed[:,:,idx] / speed_max
				# Contour plot.
				cp = plt.contourf(y, z, vorticity[:,:,idx])
				# # Quiver plot.
				quiv = plt.quiver(y, z, v[:,:,idx], w[:,:,idx], color='black', headlength=4)
				# Stream plot. INCOMPLETE
				# lw = 3 * speed_n
				# sp = plt.streamplot(y, z, v[:,:,idx], w[:,:,idx], density=1, color='k', linewidth=lw)
				# Label axes.
				plt.xlabel('Y')
				plt.ylabel('Z')
				# Plot filename.
				filename = 'vorticity_x={}.png'.format(np.round(location, 4))
			elif direction == 1:
				location = y[idx]
				speed_max = speed[:,idx,:].max()
				speed_n = speed[:,idx,:] / speed_max
				cp = plt.contourf(x, z, vorticity[:,idx,:])
				quiv = plt.quiver(x, z, u[:,idx,:], w[:,idx,:], color='black', headlength=4)
				# lw = 3 * speed_n
				# sp = plt.streamplot(y, z, v[:,idx,:], w[:,idx,:], density=1, color='k', linewidth=lw)
				plt.xlabel('X')
				plt.ylabel('Z')
				filename = 'vorticity_y={}.png'.format(np.round(location, 4))
			elif direction == 2:
				location = z[idx]
				speed_max = speed[idx,:,:].max()
				speed_n = speed[idx,:,:] / speed_max
				cp = plt.contourf(y, x, vorticity[idx,:,:])
				quiv = plt.quiver(y, x, v[idx,:,:], u[idx,:,:], color='black', headlength=4)
				# lw = 3 * speed_n
				# sp = plt.streamplot(y, z, v[idx,:,:], w[idx,:,:], density=1, color='k', linewidth=lw)
				plt.xlabel('Y')
				plt.ylabel('X')
				filename = 'vorticity_z={}.png'.format(np.round(location, 4))
			else:
				raise ValueError("Direction must be either 0, 1 or 2")

			cb = plt.colorbar(cp)
			# plt.show()
			plt.savefig(os.path.join(folder, filename))
			plt.gcf().clear()

	def _plots(self):
		"""
		Function to plot kinetic energy and dissipation rate against time.
		"""
		# Create directory.
		folder = os.path.join(self.output_path, 'plots')
		if not os.path.exists(folder):
			os.makedirs(folder)

		# Create energy and dissipation lists.
		energies = [self.flow_data[time]['E_k'] for time in self.times]
		dissipations = [self.flow_data[time]['eps'] for time in self.times]

		# Plot kinetic energy against time.
		print("Ploting kinetic energy over time...")
		plt.plot(self.times, energies, 'k-')
		plt.ylabel("Non-dimensional kinetic energy")
		plt.xlabel("Non-dimensional time")
		plt.xlim(0.0, 20.0)
		# plt.show()
		plt.savefig(os.path.join(folder, 'kinetic_energy.png'))
		plt.gcf().clear()

		# Plot dissipation rate against time.
		print("Ploting dissipation rate over time...")
		plt.plot(self.times, dissipations, 'k-')
		plt.ylabel("Non-dimensional dissipation rate")
		plt.xlabel("Non-dimensional time")
		plt.xlim(0.0, 20.0)
		# plt.show()
		plt.savefig(os.path.join(folder, 'dissipation_rate.png'))
		plt.gcf().clear()

	def _create_coords(self):
		"""
		Function to create coordinate arrays in all directions.
		"""
		# Initialise NxNxN arrays for coordinate values.
		x0 = np.zeros((self.n[2], self.n[1], self.n[0]))
		x1 = np.zeros((self.n[2], self.n[1], self.n[0]))
		x2 = np.zeros((self.n[2], self.n[1], self.n[0]))

		# Store coordinate values.
		for i, p0 in enumerate(np.linspace(self.domain[2][0], self.domain[2][1], self.n[2])):
			for j, p1 in enumerate(np.linspace(self.domain[1][0], self.domain[1][1], self.n[1])):
				for k, p2 in enumerate(np.linspace(self.domain[0][0], self.domain[0][1], self.n[0])):
					x0[k, j, i] = p0
					x1[k, j, i] = p1
					x2[k, j, i] = p2

		self.coordinates = [x0, x1, x2]

	def _read_file(self, filename):
		"""
		Function to read a .h5 file. 
		params:  
			filename:	string of file filpath
		returns:
			f:			h5 file
			group:		group containing data
		"""
		f = h5py.File(filename, 'r')
		group = f["opensbliblock00"]
		return f, group

	def _read_dataset(self, group, dataset):
		"""
		Function to read data from h5 group.
		params:  
			group:   	h5 group containing data
			dataset: 	string of variable name
		returns: 
			read_data:	numpy array of data read from group
		"""
		# Extract dataset
		data = group['{}'.format(dataset)]
		size = data.shape

		# Trim dataset for valid data only.
		d_m = data.attrs['d_m']
		read_start = [abs(d+self.halo_points) for d in d_m]
		read_end = [s-abs(d+self.halo_points) for d, s in zip(d_m, size)]
		if len(read_end) == 2:
			read_data = data.value[read_start[0]:read_end[0],
								   read_start[1]:read_end[1]]
		elif len(read_end) == 3:
			read_data = data.value[read_start[0]:read_end[0],
								   read_start[1]:read_end[1],
								   read_start[2]:read_end[2]]
		else:
			raise NotImplementedError("Only 2 or 3 dimensional cases implemented.")

		return read_data

	def _extract_flow_variables(self, group):
		"""
		Function to extract variables from h5 group.
		params:
			group:		h5 group containing data
		returns: 
			rho: 		numpy array of density
			u, v, w:	numpy array of velocity components
			p: 			numpy array of pressure
		"""
		# Specific heat ratio.
		gamma = 1.4

		# Variable strings
		strings = ["rho_B0", 
				   "rhou0_B0", "rhou1_B0", "rhou2_B0",
				   "rhoE_B0"]

		# Read conservative variables from group.
		variables = [self._read_dataset(group, variable) for variable in strings]
		rho = variables[0]
		rhoE = variables[4]

		# Velocity components.
		u = variables[1] / rho
		v = variables[2] / rho
		w = variables[3] / rho

		# Pressure from ideal gas equation of state.
		# P = (gamma-1).rho.e where e = total energy - kinetic energy.
		# P: pressure, gamma: specific heat, rho: density, e: internal specific energy.
		p = (gamma - 1) * (rhoE - 0.5*(u**2+v**2+w**2)*rho)

		return rho, u, v, w, p

	def _derivative(self, flow_variable, direction):
		"""
		Function to calculate first order derivative of a given flow variable
		in a given direction for a uniform grid.
		params:
			flow_variable:		numpy array of flow variable
			direction:			int direction in which to calculate derivative
		returns:
			df_dx:				numpy array of derivative
		"""
		# Setup
		coordinate_array = self.coordinates[direction]
		df_dx = np.zeros(flow_variable.shape)

		# Number of points in given direction of domain.
		n0 = flow_variable.shape[2]
		n1 = flow_variable.shape[1]
		n2 = flow_variable.shape[0]

		# Delta H for periodic cube domain.
		delta_h = 2 * np.pi / self.n[0]

		# Calculate derivatives. 
		if direction == 2:		# Calculate z derivative
			# delta_h = coordinate_array[1,0,0] - coordinate_array[0,0,0]
			for i in range(self.halo_points, n0-self.halo_points):
				for j in range(self.halo_points, n1-self.halo_points):
					for k in range(self.halo_points, n2-self.halo_points):
						df_dx[k,j,i] = (flow_variable[k-2,j,i] - 8*flow_variable[k-1,j,i] + 
										8*flow_variable[k+1,j,i] - flow_variable[k+2,j,i]) / (12 * delta_h)
		elif direction == 1:	# Calculate y derivative
			# delta_h = coordinate_array[0,1,0] - coordinate_array[0,0,0]
			for i in range(self.halo_points, n0-self.halo_points):
				for j in range(self.halo_points, n1-self.halo_points):
					for k in range(self.halo_points, n2-self.halo_points):
						df_dx[k,j,i] = (flow_variable[k,j-2,i] - 8*flow_variable[k,j-1,i] + 
										8*flow_variable[k,j+1,i] - flow_variable[k,j+2,i]) / (12 * delta_h)
		elif direction == 0:	# Calculate x derivative
			# delta_h = coordinate_array[0,0,1] - coordinate_array[0,0,0]
			for i in range(self.halo_points, n0-self.halo_points):
				for j in range(self.halo_points, n1-self.halo_points):
					for k in range(self.halo_points, n2-self.halo_points):
						df_dx[k,j,i] = (flow_variable[k,j,i-2] - 8*flow_variable[k,j,i-1] + 
										8*flow_variable[k,j,i+1] - flow_variable[k,j,i+2]) / (12 * delta_h)
		else:
			raise ValueError("Direction must be either 0, 1 or 2")

		return df_dx


if __name__ == '__main__':
	grid_compare(GRIDS, TIMESTEPS, DOMAIN, '.')
