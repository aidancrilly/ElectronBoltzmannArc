import numpy as np
from scipy.linalg import lu_solve,lu_factor,block_diag
import ElectronBoltzmannArc.cross_sections as xs
from ElectronBoltzmannArc.constants import *
from ElectronBoltzmannArc.utils import *
import matplotlib.pyplot as plt
from sys import exit

def test_source(v,t,t0=8e-9,FWHM=10e-9,T=10.0):
	Ke = 0.5*me*v**2/qe
	R      = Gaussian_from_FWHM(t,t0,FWHM)
	return 1e-3*Maxwellian(Ke,T)*R

class EBA_solver:
	def __init__(self,E_applied,n_neutral,vmax,Nv,mu,mu_w,neutral_species):
		# Physical parameters
		self.E_applied = np.abs(E_applied) # V/m
		self.n_neutral = n_neutral # 1/m^3

		# Program parameters
		self.vmax = vmax # m/s
		self.Nv   = Nv

		self.dv	    = self.vmax/(self.Nv-1)
		self.v_arr  = self.dv*(np.arange(self.Nv)+0.5)
		self.E_arr  = 0.5*me*(self.v_arr)**2/qe
		self.A_v    = 4*np.pi*self.dv**2*(np.arange(self.Nv+1))**2
		self.V_v    = 4*np.pi/3.0*((self.v_arr+0.5*self.dv)**3-(self.v_arr-0.5*self.dv)**3)
		self.v_edges= self.dv*(np.arange(self.Nv+1))

		self.mu_arr = mu
		self.Nmu    = mu.size
		self.mu_w   = mu_w
		self.alpha  = np.zeros(self.Nmu+1)
		self.mu_edges = np.zeros(self.Nmu+1)
		self.mu_edges[0] = -1.0
		for i in range(1,self.Nmu+1):
			self.alpha[i] = self.alpha[i-1]-self.mu_arr[i-1]*self.mu_w[i-1]
			self.mu_edges[i] = self.mu_edges[i-1]+self.mu_w[i-1]

		self.v_grid,self.mu_grid = np.meshgrid(self.v_arr,self.mu_arr)
		self.v_flat  = self.v_grid.flatten()
		self.mu_flat = self.mu_grid.flatten()
		self.Nmatrix = self.v_flat.size

		self.fe = np.zeros_like(self.v_grid)
		self.t  = 0.0

		xs.neutral_species.set_material_type(neutral_species)

		self.intialise_matrices()

	def intialise_matrices(self):
		self.initialise_transport()
		self.initialise_cross_sections()

		self.D = np.eye(self.Nmatrix)
		self.M = self.L+self.C
		# self.lu, self.piv = lu_factor(self.M)

	def initialise_transport(self):
		prefactor = qe*self.E_applied/me

		A_imh = self.A_v[:-1]
		A_iph = self.A_v[1:]
		V_i = self.V_v

		D_matrix = np.zeros((self.Nmu,self.Nv,self.Nv))
		O_matrix = np.zeros((self.Nmu,self.Nv,self.Nv))
		R_matrix = np.zeros((self.Nmu//2,self.Nv,self.Nv))
		T_matrix = np.zeros((self.Nmatrix,self.Nmatrix))
		for n in range(self.Nmu):
			mun = self.mu_arr[n]
			abs_mun = np.abs(mun)
			wn = self.mu_w[n]
			a_nph = self.alpha[n+1]
			a_nmh = self.alpha[n]

			stacked_deriv = np.vstack((-(A_imh+A_iph)/V_i,)*self.Nv).T

			# Diagonal matrices
			if(n < self.Nmu//2): # mu < 0
				D_matrix[n,:,:] = np.diag(2*A_imh*abs_mun/V_i)
			else: # mu > 0
				D_matrix[n,:,:] = np.diag(2*A_iph*abs_mun/V_i)

			# Off-diagonal matrices
			checkerboard_u = np.zeros((self.Nv,self.Nv))
			checkerboard_l = np.zeros((self.Nv,self.Nv))
			for k in range(-self.Nv,self.Nv+1):
				checkerboard_u += (-1)**(k+1)*np.eye(self.Nv,k=k)
				checkerboard_l += (-1)**(k+1)*np.eye(self.Nv,k=k)
			O_base = 2*abs_mun*stacked_deriv
			if(n < self.Nmu//2): # mu < 0
				O_matrix[n,:,:] = np.triu(O_base,k=1)*checkerboard_u
			else: # mu > 0
				O_matrix[n,:,:] = np.tril(O_base,k=-1)*checkerboard_l

			m = self.Nmu-n-1
			if(m < self.Nmu//2):
				R_matrix[m,:,:] = -2*mun*stacked_deriv*checkerboard_u

			# Diagonal
			m = n
			ridx = n*self.Nv
			kron_n = 1*int(n == 0)
			cidx = m*self.Nv
			T_matrix[ridx:ridx+self.Nv,cidx:cidx+self.Nv] += np.diag(2*a_nph*(A_iph-A_imh)/(wn*V_i)-kron_n*(a_nph+a_nmh)*(A_iph-A_imh)/(wn*V_i))
			# Off diagonal
			for m in range(0,n):
				cidx = m*self.Nv
				kron_m = 1*int(m == 0)
				if(n > 0):
					T_matrix[ridx:ridx+self.Nv,cidx:cidx+self.Nv] += -(2-kron_m)*(-1)**(n-m+1)*(a_nph+a_nmh)/wn*np.diag((A_iph-A_imh)/V_i)

		# Velocity derivative
		blocks = D_matrix+O_matrix
		self.L = block_diag(*blocks)
		for m in range(self.Nmu//2):
			n = self.Nmu-m-1
			ridx = n*self.Nv
			cidx = m*self.Nv
			self.L[ridx:ridx+self.Nv,cidx:cidx+self.Nv] = R_matrix[m,:,:]

		# Angular derivative
		self.L += T_matrix

		self.L *= prefactor

	def initialise_cross_sections(self):
		self.total_rate      = self.n_neutral*self.v_flat*xs.total_xsec(self.v_flat)
		self.transfer_matrix = self.n_neutral*self.v_flat*xs.transfer_matrix(self.v_arr,self.V_v,self.mu_arr,self.mu_w)

		# if(self.dt*np.amax(self.total_rate) > 1.0):
		# 	self.dt = 1/np.amax(self.total_rate)

		self.C = np.diag(self.total_rate)-self.transfer_matrix

	def evolve(self,tmax,external_source,dt):

		self.dt = dt #safety_factor*self.dv*np.amin(np.abs(self.mu_arr))/(qe*self.E_applied/me)
		Nt = int(np.ceil(tmax/self.dt))

		self.A = self.M+self.D/self.dt

		for it in range(Nt):
			S = self.n_neutral*0.5*(external_source(self.v_grid,self.t)+external_source(self.v_grid,self.t+self.dt))
			b = (self.fe/self.dt+S).flatten()
			#sol = lu_solve((self.lu, self.piv),b.flatten())
			sol = np.linalg.solve(self.A,b)
			self.fe = sol.reshape(self.Nmu,self.Nv)

			self.t = self.t + self.dt

		self.ne,self.Ve,self.Te = self.df_moments()

	def df_moments(self):
		ne = np.sum(self.mu_w*np.sum(self.V_v*self.fe,axis=1))
		Ve = np.sum(self.mu_arr*self.mu_w*np.sum(self.V_v*self.v_grid*self.fe,axis=1))/ne
		x = me/qe*((self.v_grid*self.mu_grid-Ve)**2+self.v_grid**2*(1-self.mu_grid**2))/3.0
		Te = np.sum(self.mu_w*np.sum(self.V_v*x*self.fe,axis=1))/ne
		return ne,Ve,Te