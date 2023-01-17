import numpy as np
from scipy.linalg import lu_solve,lu_factor,block_diag
from ElectronBoltzmannArc.constants import *
from ElectronBoltzmannArc.cross_sections import *
from ElectronBoltzmannArc.utils import *
import matplotlib.pyplot as plt
from sys import exit

def test_source(v,t,t0=8e-9,FWHM=10e-9,T=10.0):
	Ke = 0.5*me*v**2/qe
	R      = Gaussian_from_FWHM(t,t0,FWHM)
	return Maxwellian(Ke,T)*R

class EBA_solver:
	def __init__(self,E_applied,n_neutral,vmax,Nv,mu,mu_w):
		# Physical parameters
		self.E_applied = np.abs(E_applied) # V/m
		self.n_neutral = n_neutral # 1/m^3

		# Program parameters
		self.vmax = vmax # m/s
		self.Nv   = Nv

		self.dv	 = self.vmax/(self.Nv-1)
		self.v_arr  = self.dv*(np.arange(self.Nv)+0.5)

		self.mu_arr = mu
		self.Nmu    = mu.size
		self.mu_w   = mu_w

		self.v_grid,self.mu_grid = np.meshgrid(self.v_arr,self.mu_arr)
		self.v_flat  = self.v_grid.flatten()
		self.mu_flat = self.mu_grid.flatten()
		self.Nmatrix = self.v_flat.size
		self.v_vol   = 2*np.pi*self.v_grid**2*self.dv

		self.fe = np.zeros_like(self.v_grid)
		self.t  = 0.0

		self.intialise_matrices()

	def intialise_matrices(self,safety_factor=0.1):
		self.dt = safety_factor*self.dv/(qe*self.E_applied/me)

		self.initialise_transport()
		self.initialise_cross_sections()

		self.M = np.eye(self.Nmatrix)-self.dt*(self.L+self.C)
		self.lu, self.piv = lu_factor(self.M)

	def initialise_cross_sections(self):
		# self.total_rate      = self.n_neutral*self.v_flat*total_xsec(self.v_flat).flatten()
		# self.transfer_matrix = self.n_neutral*self.v_flat*transfer_matrix(self.v_flat,self.vol)*self.vol.flatten()[None,:]

		# if(self.dt*np.amax(self.total_rate) > 1.0):
		# 	self.dt = 1/np.amax(self.total_rate)

		# self.C = self.transfer_matrix+np.diag(-self.total_rate)
		self.C = np.zeros_like(self.L)
		
	def initialise_transport(self):
		prefactor = qe*self.E_applied/me

		self.L_v = np.zeros((self.Nv,self.Nv))
		self.L_v = (np.diag(np.ones(self.Nv-1),-1)-np.diag(np.ones(self.Nv)))/self.dv
		blocks = ((self.L_v,)*self.Nmu)
		self.L_v = block_diag(*blocks)

		ones_block = np.ones((self.Nv,self.Nv))
		blocks = [-ones_block for i in range(self.Nmu-1)]
		blocks.append(ones_block)
		self.L_mu = block_diag(*blocks)

		offset = np.empty((0,self.Nv),int)
		blocks = [ones_block for i in range(self.Nmu-1)]
		self.L_mu += block_diag(offset.T,*blocks,offset)

		self.L = self.L_v/self.mu_flat#+self.L_mu/self.v_flat
		self.L = prefactor*self.L

	def evolve(self,tmax,external_source):

		self.dt = 1e-12
		Nt = int(np.ceil(tmax/self.dt))

		for it in range(Nt):
			S = self.dt*self.n_neutral*0.5*(external_source(self.v_grid,self.t)+external_source(self.v_grid,self.t+self.dt))
			fe_intermediate = self.fe+S
			sol = lu_solve((self.lu, self.piv),fe_intermediate.flatten())
			self.fe = sol.reshape(self.Nmu,self.Nv)
			plt.imshow(self.fe)
			plt.show()
			exit()

			self.t = self.t + self.dt

		self.ne,self.Ve,self.Te = self.df_moments()

	def df_moments(self):
		ne = np.sum(self.mu_w*np.sum(self.v_vol*self.fe,axis=1))
		Ve = np.sum(self.mu_arr*self.mu_w*np.sum(self.v_vol*self.v_grid*self.fe,axis=1))/ne
		x = me/qe*((self.v_grid*self.mu_grid-Ve)**2+self.v_grid**2*(1-self.mu_grid**2))/3.0
		Te = np.sum(self.mu_w*np.sum(self.v_vol*x*self.fe,axis=1))/ne
		return ne,Ve,Te