import numpy as np
from scipy.linalg import lu_solve,lu_factor
import matplotlib.pyplot as plt
import ElectronBoltzmannArc.cross_sections as xs
from ElectronBoltzmannArc.constants import *
from ElectronBoltzmannArc.utils import *

def test_source(vpara_grid,vperp_grid,t,t0=8e-9,FWHM=10e-9,T=10.0):
	E_grid = energy_from_vps(vpara_grid,vperp_grid)
	R      = Gaussian_from_FWHM(t,t0,FWHM)
	return Maxwellian(E_grid,T)*R

class EBA_solver:
	def __init__(self,E_applied,n_neutral,vmax,Nv,neutral_species='He'):
		# Physical parameters
		self.E_applied = E_applied # V/m
		self.n_neutral = n_neutral # 1/m^3

		# Program parameters
		self.vmax = vmax # m/s
		self.Nv   = Nv

		self.vpara  = np.linspace(-self.vmax,self.vmax,2*self.Nv)
		self.dv     = self.vpara[1]-self.vpara[0]
		self.vperp  = self.dv*(np.arange(self.Nv)+0.5)

		self.vpara_grid,self.vperp_grid = np.meshgrid(self.vpara,self.vperp)
		self.vmag_grid = np.sqrt(self.vpara_grid**2+self.vperp_grid**2)
		self.vmag = self.vmag_grid.flatten()
		self.E_grid = energy_from_vps(self.vpara_grid,self.vperp_grid)
		self.vol    = np.pi*self.dv*((self.vperp_grid+0.5*self.dv)**2-(self.vperp_grid-0.5*self.dv)**2) 
		self.Nmatrix = self.vmag.size

		self.fe = np.zeros_like(self.vpara_grid)
		self.t  = 0.0

		if(neutral_species=='He'):
			xs.Lotz_a,xs.Lotz_b,xs.Lotz_c,xs.Lotz_q,xs.Lotz_P = 4.0,0.75,0.46,[2.],[24.6]

	def initialise_cross_sections(self):

		self.total_rate      = self.n_neutral*self.vmag*xs.total_xsec(self.E_grid).flatten()
		self.transfer_matrix = self.n_neutral*self.vmag*xs.transfer_matrix(self.vmag,self.E_grid,self.vol)*self.vol.flatten()[None,:]

		if(self.dt*np.amax(self.total_rate) > 1.0):
			self.dt = 1/np.amax(self.total_rate)

		self.M = np.eye(self.Nmatrix)-self.dt*(self.transfer_matrix+np.diag(-self.total_rate))
		self.lu, self.piv = lu_factor(self.M)

	def evolve(self,tmax,external_source,safety_factor=0.1):

		self.dt = safety_factor*self.dv/(qe*self.E_applied/me)
		self.initialise_cross_sections()

		Nt = int(np.ceil(tmax/self.dt))

		for it in range(Nt):
			fe_old = np.copy(self.fe)

			# Velocity advection by E, upwind
			self.fe[:,:-1] = fe_old[:,:-1]+self.dt/self.dv*qe*self.E_applied/me*(fe_old[:,1:]-fe_old[:,:-1])
			self.fe[:,-1]  = fe_old[:,-1]+self.dt/self.dv*qe*self.E_applied/me*(-fe_old[:,-1])

			# Collision operators + external source, implicit
			S = self.dt*self.n_neutral*0.5*(external_source(self.vpara_grid,self.vperp_grid,self.t)+external_source(self.vpara_grid,self.vperp_grid,self.t+self.dt))
			fe_intermediate = self.fe+S
			sol = lu_solve((self.lu, self.piv),fe_intermediate.flatten())
			self.fe = sol.reshape(self.Nv,2*self.Nv)

			self.t = self.t + self.dt

		self.ne,self.Ve,self.Te = self.df_moments()

	def df_moments(self):
		ne = np.trapz(2*np.pi*self.vperp*np.trapz(self.fe,self.vpara,axis=1),self.vperp)
		Ve = np.trapz(2*np.pi*self.vperp*np.trapz(self.vpara_grid*self.fe,self.vpara,axis=1),self.vperp)/ne
		x = me/qe*((self.vpara_grid-Ve)**2+self.vperp_grid**2)/3.0
		Te = np.trapz(2*np.pi*self.vperp*np.trapz(x*self.fe,self.vpara,axis=1),self.vperp)/ne
		return ne,Ve,Te