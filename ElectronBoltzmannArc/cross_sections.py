import numpy as np
import matplotlib.pyplot as plt
from ElectronBoltzmannArc.constants import *

# Lotz constants
Lotz_a,Lotz_b,Lotz_c,Lotz_q,Lotz_P = None,None,None,None,None
Lotz_a_unit = 1e-20 # m^2 (eV)^2

def elastic_xsec(Ee):
	sigma = np.zeros_like(Ee)
	return sigma

def Lotz_collisional_ionisation_xsec(Ee):
	summed_term = np.zeros_like(Ee)
	for i,P in enumerate(Lotz_P):
		mask = Ee > P
		summed_term[mask] = Lotz_q[i]*np.log(Ee[mask]/P)/(Ee[mask]*P)
	U = Ee/Lotz_P[0]
	sigma = Lotz_a_unit*Lotz_a*(1-Lotz_b*np.exp(-Lotz_c*(U-1)))*summed_term
	return sigma

def collisional_excitation_xsec(Ee):
	sigma = np.zeros_like(Ee)
	return sigma

def total_xsec(Ee):
	sigma_tot = elastic_xsec(Ee)
	sigma_tot += Lotz_collisional_ionisation_xsec(Ee)
	sigma_tot += collisional_excitation_xsec(Ee)
	return sigma_tot

def elastic_transfer(vmag,E_grid,pvol_grid):
	Ee,pvol = E_grid.flatten(),pvol_grid.flatten()
	E2,E1 = np.meshgrid(Ee,Ee,indexing='ij')

	total_T = np.zeros_like(E1)

	return total_T 

def collisional_ionisation_transfer(vmag,E_grid,pvol_grid):
	Ee,pvol = E_grid.flatten(),pvol_grid.flatten()
	E2,E1 = np.meshgrid(Ee,Ee,indexing='ij')

	total_T = np.zeros_like(E1)

	for i,P in enumerate(Lotz_P):
		# Ionising electron
		A = np.heaviside(E1-(E2+P),0.5)/(4*np.pi/3.0*vmag**3)
		# Normalise
		norm = np.sum(A*pvol[None,:])
		A = A/norm

		U = E1/Lotz_P[0]
		x = E1/P
		x[x < 1.0] = 1.0
		total_T += 2*Lotz_a_unit*Lotz_a*(1-Lotz_b*np.exp(-Lotz_c*(U-1)))*Lotz_q[i]*np.log(x)/(E1*P)*A

	return total_T 

def collisional_excitation_transfer(vmag,E_grid,pvol_grid):
	Ee,pvol = E_grid.flatten(),pvol_grid.flatten()
	E2,E1 = np.meshgrid(Ee,Ee,indexing='ij')

	total_T = np.zeros_like(E1)

	return total_T 

def transfer_matrix(vmag,E_grid,pvol_grid):

	t_matrix = elastic_transfer(vmag,E_grid,pvol_grid)
	t_matrix += collisional_ionisation_transfer(vmag,E_grid,pvol_grid)
	t_matrix += collisional_excitation_transfer(vmag,E_grid,pvol_grid)

	return t_matrix

