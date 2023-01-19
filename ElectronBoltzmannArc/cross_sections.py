import numpy as np
import matplotlib.pyplot as plt
from ElectronBoltzmannArc.constants import *
from sys import exit

# Lotz constants
Lotz_a,Lotz_b,Lotz_c,Lotz_q,Lotz_P = None,None,None,None,None
Lotz_a_unit = 1e-20 # m^2 (eV)^2

def elastic_xsec(Ee):
	sigma = Lotz_collisional_ionisation_xsec(Ee)
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

def total_xsec(ve):
	Ee = 0.5*me*ve**2/qe
	sigma_tot = elastic_xsec(Ee)
	# sigma_tot += Lotz_collisional_ionisation_xsec(Ee)
	sigma_tot += collisional_excitation_xsec(Ee)
	return sigma_tot

def elastic_transfer(E1,E2):
	t_matrix = np.zeros_like(E1)

	t_matrix[E1 == E2] = elastic_xsec(E1[E1 == E2])

	return t_matrix 

def collisional_ionisation_transfer(E1,E2):
	t_matrix = np.zeros_like(E1)

	for i,P in enumerate(Lotz_P):
		sub_t = np.zeros_like(t_matrix)
		# idx = np.argmin(np.abs(E1-(E2+P)))
		# sub_t[idx,:] += 1.0

		U = E1/Lotz_P[0]
		x = E1/P
		x[x < 1.0] = 1.0
		sigma = Lotz_a_unit*Lotz_a*(1-Lotz_b*np.exp(-Lotz_c*(U-1)))*Lotz_q[i]*np.log(x)/(E1*P)
		t_matrix += sigma*sub_t

	return t_matrix 

def collisional_excitation_transfer(E1,E2):

	t_matrix = np.zeros_like(E1)

	return t_matrix 

def transfer_matrix(ve,Vv,mu,mu_w):

	Ee = 0.5*me*ve**2/qe
	E1,E2 = np.meshgrid(Ee,Ee)

	t_matrix = np.zeros_like(E1)

	t_matrix += elastic_transfer(E1,E2)
	t_matrix += collisional_ionisation_transfer(E1,E2)
	# t_matrix += collisional_excitation_transfer(E1,E2)
	# t_matrix = t_matrix*Vv[None,:]

	Nmu = mu.size
	t_rows  = np.hstack([0.5*t_matrix*w for w in mu_w])
	total_T = np.vstack((t_rows,)*Nmu)

	return total_T

