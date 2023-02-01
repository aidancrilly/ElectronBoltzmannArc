import numpy as np
import matplotlib.pyplot as plt
# from ElectronBoltzmannArc.constants import *
from constants import *
from sys import exit

class Material:
	def __init__(self):
		
		# Lotz constants
		self.Lotz_a_unit = 1e-18 # m^2 (eV)^2
		self.Lotz_a,self.Lotz_b,self.Lotz_c,self.Lotz_q,self.Lotz_P = None,None,None,None,None

		# Lagushenko and Maya
		self.LM_xs_unit = 1e-20 # m^2
		self.LM_a_el,self.LM_b_el,self.LM_c1_el,self.LM_c2_el = None,None,None,None
		self.LM_a_ex,self.LM_b0_ex,self.LM_b1_ex,self.LM_b2_ex,self.LM_e0_ex,self.LM_e1_ex = None,None,None,None,None,None

	def set_material_type(self,material_name):
		if(material_name=='He'):
			self.Lotz_a,self.Lotz_b,self.Lotz_c,self.Lotz_q,self.Lotz_P = [4.0],[0.75],[0.46],[2.],[24.6]
			self.LM_a_el,self.LM_b_el,self.LM_c1_el,self.LM_c2_el = 6.16,15.0,0.0,1.5
			self.LM_a_ex,self.LM_b0_ex,self.LM_b1_ex,self.LM_b2_ex,self.LM_e0_ex,self.LM_e1_ex = 0.282,0.13,0.08,0.0,10.0,20.0
		if(material_name=='Ne'):
			self.Lotz_a,self.Lotz_b,self.Lotz_c,self.Lotz_q,self.Lotz_P = [2.6,4.0],[0.92,0.7],[0.19,0.5],[6.,2.],[21.6,48.5]
			self.LM_a_el,self.LM_b_el,self.LM_c1_el,self.LM_c2_el = 1.51,40.0,0.33333,1.4
			self.LM_a_ex,self.LM_b0_ex,self.LM_b1_ex,self.LM_b2_ex,self.LM_e0_ex,self.LM_e1_ex = 0.282,0.10,0.0,0.009,12.0,16.5

neutral_species = Material()

def LM_elastic_xsec(Ee):
	return neutral_species.LM_xs_unit*neutral_species.LM_a_el*(Ee)**neutral_species.LM_c1_el/(1.0+(Ee/neutral_species.LM_b_el)**neutral_species.LM_c2_el)

def LM_excitation_xsec(Ee):
	x = Ee/neutral_species.LM_e0_ex
	log_term = 1.0+(neutral_species.LM_b0_ex+neutral_species.LM_b1_ex*(Ee-neutral_species.LM_e1_ex)+neutral_species.LM_b2_ex*(Ee-neutral_species.LM_e1_ex)**2)*x
	return 0.0*neutral_species.LM_xs_unit*neutral_species.LM_a_ex*np.log(log_term)/x*np.heaviside(Ee-neutral_species.LM_e1_ex,0.0)

def Lotz_collisional_ionisation_xsec(Ee):
	sigma = np.zeros_like(Ee)
	for i,P in enumerate(neutral_species.Lotz_P):
		summed_term = np.zeros_like(Ee)
		mask = Ee > P
		summed_term[mask] = neutral_species.Lotz_q[i]*np.log(Ee[mask]/P)/(Ee[mask]*P)
		U = Ee/P
		sigma += neutral_species.Lotz_a_unit*neutral_species.Lotz_a[i]*(1-neutral_species.Lotz_b[i]*np.exp(-neutral_species.Lotz_c[i]*(U-1)))*summed_term
	return sigma

def elastic_xsec(Ee):
	sigma = LM_elastic_xsec(Ee)
	return sigma

def collisional_ionisation_xsec(Ee):
	sigma = Lotz_collisional_ionisation_xsec(Ee)
	return sigma

def collisional_excitation_xsec(Ee):
	sigma = LM_excitation_xsec(Ee)
	return sigma

def total_xsec(ve):
	Ee = 0.5*me*ve**2/qe
	sigma_tot = elastic_xsec(Ee)
	sigma_tot += collisional_ionisation_xsec(Ee)
	sigma_tot += collisional_excitation_xsec(Ee)
	return sigma_tot

def elastic_transfer(E1,E2):
	t_matrix = np.zeros_like(E1)

	t_matrix[E1 == E2] = elastic_xsec(E1[E1 == E2])

	return t_matrix 

def Clark_ionisation_differential(E1,E2,I):
	# https://www.sciencedirect.com/science/article/pii/S0022407305001780
	a = 14.4
	epsilon = E1-E2-I
	epsilon[E1 > 2*E2+I] = E2[E1 > 2*E2+I]
	dE1 = E1-I
	omega = np.heaviside(E1-(E2+I),0.0)/dE1/(E1**2+a*I**2)*(2*(a+1)*I**2+(160*(E1+I)*(epsilon-0.5*dE1)**4)/dE1**3)
	return omega

def collisional_ionisation_transfer(E1,E2,ve,Vv2):
	t_matrix = np.zeros_like(E1)
	dv = ve[1]-ve[0]

	for i,P in enumerate(neutral_species.Lotz_P):
		U = E1/P
		x = E1/P
		x[x < 1.0] = 1.0
		sigma = neutral_species.Lotz_a_unit*neutral_species.Lotz_a[i]*(1-neutral_species.Lotz_b[i]*np.exp(-neutral_species.Lotz_c[i]*(U-1)))*neutral_species.Lotz_q[i]*np.log(x)/(E1*P)
		sub_t = Clark_ionisation_differential(E1,E2,P)#*me*ve[:,None]/qe
		# Normalise
		norm = np.sum(sub_t*Vv2,axis=0)#np.sum(sub_t*dv,axis=0)
		norm[norm == 0] = 1.0
		sub_t = 2*sub_t/norm
		t_matrix += sigma*sub_t*Vv2

	return t_matrix

def collisional_excitation_transfer(E1,E2,Vv1,Vv2):
	t_matrix = np.zeros_like(E1)

	I = neutral_species.LM_e1_ex

	mask = np.argmin(np.abs(E1-(E2+I)),axis=0)

	for i,m in enumerate(mask):
		t_matrix[m,i] = LM_excitation_xsec(E1[m,i])

	return t_matrix 

def transfer_matrix(ve,V_v,mu,mu_w):

	Ee = 0.5*me*ve**2/qe
	E1,E2 = np.meshgrid(Ee,Ee)
	Vv1,Vv2 = np.meshgrid(V_v,V_v)

	t_matrix = np.zeros_like(E1)

	t_matrix += elastic_transfer(E1,E2)
	t_matrix += collisional_ionisation_transfer(E1,E2,ve,Vv2)
	t_matrix += collisional_excitation_transfer(E1,E2,Vv1,Vv2)

	t_matrix *= Vv1/Vv2

	Nmu = mu.size
	t_rows  = np.hstack([0.5*t_matrix*w for w in mu_w])
	total_T = np.vstack((t_rows,)*Nmu)

	return total_T

