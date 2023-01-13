import numpy as np
import matplotlib.pyplot as plt
from ElectronBoltzmannArc.constants import *

Mbarn = 1e-22

def total_xsec(Ee):
	return 1e1*Mbarn*np.ones_like(Ee)

def transfer_matrix(vmag,E_grid,pvol_grid):
	Ee,pvol = E_grid.flatten(),pvol_grid.flatten()

	E2,E1 = np.meshgrid(Ee,Ee,indexing='ij')
	A = total_xsec(E1)/(4.0*np.pi/3.0*vmag**3)*np.heaviside(E1-E2,0.5)

	return A