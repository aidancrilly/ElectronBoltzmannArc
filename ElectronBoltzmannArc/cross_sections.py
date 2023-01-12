import numpy as np

Mbarn = 1e-22

def total_xsec(Ee):
	return 1e1*Mbarn*np.ones_like(Ee)

def transfer_matrix(E_grid):
	Ee = E_grid.flatten()
	E1,E2 = np.meshgrid(Ee,Ee)
	A = np.zeros_like(E1)
	A[E1 > E2] = 1.0/np.sum(1.*(E1 > E2))

	xsec_t = total_xsec(Ee)
	return xsec_t[None,:]*A