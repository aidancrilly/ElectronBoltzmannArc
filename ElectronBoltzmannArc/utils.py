import numpy as np
from ElectronBoltzmannArc.constants import *

def energy_from_vps(v1,v2):
	return 0.5*me*(v1**2+v2**2)/qe

def Gaussian_from_FWHM(t,t0,FWHM):
	sigma = FWHM/2.355
	return np.exp(-0.5*((t-t0)/sigma)**2)/sqrt_2_pi/sigma

def Maxwellian(E,T):
	return (me/(qe*T))**1.5/sqrt_2_pi**3*np.exp(-E/T)