import numpy as np
import matplotlib.pyplot as plt
# from ElectronBoltzmannArc.constants import *
# from ElectronBoltzmannArc.cross_sections import *
# from ElectronBoltzmannArc.utils import *
from constants import *
from cross_sections import *
from utils import *

# Physical parameters
E_applied = 1e2 # V/m
n_neutral = 1e22 # 1/m^3

def external_source(vpara_grid,vperp_grid,t,t0=8e-9,FWHM=10e-9,T=10.0):
	E_grid = energy_from_vps(vpara_grid,vperp_grid)
	R      = Gaussian_from_FWHM(t,t0,FWHM)
	return Maxwellian(E_grid,T)*R

# Program parameters
vmax = 1e7 # m/s
Nv = 50
dt = 1e-9
tmax = 100e-9
Nt = int(np.ceil(tmax/dt))

vpara = np.linspace(-vmax,vmax,2*Nv)
vperp = np.linspace(0.0,vmax,Nv)

vpara_grid,vperp_grid = np.meshgrid(vpara,vperp)

fe = np.zeros_like(vpara_grid)

t = 0.0
for it in range(Nt):
	fe_old = np.copy(fe)
	S = n_neutral*external_source(vpara_grid,vperp_grid,t)
	fe = fe_old+dt*S
	t = t + dt

plt.imshow(fe)
plt.colorbar()

plt.show()