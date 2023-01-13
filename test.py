import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ElectronBoltzmannArc.core import EBA_solver,test_source

E_applied,n_neutral,vmax,Nv = 1e3,1e22,1e7,50
Solver = EBA_solver(E_applied,n_neutral,vmax,Nv)

ts,nes,Ves,Tes = [],[],[],[]
tmax = 200e-9
tstep = 10e-9
Nt = int(np.ceil(tmax/tstep))

for i in range(Nt):
	Solver.evolve(tstep,test_source)

	ts.append(Solver.t*1e9)
	nes.append(Solver.ne)
	Ves.append(Solver.Ve*1e-6)
	Tes.append(Solver.Te)

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.imshow(Solver.fe,origin='lower',extent=[Solver.vpara[0]*1e-6,Solver.vpara[-1]*1e-6,Solver.vperp[0]*1e-6,Solver.vperp[-1]*1e-6])
	ax2,ax3,ax4 = fig.add_subplot(234),fig.add_subplot(235),fig.add_subplot(236)
	ax2.plot(ts,nes,'bo')
	ax3.plot(ts,Ves,'bo')
	ax4.plot(ts,Tes,'bo')
	ax2.set_xlim(0.0,tmax*1e9)
	ax3.set_xlim(0.0,tmax*1e9)
	ax4.set_xlim(0.0,tmax*1e9)
	ax2.set_ylabel(r"$n_e$ (1/m$^3$)")
	ax3.set_ylabel(r"$V_e$ (mm/ns)")
	ax4.set_ylabel(r"$T_e$ (eV)")
	ax3.set_xlabel("Time (ns)")
	fig.tight_layout()
	fig.savefig(f'C:\\Users\\Aidan Crilly\\Documents\\GitHub\\ElectronBoltzmannArc\\images\\tidx_{i}.png')

