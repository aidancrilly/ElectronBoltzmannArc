import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ElectronBoltzmannArc.core import EBA_solver,test_source

E_applied,n_neutral,vmax,Nv = 1e-2,1e22,1e7,50
Solver = EBA_solver(E_applied,n_neutral,vmax,Nv)
for i in range(10):
	tmax = 30e-9
	Solver.evolve(tmax,test_source)

	plt.plot(Solver.t,Solver.ne,'bo')

plt.show()