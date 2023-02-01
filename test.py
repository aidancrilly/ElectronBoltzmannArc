import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sys import exit
from ElectronBoltzmannArc.core import EBA_solver,test_source
import ElectronBoltzmannArc.cross_sections as xs
from ElectronBoltzmannArc.constants import *

Nmu = 16
mu,mu_w = np.polynomial.legendre.leggauss(Nmu)
Emax = 200.0 # eV
E_applied,n_neutral,vmax,Nv = 0e0,1e22,np.sqrt(2*Emax*qe/me),64
neutral_species = 'Ne'
Solver = EBA_solver(E_applied,n_neutral,vmax,Nv,mu,mu_w,neutral_species)

v_mesh,th_mesh = np.meshgrid(Solver.v_edges,np.arccos(Solver.mu_edges))
ts,nes,Ves,Tes = [],[],[],[]
tmax = 200e-9
tstep = 10e-9
Nt = int(np.ceil(tmax/tstep))

for i in range(Nt):
    print(i)
    Solver.evolve(tstep,test_source,1e-10)

    ts.append(Solver.t*1e9)
    nes.append(Solver.ne)
    Ves.append(Solver.Ve*1e-6)
    Tes.append(Solver.Te)

    fig = plt.figure(dpi=200,figsize=(9.5,5))
    ax1 = fig.add_subplot(231)
    f0 = np.sum(Solver.fe*Solver.mu_w[:,None],axis=0)
    f1 = np.sum(Solver.fe*Solver.mu_w[:,None]*Solver.mu_arr[:,None],axis=0) 
    ax1.plot(Solver.E_arr,f0/np.amax(f0),label='f0')
    ax1.plot(Solver.E_arr,f1/np.amax(f0),label='f1')
    ax1.legend()
    ax1.set_xlim(0.0,Emax)
    ax2 = fig.add_subplot(233,polar=True)
    ax2.grid(False)
    ax2.pcolormesh(th_mesh,v_mesh/1e6,Solver.fe)
    im1 = ax2.pcolormesh(2*np.pi-th_mesh,v_mesh/1e6,Solver.fe)
    ax2.tick_params(labelsize=0)
    fig.colorbar(im1,ax=ax2)
    ax2.set_title(r"$f_e$")
    # ax2.text(-0.6, 0.85, f't = {ts[i]:.1f} ns\nApplied E = {E_applied:.1f} V/m\n$n_{{neutral}}$ = {n_neutral:.1e} 1/m$^3$', transform=ax1.transAxes)
    ax3,ax4,ax5 = fig.add_subplot(234),fig.add_subplot(235),fig.add_subplot(236)
    ax3.plot(ts,nes,'bo')
    ax4.plot(ts,Ves,'bo')
    ax5.plot(ts,Tes,'bo')
    ax3.set_xlim(0.0,tmax*1e9)
    ax4.set_xlim(0.0,tmax*1e9)
    ax5.set_xlim(0.0,tmax*1e9)
    ax3.set_ylabel(r"$n_e$ (1/m$^3$)")
    ax4.set_ylabel(r"$V_e$ (mm/ns)")
    ax5.set_ylabel(r"$T_e$ (eV)")
    ax4.set_xlabel("Time (ns)")
    fig.tight_layout()
    fig.savefig(f'C:\\Users\\Aidan Crilly\\Documents\\GitHub\\ElectronBoltzmannArc\\images\\tidx_{i}.png')

