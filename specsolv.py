#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft


"""
Solve a simple SE problem.
Philip Mocz (2016)
Harvard University

i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
Domain: [-L,L]  (periodic)
Potential: e.g. 1/2 x^2
Initial condition: ???

usage: python specsolv.py problem#
usage example: python specsolv.py 0
"""

def main(problem):
  
  
  if (problem == "0"): # Particle in harmonic oscillator potential
    # parameters
    N = 1024                        # spatial resolution
    Nt = 1000                       # number of timesteps
    Tend = 1.0                      # end of simulation
    Tout = 0.2                      # draw solution every Tout time interval    
      
    # domain 
    L = 1    
    x = np.linspace(-L,L, num=N+1)  # Note: x=-1 & x=1 are the same point!
    x = x[0:N]                      # chop off periodic point
      
    # initial condition
    psi = np.sqrt( np.exp(-x**2*40)/(np.sqrt(np.pi/10)/2) * 0.5 + 0.25 ) 
    
    # potential
    V = 0.5 * x**2
          
  elif (problem == "1"): # Free particle
    # parameters
    N = 1024  
    Nt = 1000 
    Tend = 1.
    Tout = 0.1
      
    # domain     
    L = 8    
    x = np.linspace(-L,L, num=N+1)  # Note: x=-1 & x=1 are the same point!
    x = x[0:N]                      # chop off periodic point                
 
    # initial condition
    psi = np.pi**(-0.25) * np.exp(-x**2 / 2.0) * np.exp(1.j*x)
    
    # potential
    V = 0*x
    
  else: # Particle in SHO - c.f. Mocz & Succi (2015) Fig. 2
    # parameters
    N = 1024  
    Nt = 1000 
    Tend = 3.*np.pi/8.
    Tout = np.pi/8.
      
    # domain     
    L = 4    
    x = np.linspace(-L,L, num=N+1)  # Note: x=-1 & x=1 are the same point!
    x = x[0:N]                      # chop off periodic point                
 
    # initial condition
    psi = np.pi**(-0.25) * np.exp(-x**2 / 2.0) * np.exp(1.j*x)
    
    # potential
    V = 0.5 * x**2
    
    
    
  # other variables   
  dx = x[1]-x[0]                  # domain spacing 
  Nout = int(Tout/Tend * Nt)      # output every Nout steps 
  dt = Tout / Nout                # timestep
  k = 2.0*  np.pi / (2.*L) * np.arange(-N/2, N/2)   # fourier space wave numbers


  # print np.sum( dx * abs(psi)**2 )    # check normalization (=1)
  
  
  
  # main loop (time evolution)
  for i in np.arange(0,Nt+1):

    # plot solution every Nout steps
    t = i*dt
    if( (i % Nout) == 0):
      fig = plt.plot(x, abs(psi)**2, linewidth=2, color=[1.*i/Nt, 0, 1.-1.*i/Nt], label='$t='+"%.2f" % t +'$')

    # fourier transform psi
    psihat = np.fft.fftshift( np.fft.fft(psi) );
    
    # evolve solution with kinetic term
    psihat = np.exp(dt * (-1.j*k**2/2.))  * psihat;
    
    # undo fft
    psi = np.fft.ifft( np.fft.ifftshift(psihat) );
    
    # evolve solution with potential term
    psi = np.exp(-1.j*dt*V) * psi;
  
  plt.legend()
  plt.xlabel('$x$')
  plt.ylabel('$|\psi|^2$')
  plt.savefig('solution' + problem + '.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
  plt.close()
    


if __name__ == "__main__":
  main(sys.argv[1])
