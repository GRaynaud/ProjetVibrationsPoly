# -*- coding: utf-8 -*-
"""
Code Python
Test d'implantation d'un schéma numérique pour un problème de poutre 1D
Cas de vibrations libres (pas de forçage)
conditions initiales : y(x,0) = 0, dy/dt(x,0) = 1
"""

# Bibliotheques
import numpy as np
import matplotlib.pyplot as plt

# Description des variables geometriques

L = 10. #m
Nx = 1000 # nombre de points pour discrétiser l'axe x
dx = L/Nx

# Description des variables temporelles

Tmax = 10
dt = 0.001
Nt = int(Tmax/dt)
t = np.linspace(0,Tmax,Nt)

# Description des variables dynamiques

EI = 1e4
rhoA = 1e0

# Conditions initiales
x = np.linspace(0,L,Nx)
y0 = np.zeros(Nx)
dty0 = 0.1*np.sin(np.pi*x/L)

# Construction de la derivee 4eme en espace

A = 6.*np.eye(Nx)
A += -4.*np.eye(Nx,Nx,1) + (-4.)*np.eye(Nx,Nx,-1)
A += np.eye(Nx,Nx,2) + np.eye(Nx,Nx,-2)
A[0,0] = -2. #-2.*dx**2
A[0,1] = 1 #dx**2
A[0,2] = 0
A[-1,-1] = -2. #-2.*dx**2
A[-1,-2] = 1 #dx**2
A[-1,-3] = 0
A *= 1./dx**4


# Construction de l'operateur du membre de gauche

B = np.eye(Nx) + 0.5*(dt**2)*(EI/rhoA)*A
invB = np.linalg.inv(B)

# Conteneur pour la solution
y = np.zeros((Nt,Nx))
y[0,:] = y0
y[1,:] = y0 + dt*dty0

# Intégration en temps

for i in range(2,Nt):
    X = 2.*y[i-1,:] - y[i-2,:] - 0.5*(dt**2)*(EI/rhoA)*np.dot(A,y[i-1,:])
    y[i,:] = np.dot(invB,X)

# Plot


plt.figure()
for i in range(Nt):
    if i%10 == 0:
        plt.plot(x,y[i,:], label=str(i))
plt.xlabel('Horizontal axis x')
plt.ylabel('Deflection (m)')
plt.legend()
plt.show()

    
plt.figure()
plt.plot(t,y[:,int(0.5*Nx)])
plt.xlabel('time t')
plt.ylabel('Mid span deflection (m)')    
    