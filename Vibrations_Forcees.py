# -*- coding: utf-8 -*-
"""
Code Python
Test d'implantation d'un schéma numérique pour un problème de poutre 1D
Cas de vibrations forcees par le modele 1 (continue)
conditions initiales : y(x,0) = 0, dy/dt(x,0) = 0
"""

# Bibliotheques
import numpy as np
import matplotlib.pyplot as plt

# Description des variables geometriques

L = 10. #m
Nx = 100 # nombre de points pour discrétiser l'axe x
dx = L/Nx
alpha0 = 0.3 #rad, angle d'attaque vertical
v = 1 #m/s vitesse du pieton
dpas = 0.5 #m taille d'un pas
mg = 1e3 #N poids du marcheur

# Description des variables temporelles

Tmax = 10
dt = 0.005
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


# definition de l'excitation

# Excitation discontinue --> Cas 2
def f2(xpos,tpos):
    # on remplace le dirac par une gaussienne de variance tres faible
    sigma = dx
    fvalue = -1.*mg*np.cos(alpha0*(2*(v*tpos/dpas - np.floor(v*tpos/dpas))-1))
    fvalue *= 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(xpos-dpas*np.floor(v*tpos/dpas))**2/(2*sigma**2))
    return fvalue

# Excitation continue --> Cas 1
def f1(xpos,tpos):
    # on remplace le dirac par une gaussienne de variance tres faible
    sigma = dx
    fvalue = -1.*mg
    fvalue *= 0.5*(1+np.cos(alpha0)) + 0.5*(1-np.cos(alpha0))*np.cos(v*tpos/dpas)
    fvalue *= 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(xpos-v*tpos)**2/(2*sigma**2))
    return fvalue


# Intégration en temps

'''
(Id + EI*dt^2/2rhoA * A) y^k+1 = 2*y^k - y^k-1 - EI*dt^2/2rhoA * A*y^k + dt^2/rhoA * f(t)

A : operateur de derivation en espace
f(t) : vecteur dont les coordonnées sont f(x,t) --> excitation. Choisir au choix f1 ou f2

'''

for i in range(2,Nt):
    X = 2.*y[i-1,:] - y[i-2,:] - 0.5*(dt**2)*(EI/rhoA)*np.dot(A,y[i-1,:]) + (dt)**2/rhoA * f1(x,i*dt)
    y[i,:] = np.dot(invB,X)

# Plot


# Plot


#plt.figure()
#for i in range(Nt):
#    if i%10 == 0:
#        plt.plot(x,y[i,:], label=str(i))
#plt.xlabel('Horizontal axis x')
#plt.ylabel('Deflection (m)')
#plt.legend()
#plt.show()

# Plot du midspan deflection
    
plt.figure()
plt.plot(t,y[:,int(0.5*Nx)])
plt.xlabel('time t')
plt.ylabel('Mid span deflection (m)')    

    
    
    
    