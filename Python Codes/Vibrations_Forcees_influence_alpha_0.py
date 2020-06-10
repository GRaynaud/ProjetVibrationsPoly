# -*- coding: utf-8 -*-
"""
Code Python
Test d'implantation d'un schéma numérique pour un problème de poutre 1D
Cas de vibrations forcees par le modele 1 (continue)
conditions initiales : y(x,0) = 0, dy/dt(x,0) = 0
"""
#####################################
# Bibliotheques
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)

ListeTHETA_0_deg = np.linspace(60,89,100)

liste_peak_acc_f2 = []
liste_peak_acc_f1 = []


for theta_0 in ListeTHETA_0_deg:
    # Description des variables geometriques
    
    L = 11. #m - valeur article
    Nx = 500 # nombre de points pour discrétiser l'axe x
    dx = L/Nx
    alpha0 = (90-theta_0)*np.pi/180. #rad, angle d'attaque vertical, valeur article
    v = 1.25 #m/s vitesse du pieton
    lj = 1 #m longueur d'une jambe
    dpas = 2*lj*np.sin(alpha0) #m taille d'un pas
    mg = 80*9.81 #N poids du marcheur - valeur article
    
    # Description des variables temporelles
    
    Tmax = L/v
    dt = 0.001
    Nt = int(Tmax/dt)
    t = np.linspace(0,Tmax,Nt)
    
    # Description des variables dynamiques
    
    EI = 1.64e8 #valeur article
    rhoA = 1.364e3 #valeur article
    
    # Conditions initiales
    x = np.linspace(0,L,Nx)
    y0 = np.zeros(Nx)
    dty0 = np.zeros(Nx)
    
    ####################################
    # Conteneur pour la solution
    y = np.zeros((Nt,Nx))
    y[0,:] = y0
    y[1,:] = y0 + dt*dty0
    
    ###################################
    # Construction de la derivee 4eme en espace - operateur du membre de droite
    
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
    
    ####################################"
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
    
    
    #################################
    # Intégration en temps
    
    '''
    1/2 - Schema Implicite en temps :
        
    (Id + EI*dt^2/2rhoA * A) y^k+1 = 2*y^k - y^k-1 - EI*dt^2/2rhoA * A*y^k + dt^2/rhoA * f(t)
    
    y^k : vecteur y(x,k*dt) de taille Nx 
    A : operateur de derivation en espace
    f(t) : vecteur dont les coordonnées sont f(x,t) --> excitation. Choisir au choix f1 ou f2
    
    '''
    # Frcage discontinu
    for i in range(2,Nt):
        X = 2.*y[i-1,:] - y[i-2,:] - 0.5*(dt**2)*(EI/rhoA)*np.dot(A,y[i-1,:]) + (dt**2)/rhoA * f2(x,i*dt)
        y[i,:] = np.dot(invB,X)
    
    d = y[:,int(0.5*Nx)]
    acc = np.diff(np.diff(d))*1./dt**2
    f0 = 2*(np.pi/L)**2*np.sqrt(EI/rhoA) * dt
    b, a = scipy.signal.butter(3, f0) # paramètres du filtre pour couper les HF
    acc = scipy.signal.filtfilt(b,a,acc)
    
    liste_peak_acc_f2.append(max(abs(acc)))
    
    print('-', end='')
    
#    # Forcage continu
#    y = np.zeros((Nt,Nx))
#    y[0,:] = y0
#    y[1,:] = y0 + dt*dty0
#    
#    for i in range(2,Nt):
#        X = 2.*y[i-1,:] - y[i-2,:] - 0.5*(dt**2)*(EI/rhoA)*np.dot(A,y[i-1,:]) + (dt**2)/rhoA * f1(x,i*dt)
#        y[i,:] = np.dot(invB,X)
#    
#    d = y[:,int(0.5*Nx)]
#    acc = np.diff(np.diff(d))*1./dt**2
#    f0 = 2*(np.pi/L)**2*np.sqrt(EI/rhoA) * dt
#    b, a = scipy.signal.butter(3, f0) # paramètres du filtre pour couper les HF
#    acc = scipy.signal.filtfilt(b,a,acc)
#    
#    liste_peak_acc_f1.append(max(abs(acc)))
        
    
    
    print(str(len(liste_peak_acc_f2)),end='')


plt.figure()
#plt.plot(ListeTHETA_0_deg,liste_peak_acc_f1,marker='o',markerfacecolor='none', label='Forçage Continu')
plt.plot(ListeTHETA_0_deg,liste_peak_acc_f2,marker='s',markerfacecolor='none', label='Forçage Discontinu')
plt.xlabel('Angle of attack $\\theta_0$ (°)')
plt.ylabel('Peak acceleration at midspan ($m.s^{-2}$)')
plt.title('$v = '+str(v)+'$ - Forçage discontinu')
#plt.legend()
plt.tight_layout()