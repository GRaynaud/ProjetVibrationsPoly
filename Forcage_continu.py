# -*- coding: utf-8 -*-
"""
Code Python
Test des angles d'incidence du piéton pour lequel une poutre va entrer en résonance
Cas de vibrations forcees par le modele 1 (continue)
"""
#####################################
# Bibliotheques
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


# Description des variables geometriques
L = 11. #m - valeur article
alpha0 = (90-69)*np.pi/180. #rad, angle d'attaque vertical, valeur article
v = 1.2 #m/s vitesse du pieton
lj = 1 #m longueur d'une jambe
Tfinal = L/v

# Description des variables dynamiques
EI = 1.64e8 #valeur article
rhoA = 1.364e3 #valeur article
mg = 80*9.81 #N poids du marcheur - valeur article

# Nombre de modes considérés
kmax = 20

# Pulsations caractéristiques

def beta_p(alpha0,v):
    return (np.pi *v) / (lj * np.sin(alpha0))

def omega(k):
    omega0 = ((np.pi/L)**2)*np.sqrt(EI/rhoA)
    return k*k*omega0

def Omega(k,v):
    Omega0 = np.pi*v/L
    return k*Omega0

# Constantes caractéristiques

def B_p(alpha0):
    return mg * ((1 + np.cos(alpha0)) / 2) * np.sqrt(2 / (rhoA * L))

def C_p(alpha0):
    return mg * ((1 - np.cos(alpha0)) / 2) * np.sqrt(2 / (rhoA * L))

####################################
# On trace les angles d'incidence qui mènent à une résonance à v fixé.

def angle_res():
    C = np.pi*v/lj
    eps = 1e-3
    Omega0 = Omega(1,v)
    omega0 = omega(1)
    if abs( round(Omega0/omega0) - Omega0/omega0 ) < eps:
        print("La vitesse du piéton est telle que le pont rentre en résonance")

    plt.figure()
    for k in range(1,kmax+1):
        alpha = np.arcsin(C/abs(omega(k) - Omega(k,v))) * 180/np.pi
        plt.plot([alpha,alpha],[0,1/k])
    plt.xlabel('Alpha_0 (°)')
    plt.show()

####################################
# On trace les vitesses qui mènent à une résonance à alpha0 fixé.

def vitesse_res():
    vmax = 100
    plt.figure()
    for k in range(1,kmax+1):
        omegak = omega(k)
        v1 = omegak / (np.pi * (k/L))
        v2 = omegak / (np.pi * ( (k/L) + 1/(lj * np.sin(alpha0)) ))
        v3 = omegak / (np.pi * ( (k/L) - 1/(lj * np.sin(alpha0)) ))
        if v1>0 and v1<vmax:
            plt.plot([v1,v1],[0,1/k])
        if v2>0 and v2<vmax:
            plt.plot([v2,v2],[0,1/k])
        if v3>0 and v3<vmax:
            plt.plot([v3,v3],[0,1/k])
    plt.xlabel('v (m/s)')
    plt.show()

####################################
# On trace les couples (v,alpha0) qui mènent à une résonance.

def couple_res():
    nb_alpha = 90
    alpha = np.linspace(1,90,nb_alpha)
    vmax = 10
    plt.figure()
    for k in range(1,kmax+1):
        omegak = omega(k)
        tab_alpha = []
        tab_v = []
        for i in range(nb_alpha):
            alpha0 = alpha[i] * np.pi/180.
            v1 = omegak / (np.pi * (k/L))
            v2 = omegak / (np.pi * ( (k/L) + 1/(lj * np.sin(alpha0)) ))
            v3 = omegak / (np.pi * ( (k/L) - 1/(lj * np.sin(alpha0)) ))
            if v1>0 and v1<vmax:
                tab_alpha.append(alpha[i])
                tab_v.append(v1)
            if v2>0 and v2<vmax:
                tab_alpha.append(alpha[i])
                tab_v.append(v2)
            if v3>0 and v3<vmax:
                tab_alpha.append(alpha[i])
                tab_v.append(v3)
        plt.scatter(tab_alpha,tab_v)
    plt.xlabel('alpha_0 (°)')
    plt.ylabel('v (m/s)')
    plt.show()

####################################
# On calcule l'amplitude du pont au cours du temps

def y(x,t,alpha0,v):
    y_p = 0
    B = B_p(alpha0)
    C = C_p(alpha0)
    beta = beta_p(alpha0,v)
    for k in range(1,kmax+1):
        omegak = omega(k)
        Omegak = Omega(k,v)
        bk = - (1/omegak) * ( ((Omegak*B)/(omegak**2 - Omegak**2)) + (((Omegak+beta)*C)/(2 * (omegak**2 - (Omegak+beta)**2))) + (((Omegak-beta)*C)/(2 * (omegak**2 - (Omegak-beta)**2))) )
        K1 = bk * np.sin(omegak * t) + (B/(omegak**2 - Omegak**2)) * np.sin(Omegak * t) + (C/(2 * (omegak**2 - (Omegak+beta)**2))) * np.sin((Omegak+beta) * t) + (C/(2 * (omegak**2 - (Omegak-beta)**2))) * np.sin((Omegak-beta) * t)
        y_p += K1 * np.sqrt(2/(rhoA * L)) * np.sin(k * np.pi *x/L)
    return(-y_p)

####################################
# On affiche l'évolution du pont au cours du temps

def y_maj(alpha0,v):
    y_m = 0
    B = B_p(alpha0)
    C = C_p(alpha0)
    beta = beta_p(alpha0,v)
    for k in range(1,kmax+1):
        omegak = omega(k)
        Omegak = Omega(k,v)
        bk = (1/omegak) * ( abs((Omegak*B)/(omegak**2 - Omegak**2)) + abs(((Omegak+beta)*C)/(2 * (omegak**2 - (Omegak+beta)**2))) + abs(((Omegak-beta)*C)/(2 * (omegak**2 - (Omegak-beta)**2))) )
        K1 = bk + abs(B/(omegak**2 - Omegak**2)) + abs(C/(2 * (omegak**2 - (Omegak+beta)**2))) + abs(C/(2 * (omegak**2 - (Omegak-beta)**2)))
        y_m += K1 * np.sqrt(2/(rhoA * L))
    return(y_m)

def plot_evolution():
    Nx = 100
    Nt = 100
    dt = Tfinal/Nt
    tab_x = np.linspace(0,L,Nx)
    tab_t = np.linspace(0,Tfinal,Nt)
    ymaj = 1e3*y_maj(alpha0,v)
    for i in range(Nt):
        tab_y = np.zeros(Nx)
        t = tab_t[i]
        for j in range(Nx):
            x = tab_x[j]
            tab_y[j] = y(x,t,alpha0,v)
        x_pieton = v*t
        y_pieton = y(x_pieton,t,alpha0,v)
        plt.figure(1)
        plt.clf()
        plt.plot(tab_x,1e3*tab_y)
        plt.scatter(x_pieton,1e3*y_pieton)
        plt.xlabel('x (m)')
        plt.ylabel('y ($\times 10^{-3}$m)')
        plt.title('$t=$%s' %t)
        axes = plt.gca()
        axes.set_ylim([-ymaj,ymaj])
        plt.pause(dt)

####################################
# On calcule l'amplitude max du pont

def y_max(alpha0,v):
    Nx = 100
    Nt = 100
    tab_x = np.linspace(0,L,Nx)
    tab_t = np.linspace(0,Tfinal,Nt)
    ymax = 0
    for i in range(Nt):
        t = tab_t[i]
        for j in range(Nx):
            x = tab_x[j]
            y_p = y(x,t,alpha0,v)
            if abs(y_p) > ymax:
                ymax = abs(y_p)

def amplitude_angle():
    nb_alpha = 100
    alpha_min = 1
    alpha_max = 70
    tab_alpha = np.linspace(alpha_min,alpha_max,nb_alpha)
    tab_ymax = np.zeros(nb_alpha)
    for i in range(nb_alpha):
        alpha0 = tab_alpha[i] * np.pi/180
        tab_ymax[i] = y_maj(alpha0,v)
    plt.figure()
    plt.plot(tab_alpha,tab_ymax)
    plt.xlabel('alpha_0 (°)')
    plt.ylabel('amplitude_max (m)')
    plt.show()

def amplitude_vitesse():
    nb_vitesse = 100
    vmin = 0.1
    vmax = 10
    tab_vitesse = np.linspace(vmin,vmax,nb_vitesse)
    tab_ymax = np.zeros(nb_vitesse)
    for i in range(nb_vitesse):
        v = tab_vitesse[i]
        tab_ymax[i] = y_maj(alpha0,v)
    plt.figure()
    plt.plot(tab_vitesse,tab_ymax)
    plt.xlabel('v (m/s)')
    plt.ylabel('amplitude_max (m)')
    plt.show()

def amplitude_couple():
    nb_alpha = 100
    alpha_min = 1
    alpha_max = 45
    nb_vitesse = 100
    vmin = 0.1
    vmax = 10
    tab_alpha = np.linspace(alpha_min,alpha_max,nb_alpha)
    tab_vitesse = np.linspace(vmin,vmax,nb_vitesse)
    tab_ymax = np.zeros([nb_alpha,nb_vitesse])
    for i in range(nb_alpha):
        for j in range(nb_vitesse):
            alpha0 = tab_alpha[i] * np.pi/180
            v = tab_vitesse[j]
            tab_ymax[i,j] = y_maj(alpha0,v)
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111, projection='3d')
    tab_alpha2,tab_vitesse2 = np.meshgrid(tab_alpha,tab_vitesse)
    mycmap = plt.get_cmap('Reds')
    ax1.set_xlabel('alpha_0 (°)')
    ax1.set_ylabel('vitesse (m/s)')
    ax1.set_zlabel('amplitude_max (m)')
    surf1 = ax1.plot_surface(tab_alpha2, tab_vitesse2, tab_ymax, cmap=mycmap)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    plt.show()