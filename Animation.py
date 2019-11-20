# -*- coding: utf-8 -*-
"""
Code Python
Animation pour représenter les deux types d'excitation proposées
"""

#####################################
# Bibliotheques
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)

# Description des variables geometriques

L = 10. #m
Nx = 100 # nombre de points pour discrétiser l'axe x
dx = L/Nx
alpha0 = 0.6 #rad, angle d'attaque vertical
v = 1.3 #m/s vitesse du pieton
lj = 0.6 #m longueur d'une jambe
dpas = 2*lj*np.sin(alpha0) #m taille d'un pas
mg = 1e3 #N poids du marcheur

# Plot property
e = L/10. #epaisseur du rectangle
delta = e/3 #taille de l'appui

# Description des variables temporelles

Tmax = 10
dt = 0.05
Nt = int(Tmax/dt)
t = np.linspace(0,Tmax,Nt)


# Repertoire enregistrement

repertoire_f_continue = 'DataAnimaContinue/File_'
repertoire_f_discontinue = 'DataAnimaDiscontinue/File_'

# Premier type d'animation  continue
# Ensuite lancer dans la console windows
# ffmpeg -framerate 25 -start_number 0 -i File_%03d.png -vframes 200 -vcodec libx264 -crf 25 -pix_fmt yuv420p out.avi

for k in range(Nt):
    plt.figure()
    plt.axis('equal')
    # Plot contour rectangle + appui
    #rectangle
    plt.hlines(0,-e, L+e)
    plt.hlines(e,-e, L+e)
    plt.vlines(-e,0,e)
    plt.vlines(L+e,0,e)
    #appui gauche
    plt.plot([0, 0.5*delta],[0,-delta],c='black')
    plt.plot([-0.5*delta, 0],[-delta,0],c='black')
    plt.plot([-0.5*delta, 0.5*delta],[-delta,-delta],c='black')
    #appui droite
    plt.plot([L+0, L+0.5*delta],[0,-delta],c='black')
    plt.plot([L-0.5*delta, L+0],[-delta,0],c='black')
    plt.plot([L-0.5*delta, L+0.5*delta],[-delta,-delta],c='black')
    #Fleche
    xarrow = v*t[k]
    yarrow = 0.5*(1+np.cos(alpha0)) + 0.5*(1-np.cos(alpha0))*np.cos(v*t[k]/dpas)
    plt.arrow(xarrow,1.1*e+yarrow,0,-yarrow, shape='full', length_includes_head=True,
          head_width=0.1, head_length=0.1)
    plt.savefig(repertoire_f_continue+'%03d.png' % (k))
    plt.close()


## Test animation
#fig = plt.figure()
#ax = fig.gca()
#ax.axis('equal')
## Plot contour rectangle + appui
##rectangle
#ax.hlines(0,-e, L+e)
#ax.hlines(e,-e, L+e)
#ax.vlines(-e,0,e)
#ax.vlines(L+e,0,e)
##appui gauche
#ax.plot([0, 0.5*delta],[0,-delta],c='black')
#ax.plot([-0.5*delta, 0],[-delta,0],c='black')
#ax.plot([-0.5*delta, 0.5*delta],[-delta,-delta],c='black')
##appui droite
#ax.plot([L+0, L+0.5*delta],[0,-delta],c='black')
#ax.plot([L-0.5*delta, L+0],[-delta,0],c='black')
#ax.plot([L-0.5*delta, L+0.5*delta],[-delta,-delta],c='black')
#
#global line
#line = plt.arrow(0,0,0,0)
#ax.add_patch(line)
#
#def init():
##    line.set_data([], [])
##    global line
#    ax.patches.pop(0)
#    line = plt.Arrow(0,0,0,0)
#    ax.add_patch(line)
#    return (line,)
#    
#def animatecontinue(i):
##    global line
##    ax.patches.remove(line)
#    ax.patches.pop(0)
#    xarrow = v*t[i]
#    yarrow = 0.5*(1+np.cos(alpha0)) + 0.5*(1-np.cos(alpha0))*np.cos(v*t[i]/dpas)
#    line = plt.arrow(xarrow,1.1*e+yarrow,0,-yarrow, shape='full', length_includes_head=True,
#          head_width=0.1, head_length=0.1)
#    ax.add_patch(line)
#    return (line,)
#anim = animation.FuncAnimation(fig, animatecontinue, init_func=init, frames=Nt, interval=25, blit=True)
#
##
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=25, bitrate=1800)
##anim.save('AnimContinue.mp4', writer=writer)
#
#anim.save('AnimContinue.gif', writer='imagemagick', fps=25)


for k in range(Nt):
    plt.figure()
    plt.axis('equal')
    # Plot contour rectangle + appui
    #rectangle
    plt.hlines(0,-e, L+e)
    plt.hlines(e,-e, L+e)
    plt.vlines(-e,0,e)
    plt.vlines(L+e,0,e)
    #appui gauche
    plt.plot([0, 0.5*delta],[0,-delta],c='black')
    plt.plot([-0.5*delta, 0],[-delta,0],c='black')
    plt.plot([-0.5*delta, 0.5*delta],[-delta,-delta],c='black')
    #appui droite
    plt.plot([L+0, L+0.5*delta],[0,-delta],c='black')
    plt.plot([L-0.5*delta, L+0],[-delta,0],c='black')
    plt.plot([L-0.5*delta, L+0.5*delta],[-delta,-delta],c='black')
    #Fleche
    alpha = alpha0*(2*(v*t[k]/dpas - np.floor(v*t[k]/dpas))-1)
    xarrow = np.floor(v*t[k]/dpas)*dpas + np.sin(alpha)*lj
    yarrow = 1.1*e + lj*np.cos(alpha)
    dx = - lj*np.sin(alpha)
    dy = -lj*np.cos(alpha)
    
    plt.arrow(xarrow,yarrow,dx,dy, shape='full', length_includes_head=True,
          head_width=0.1, head_length=0.1)
    
    plt.savefig(repertoire_f_discontinue+'%03d.png' % (k))
    plt.close()