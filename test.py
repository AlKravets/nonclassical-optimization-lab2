import numpy as np
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from matplotlib.animation import FuncAnimation

from main import *




# функции рисования
def draw_all(x,y, obstacle, phi, psi, Vx, Vy, levels = None ):
    
    draws = []
    figsize = (15,12)
    draws.append(standart_picture(x,y,obstacle, phi, Vx, Vy,figsize = figsize ,title = 'Phi', levels = levels))
    draws.append(standart_picture(x,y,obstacle, psi, Vx, Vy,figsize = figsize ,title = 'Psi', levels = levels))

    # draws.append(modified_picture(x,y,obstacle, phi, Vx, Vy, title = 'Phi'))
    # draws.append(modified_picture(x,y,obstacle, psi, Vx, Vy, title = 'Psi'))
    plt.show()


def draw_all_for_compare(args1, args2, levels = None ):
    #Памятка args  = (x,y,obstacle, phi, psi, Vx, Vy)
    figsize =(14,8) 
    fig1, axs1 = plt.subplots(1,2,figsize=figsize)
    # fig1, axs1 = plt.subplots(1,2)
    picture(fig1,axs1[0], args1[0],args1[1],args1[2], args1[3], args1[5], args1[6], title ='Phi', levels = levels, cbar_pos = 'horizontal')
    picture(fig1,axs1[1], args2[0],args2[1],args2[2], args2[3], args2[5], args2[6], title ='Phi регуляризоване', levels = levels,  cbar_pos = 'horizontal')
    
    figsize =(16,6)
    fig2, axs2 = plt.subplots(1,2,figsize=figsize )
    # fig2, axs2 = plt.subplots(1,2)
    picture(fig2,axs2[0], args1[0],args1[1],args1[2], args1[4], args1[5], args1[6], title ='Psi', levels = levels)
    picture(fig2,axs2[1], args2[0],args2[1],args2[2], args2[4], args2[5], args2[6], title ='Psi регуляризоване', levels = levels)

    plt.show()


def picture(fig, ax,x,y,obstacle, map, Vx, Vy, title ='', levels = None, cbar_pos = 'vertical'):
    ax.set_title(title)
    draw_obstacle(ax, obstacle)

    draw_new_colormap(fig,ax, x, y, map, levels, cbar_pos= cbar_pos)

    draw_velosity(ax, x,y, Vx,Vy)
    

def draw_new_colormap(fig,ax, x, y, map, levels = None, cbar_pos = 'vertical'):
    cmap= 'seismic'
    # cmap = 'gnuplot'
    if levels is None:
        cs= ax.contourf(x, y,map,cmap=plt.get_cmap(cmap))
    else:
        cs= ax.contourf(x, y,map, levels = levels,cmap=plt.get_cmap(cmap))
    
    #cbar= plt.colorbar(cs, extendfrac='auto')
    cbar= fig.colorbar(cs, ax = ax, orientation= cbar_pos )



def standart_picture(x,y,obstacle, map, Vx, Vy, figsize = None, title ='', levels = None):
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(title)
    draw_obstacle(ax, obstacle)

    draw_colormap(ax, x, y, map, levels)

    draw_velosity(ax, x,y, Vx,Vy)

    return fig, ax


def modified_picture(x,y,obstacle, map, Vx, Vy, figsize = None, title =''):
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(title)
    draw_obstacle(ax, obstacle)
    draw_colorvelosity(ax, x, y, Vx, Vy, map)
    

    return fig, ax
    

def draw_obstacle(ax, obstacle):
    ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')


def draw_colormap(ax, x, y, map, levels = None):
    cmap= 'seismic'
    # cmap = 'gnuplot'
    if levels is None:
        cs= ax.contourf(x, y,map,cmap=plt.get_cmap(cmap))
    else:
        cs= ax.contourf(x, y,map, levels = levels,cmap=plt.get_cmap(cmap))
    
    cbar= plt.colorbar(cs, extendfrac='auto')


def draw_velosity(ax, x,y, Vx,Vy):
    ax.quiver(x,y,Vx,Vy)


def draw_colorvelosity(ax, x, y, Vx, Vy, map):
    cs = ax.quiver(x,y,Vx,Vy, map, cmap=plt.get_cmap("seismic"))
    # cs = ax.quiver(x,y,Vx,Vy, map, cmap=plt.get_cmap("cool"))
    cbar= plt.colorbar(cs, extend='both', extendfrac='auto')


def draw_pretty_obstacle(ax,obstacle ,disctret_osob, kolok_dots, normals):
    draw_obstacle(ax,obstacle)
    ax.quiver(kolok_dots.real, kolok_dots.imag, normals.real, normals.imag, width = 10**-3)
    ax.scatter(disctret_osob.real, disctret_osob.imag, marker = 'x', color = 'b', s =10)
    ax.scatter(kolok_dots.real, kolok_dots.imag, marker = 'o', color = 'r', s =10)

def obstacle_picture(obstacle ,disctret_osob, kolok_dots, normals, size = 1):
    fig, ax= plt.subplots(figsize = (6,6))
    ax.set_xlim(-1*size,size)
    ax.set_ylim(-1*size,size)
    
    draw_obstacle(ax,obstacle)
    ax.quiver(kolok_dots.real, kolok_dots.imag, normals.real, normals.imag, width = 10**-3)
    s = 15
    ax.scatter(disctret_osob.real, disctret_osob.imag, marker = 'x', color = 'b', s =s)
    ax.scatter(kolok_dots.real, kolok_dots.imag, marker = 'o', color = 'r', s =s) 
    plt.show()

if __name__ == "__main__":
    eng = Engine(reverse= True, obstacle_func=create_obstacle_1)

    obstacle_picture(eng.obstacle, eng.disc_dots, eng.kol_dots, eng.kol_normals)