import numpy as np
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from matplotlib.animation import FuncAnimation



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

def obstacle_picture(obstacle ,disctret_osob, kolok_dots, normals, size = 3):
    fig, ax= plt.subplots(figsize = (12,12))
    ax.set_title("Вигляд контуру")
    ax.set_xlim(-1*size,size)
    ax.set_ylim(-1*size,size)
    
    draw_obstacle(ax,obstacle)
    ax.quiver(kolok_dots.real, kolok_dots.imag, normals.real, normals.imag, width = 10**-3)
    s = 15
    ax.scatter(disctret_osob.real, disctret_osob.imag, marker = 'x', color = 'b', s =s)
    ax.scatter(kolok_dots.real, kolok_dots.imag, marker = 'o', color = 'r', s =s) 
    plt.show()


# функции создания преграды
def create_obstacle_plate(x0,y0):
    # return np.array([[x0, x0], [y0 - 1, y0 + 1]])
    return np.array([[x0, x0], [y0 - 0.5, y0 + 0.5]])


def create_obstacle_1( x0, y0):
    angle_koof = 3**0.5

    x1, y1 = x0, y0+ 3/4
    x2, y2 = x0-1/8 , (x0-1/8)*angle_koof +( y0 + 3/4 - angle_koof*x0)
    dot = np.array([[x0, x1, x2], [y0, y1, y2]])
    # x,y = dot[:]
    # print( ((y[1:] - y[:-1])**2 + (x[1:] - x[:-1])**2 )**0.5, np.sum(((y[1:] - y[:-1])**2 + (x[1:] - x[:-1])**2 )**0.5))
    return dot

def create_obstacle_5(x0,y0):
    x  = [x0, x0+0.5, x0+0.5, x0, x0, x0+0.5]
    y = [y0, y0, y0+0.5, y0+0.5, y0+0.75, y0+0.75]
    return np.array([x,y])

def create_obstacle_3(x0, y0):
    x = [x0, x0+.25, x0, x0+0.25, x0]
    y = [y0, y0+ .25, y0 + .5 , y0+ .75, y0 + 1]
    return np.array([x,y])

def create_obstacle_2(x0,y0):
    x = [x0, x0 -.25, x0 -.25, x0, x0, x0-.25]
    y = [y0, y0, y0 + .25, y0+.25 , y0+ .5 , y0+.5]
    return np.array([x,y])

# функции решения задачи
def analytical_solution_for_plate(u_inf, scopes, step):
    # создание сетки x, y
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)

    # создание массива комплексных значений
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1]

    # функция потенциала    
    Phi = u_inf * np.sqrt(z**2 + 1)
    # Phi = u_inf * np.sqrt(z**2) 

    # разбиение потенциала на искомую функцию и исправление
    # ошибки склейки корня
    normaling = 1*(z.real >0 ) + -1* (z.real < 0)
    phi = Phi.real * normaling
    psi = Phi.imag * normaling

    # скорость в комплексных координатах
    V = np.conj(u_inf * z/ np.sqrt(z**2 + 1))  * normaling
    
    # координаты вершин препядтвия
    obstacle = np.array([[0, 0], [-1, 1]])
    
    return (x,y, obstacle, phi, psi, V.real, V.imag)


def metrix(x):
    # x - array of complex
    return x/(x.real**2 + x.imag**2)**0.5



def dots_on_obstacle(obstacle, dots):
    # разделяем координаты вершин препядствия на x, y
    x, y = obstacle[:]
    
    # расстояние между вершинами препядствия
    distance = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    param = np.sum(distance)
    distance = distance/param
    
    # вычисление кол-ва точек на каждом отрезке препядствия
    dot_on_segment = (distance * dots).astype(int)
    dot_on_segment[-1] = dots- np.sum(dot_on_segment[:-1])

    # создание массива дискретных особенностей
    # disctret_osob = np.array([np.linspace(x[i] + 1j*y[i], x[i+1] + 1j*y[i+1] ,dot_on_segment[i]+1)[:-1] for i in range(dot_on_segment.shape[0]-1)] + [np.linspace(x[-2] + 1j*y[-2], x[-1] + 1j*y[-1] , dot_on_segment[-1])])
    
    disctret_osob = np.concatenate( [np.linspace(x[i] + 1j*y[i], x[i+1] + 1j*y[i+1] ,dot_on_segment[i]+1)[:-1] for i in range(dot_on_segment.shape[0]-1)] + [np.linspace(x[-2] + 1j*y[-2], x[-1] + 1j*y[-1] , dot_on_segment[-1])])

    # создание массива точек коллокаций
    kolok_dots = (disctret_osob[1:] + disctret_osob[:-1])/2
    kolok_dots_on_segment = np.copy(dot_on_segment)
    kolok_dots_on_segment[-1] -=1
    
    # массив нормалей выходящих из точек коллокаций
    # нужно умножить нормали на -1 если проход по контуру в другую сторону.
    # ls = [np.ones(kolok_dots_on_segment[i]) for i in range(len(kolok_dots_on_segment))]
    # time_num = np.array( ls) * ((y[1:] - y[:-1]) + 1j*(x[:-1] - x[1:]))
    # print(time_num.shape)
    # normals = np.concatenate(np.array( [np.ones(kolok_dots_on_segment[i]) for i in range(len(kolok_dots_on_segment))]) * ((y[1:] - y[:-1]) + 1j*(x[:-1] - x[1:]))) 
    
    # ar = np.array([])

    
    # normals = np.concatenate([np.ones(kolok_dots_on_segment[i]) for i in range(len(kolok_dots_on_segment))] * ((y[1:] - y[:-1]) + 1j*(x[:-1] - x[1:]))) 
    # Делаем векторы нормали равными 1
    # normals = metrix(normals)
    
    # Иной вариант нормалей
    normals = (-1*(disctret_osob[1:].imag - disctret_osob[:-1].imag) + 1j*(disctret_osob[1:].real - disctret_osob[:-1].real)) / \
        np.sqrt( (disctret_osob[1:].real - disctret_osob[:-1].real)**2  + (disctret_osob[1:].imag - disctret_osob[:-1].imag)**2)
    
    
    
    
    # print(normals, '------------')
    # print(x, y, disctret_osob.shape, disctret_osob, kolok_dots, kolok_dots)
    # print(kolok_dots_on_segment)

    return disctret_osob, kolok_dots, normals





def mesh_creator(x, diskr,r):
    x_mesh, diskr_mesh = np.meshgrid(x,diskr)[:]
    # print(x_mesh.shape, diskr_mesh.shape)
    R_before = np.sqrt((x_mesh.real - diskr_mesh.real)**2 + (x_mesh.imag - diskr_mesh.imag)**2)

    Rj = np.maximum(np.ones(R_before.shape)*r, R_before )

    Vj = np.array([(diskr_mesh.imag - x_mesh.imag)/(2*pi*Rj**2), (x_mesh.real - diskr_mesh.real)/(2*pi*Rj**2)])

    return x_mesh, diskr_mesh, Rj, Vj


def calculate_G(normals, kolok, diskr, r, G0, V_inf):
    normals_mesh , diskr_mesh, Rj, _ = mesh_creator(normals, diskr,r)
    _, _, _, Vj = mesh_creator(kolok, diskr,r)
    normals_mesh= np.array([normals_mesh.real, normals_mesh.imag])
    # print(normals_mesh.shape, diskr_mesh.shape, Rj.shape, Vj.shape)
    
    lhs = np.sum(normals_mesh * Vj, axis=0).T
    # print(lhs.shape)

    lhs = np.vstack((lhs, np.ones(lhs.shape[-1])))
    # print(lhs)


    rhs = -1*(normals.real*V_inf.real + normals.imag*V_inf.imag)
    rhs = np.concatenate( [rhs, [G0]])
    # print(rhs)

    res = np.linalg.solve(lhs, rhs)
    # print(res.shape)
    return res




def solver_var1(V_inf, G0,scopes, step,obstacle, M):
    # констанста для регуляризации
    r = 0.01

    # создание сетки x, y
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    
    # матрица комплексных чисел
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1] 
    
    # разбиваем скорость на составляющие
    u_inf , v_inf = V_inf.real , V_inf.imag
 
    #    
    disctret_osob, kolok_dots, normals = dots_on_obstacle(obstacle, M)

    # 
    z_mesh, diskr_mesh, Rj, Vj = mesh_creator(z, disctret_osob, r)
    # print(z_mesh.shape, diskr_mesh.shape, Rj.shape, Vj.shape)
    # print(normals.shape)

    # Подчет массива Г    
    G  =calculate_G(normals, kolok_dots, disctret_osob, r, G0, V_inf)
    # print(G.shape)

    # создание матрицы скоростей
    V = np.array([ (u_inf, v_inf)[i] + np.sum(np.multiply(Vj[i].T,G), axis=1) for i in range(2)]).reshape(2,step,step)
    # print(V.shape)
    
    # создание матрицы psi
    psi = z.imag * u_inf - z.real*v_inf - np.sum(np.multiply(np.log(Rj).T/(2*pi), G), axis = 1).reshape(step,step)
    # print(psi.shape)

    # создание матрицы phi
    phi = z.real*u_inf + z.imag*v_inf + \
        np.sum(np.multiply( np.arctan((z_mesh.imag - diskr_mesh.imag)/(z_mesh.real - diskr_mesh.real)).T/(2*pi), G), axis=1).reshape(step, step)
    # print(phi.shape)

    return (x,y, obstacle, phi, psi, V[0], V[1]), (obstacle, disctret_osob, kolok_dots, normals)

def Sum_G(G):
    res = []
    for i in range(1,len(G)):
        res.append(np.sum(G[:i]))
    return res

def part_of_phi_psi(z, diskr, kolok, sum_G, r):
    # print(f'z: {z.shape}, diskr: {diskr.shape}, kolok: {kolok.shape}, sum_G: {sum_G.shape}')
    z_m, k_m, Rj, _ = mesh_creator(z, kolok,r)
    # print(f'Rj: {Rj.shape}')
    new_dots = diskr[1:] - diskr[:-1]
    # print(f'new_dots: {new_dots.shape}')
    # z_m , nd_m , k_m = np.meshgrid(z, new_dots, kolok)[:]
    # print(f'z_m {z_m.shape}, nd_m: {nd_m.shape}, km: {k_m.shape}')
    nd_m = np.meshgrid(z, new_dots)[1]
    # print(f'z_m {z_m.shape}, nd_m: {nd_m.shape}, km: {k_m.shape}')
    
    part_phi = np.sum(np.multiply(((nd_m.imag*(z_m.real - k_m.real) - nd_m.real*(z_m.imag - k_m.imag))/ Rj**2 / (2*pi)).T,
                                    sum_G), axis = 1).reshape(z.shape)
    
    part_psi = np.sum(np.multiply(((nd_m.real*(z_m.real - k_m.real) + nd_m.imag*(z_m.imag - k_m.imag))/ Rj**2 / (2*pi)).T,
                                    sum_G), axis = 1).reshape(z.shape)
    # print(f'res: {part_phi.shape}')
    return part_phi, part_psi



def solver_var2(V_inf, G0,scopes, step,obstacle, M):
    # констанста для регуляризации
    r = 0.01

    # создание сетки x, y
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    
    # матрица комплексных чисел
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1] 
    
    # разбиваем скорость на составляющие
    u_inf , v_inf = V_inf.real , V_inf.imag
 
    #    
    disctret_osob, kolok_dots, normals = dots_on_obstacle(obstacle, M)

    # 
    z_mesh, diskr_mesh, Rj, Vj = mesh_creator(z, disctret_osob, r)
    # print(z_mesh.shape, diskr_mesh.shape, Rj.shape, Vj.shape)
    # print(normals.shape)

    # Подчет массива Г    
    G  =calculate_G(normals, kolok_dots, disctret_osob, r, G0, V_inf)
    # print(G.shape)
    
    sum_G = np.array(Sum_G(G))
    # print(sum_G.shape)

    part_phi , part_psi = part_of_phi_psi(z,disctret_osob, kolok_dots, sum_G,r)

    # создание матрицы скоростей
    V = np.array([ (u_inf, v_inf)[i] + np.sum(np.multiply(Vj[i].T,G), axis=1) for i in range(2)]).reshape(2,step,step)
    # print(V.shape)

    # print(Rj.shape)
    # создание матрицы psi
    psi = z.imag * u_inf - z.real*v_inf - part_psi - G0/(2*pi)*np.log(Rj[-1]).reshape(step, step)


    # print(psi.shape)

    # создание матрицы phi
    n1 = ((z - disctret_osob[-1]).real < 0).astype(int)
    # phi = z.real*u_inf + z.imag*v_inf + part_phi +\
    #         G0/(2*pi)*(np.arctan((z.imag- disctret_osob[-1].imag)/(z.real - disctret_osob[-1].real)) + np.pi*n1 )

    n2 = ((z - disctret_osob[-1]).imag < 0).astype(int)
    phi = z.real*u_inf + z.imag*v_inf + part_phi + \
            G0/(2*pi)*(np.arctan2((z.imag- disctret_osob[-1].imag),(z.real - disctret_osob[-1].real)) +np.pi*2*n2)



    # phi=    (np.angle(z - disctret_osob[-1]))

    # print(phi.shape)

    return (x,y, obstacle, phi, psi, V[0], V[1]), (obstacle, disctret_osob, kolok_dots, normals)





# объединение для лекционной задачи
def main_for_plate(size = 3, step = 50, levels = None):
    u_inf = 1 +0j
    scopes = ((-1*size,size),(-1*size,size))
    step = step
    data = analytical_solution_for_plate(u_inf, scopes, step)
    draw_all (*data, levels)




def main1(V_inf = 1+0j, G0 = 0,size = 3, step = 50,x0 = 0, y0 = 0, M = 50,obstacle_func = create_obstacle_plate, levels = None, draw_part = 0, size_obstacle = 3, reverse = False):
    scopes  = ((-size, size),(-size, size))
    obstacle = obstacle_func(x0,y0)
    if reverse:
        obstacle = np.flip(obstacle, axis = 1)
    data = solver_var1(V_inf, G0,scopes, step,obstacle, M)
    x,y,obstacle, phi, psi, Vx, Vy = data[0][:]
    # print(x.shape, y.shape, Vx.shape, Vy.shape, phi.shape)
    
    if draw_part == 0 or draw_part == 1:
        draw_all(*data[0],levels= levels)
    if draw_part == 0 or draw_part == 2:
        obstacle_picture(*data[1], size=size_obstacle)
    


def main2(V_inf = 1+0j, G0 = 0,size = 3, step = 50,x0 = 0, y0 = 0, M = 50,obstacle_func = create_obstacle_plate, levels = None, draw_part = 0, size_obstacle = 3, reverse = False):
    scopes  = ((-size, size),(-size, size))
    obstacle = obstacle_func(x0,y0)
    if reverse:
        obstacle = np.flip(obstacle, axis = 1)
    data = solver_var2(V_inf, G0,scopes, step,obstacle, M)
    x,y,obstacle, phi, psi, Vx, Vy = data[0][:]
    # print(x.shape, y.shape, Vx.shape, Vy.shape, phi.shape)
    
    if draw_part == 0 or draw_part == 1:
        draw_all(*data[0],levels= levels)
    if draw_part == 0 or draw_part == 2:
        obstacle_picture(*data[1], size=size_obstacle)


def main_compare(V_inf = 1+0j, G0 = 0,size = 3, step = 50,x0 = 0, y0 = 0, M = 50,obstacle_func = create_obstacle_plate, levels = None, draw_part = 0, size_obstacle = 3, reverse = False):
    scopes  = ((-size, size),(-size, size))
    obstacle = obstacle_func(x0,y0)
    if reverse:
        obstacle = np.flip(obstacle, axis = 1)
    data1 = solver_var1(V_inf, G0,scopes, step,obstacle, M)
    data2 = solver_var2(V_inf, G0,scopes, step,obstacle, M)
    
    draw_all_for_compare(data1[0], data2[0], levels = levels)


def TEST():
    V_inf = 1+0j
    G0 = 0
    size = 3
    step = 50
    x0 = 0
    y0 = 0
    M = 50
    obstacle_func = create_obstacle_plate
    levels = 50
    draw_part = 0
    size_obstacle = 3
    reverse = True

    scopes  = ((-size, size),(-size, size))
    obstacle = obstacle_func(x0,y0)
    if reverse:
        obstacle = np.flip(obstacle, axis = 1)
    data2 = solver_var2(V_inf, G0,scopes, step,obstacle, M)[0]
    return data2


def animation():
    figsize = (15,12)
    fig, ax  = plt.subplots(figsize=figsize)
    ani = FuncAnimation(fig, update_test, frames=np.linspace(0, 1, 5),interval = 200, blit=False)
    plt.show()


def update_test(frame):
    levels = 50
    args2 = TEST()
    x,y, obstacle, phi, psi, Vx, Vy = args2
    phi = phi *frame
    fig.clear()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    test = ax.text(0,0, f't = {round(frame,2)}')

    # picture(fig, ax, args2[0],args2[1],args2[2], args2[3], args2[5], args2[6], title ='Phi регуляризоване', levels = levels,  cbar_pos = 'horizontal')
    title =f'Phi регуляризоване'
    ax.set_title(title)
    draw_obstacle(ax, obstacle)
    cbar_pos = 'horizontal'
    # draw_new_colormap(fig,ax, x, y, phi, levels, cbar_pos= cbar_pos)
    cmap= 'seismic'
    # cmap = 'gnuplot'
    if levels is None:
        cs= ax.contourf(x, y,phi,cmap=plt.get_cmap(cmap))
    else:
        cs= ax.contourf(x, y,phi, levels = levels,cmap=plt.get_cmap(cmap))
    
    #cbar= plt.colorbar(cs, extendfrac='auto')
    # fig.colorbar(cs, ax = ax, orientation= cbar_pos )

    draw_velosity(ax, x,y, Vx,Vy)


    print('t =', frame)
    # return test, 





def sdf():
    def test(a,b):
        return a+b
    return test


if __name__ == "__main__":
    # V_inf = 1+0j
    # G0 = 0
    # size = 3
    # step = 10
    # x0 = 0
    # y0 = 0
    # M = 10 
    figsize =(6,8)
    fig, ax  = plt.subplots(figsize = figsize)
    ani = FuncAnimation(fig, update_test, frames=np.linspace(0, 1, 100),interval = 100, blit=False)
    plt.show()
    
    # levels = 50
    # args2 = TEST()
    # picture(fig, ax, args2[0],args2[1],args2[2], args2[3], args2[5], args2[6], title ='Phi регуляризоване', levels = levels,  cbar_pos = 'horizontal')
    # plt.show()
    # scopes  = ((-size, size),(-size, size))
    # obstacle = create_obstacle_plate(x0,y0)
    # data = solver_var2(V_inf, G0,scopes, step,obstacle, M)
    
    
    # main_compare(levels= 50, obstacle_func=create_obstacle_1, size = 1.5, reverse=True, V_inf=1 +1j)
    # main2(levels=50, G0 = 1,step= 100, draw_part=1,size= 1.5, obstacle_func=create_obstacle_1, reverse= True)
    

# def update(frame,fig = fig,ax = ax):
    
#     fig.clear()
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     # fig, ax = plt.subplots(figsize = figsize)
#     obstacle = np.array([[.1,.2,.3],[.1,.2,.3]])* frame
    
#     text = ax.text(0.5,0.5, f't = {frame}')
#     x,y,map = tcolor()
#     cmap= 'seismic'
#     cs = ax.contourf(x, y,map*frame,cmap=plt.get_cmap(cmap))
#     cbar = plt.colorbar(cs, extendfrac='auto')
#     # draw_colormap(ax,x,y,map)
#     # obs.set_data(obstacle[0], obstacle[1])
#     ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')

#     # draw_new_colormap(fig,ax, x, y, map, levels, cbar_pos= cbar_pos)

#     # draw_velosity(ax, x,y, Vx,Vy)

#     return text,

# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),interval = 100, blit=False)
# # ani.save('df.gif')
# plt.show()