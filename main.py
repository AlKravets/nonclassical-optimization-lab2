import numpy as np
import matplotlib.pyplot as plt




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






class Engine:
    '''
    Класс в котором должна вычислятся вся физика.
    Для получения данных нужно будет толко подставить сетку.
    '''
    def __init__(self, V_inf = 1 + 0j, obstacle_func = create_obstacle_plate, x0 = 0, y0 = 0, M = 20, delta = 5*10**-2, reverse = True):
        """
        Инициализация объекта.
        V_inf -- начальная скорость
        obstacle_func -- функция вычислния препядствия
        M -- количество точек дискретных особенностей
        delta -- параметр для избежания деления на ноль
        """
        self.V_inf = V_inf
        self.M = M
        self.delta = delta
        
        self.obstacle = obstacle_func(x0,y0)
        if reverse:
            self.obstacle = np.flip(self.obstacle, axis = 1)
        
        # точки отрыва вихревой границы
        self.p_dots = self.obstacle[0] + 1j*self.obstacle[1]
        
        # создание массива дискр особ, точек коллокаций.
        self._dots_on_obstacle()

        # создание массива нормалей для точек коллокации
        self._kol_normals()

        # список массивов вихревых границ, которые выходят из p_dots
        self.Lv_dots = np.array([np.array([]) for _ in range(len(self.p_dots))])

        # саисок массивов коэфф. для Lv_dots
        self.gamma_p_i = np.array([np.array([]) for _ in range(len(self.p_dots))])

        # счетчик времени
        self.t = 0
        # величина шага
        self.tay = 0
        # счетчик шагов
        self.n = 0

        # начальные значения коэфф. G_i
        self.Gi = self.init_Gi()

               
    def _dots_on_obstacle(self):
        """
        Служебная функция для вычисления массива дискретных особ.
        и точек коллокации.
        """
        x, y = self.obstacle[:]

        # расстояние между вершинами препядствия
        distance = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        param = np.sum(distance)
        distance = distance/param

        # вычисление кол-ва точек на каждом отрезке препядствия
        dot_on_segment = (distance * self.M).astype(int)
        dot_on_segment[-1] = self.M- np.sum(dot_on_segment[:-1])

        # создание массива дискретных особенностей  
        self.disc_dots = np.concatenate( [np.linspace(x[i] + 1j*y[i], x[i+1] + 1j*y[i+1] ,
                    dot_on_segment[i]+1)[:-1] for i in range(dot_on_segment.shape[0]-1)] + 
                            [np.linspace(x[-2] + 1j*y[-2], x[-1] + 1j*y[-1] , dot_on_segment[-1])])

        # создание массива точек коллокаций
        self.kol_dots = (self.disc_dots[1:] + self.disc_dots[:-1])/2

        #массив индексов дли p_dots
        self.index_p_dots = []
        for dot in self.p_dots:
            self.index_p_dots.append(np.where(self.disc_dots == dot)[0][0])

    
    def _R0i (self,z_mesh, z1_mesh):
        """
        Служебная функция для вычисления конструкции из лекции 
        """
        
        # z_mesh, z1_mesh = np.meshgrid(z,z1)
        R_before = np.sqrt((z_mesh.real - z1_mesh.real)**2 + (z_mesh.imag - z1_mesh.imag)**2)
        return np.maximum(np.ones(R_before.shape)*self.delta, R_before)


    def _V_tech(self,z,z1):
        """
        Служебная функция для вычисления конструкции из лекции вида V(z.real, z.imag,z1.real, z1.imag)
        """
        z_mesh, z1_mesh = np.meshgrid(z,z1)
        R0i = self._R0i(z_mesh,z1_mesh)
        return np.array([(z1_mesh.imag - z_mesh.imag)/(2*np.pi*R0i**2), 
                        (z_mesh.real - z1_mesh.real)/(2*np.pi*R0i**2)])

        # return (z1_mesh.imag - z_mesh.imag)/(2*np.pi*R0i**2) +  1j*(z_mesh.real - z1_mesh.real)/(2*np.pi*R0i**2)
       
    def _kol_normals(self):
        """
        Служебная функция для вычисления нормалей в точках коллокации
        """
        self.kol_normals = (-1*(self.disc_dots[1:].imag - self.disc_dots[:-1].imag) +
                1j*(self.disc_dots[1:].real - self.disc_dots[:-1].real)) / \
                    np.sqrt( (self.disc_dots[1:].real - self.disc_dots[:-1].real)**2  +
                        (self.disc_dots[1:].imag - self.disc_dots[:-1].imag)**2)
        
    
    def V_t(self,z):
        """
        публичная функция для вычисления скорости в текущий момент в точках z
        """
        
        if self.n == 0:
            return (np.array((self.V_inf.real, self.V_inf.imag)).reshape(-1,1) + np.sum( self._V_tech(z, self.disc_dots) *
                    self.Gi.reshape(-1,1), axis = 1)).reshape((2,*z.shape))
        else:
            return (np.array((self.V_inf.real, self.V_inf.imag)).reshape(-1,1) + np.sum( self._V_tech(z, self.disc_dots) *
                    self.Gi.reshape(-1,1), axis = 1)).reshape((2,*z.shape)) +\
                        np.sum(self._V_tech(z,self.Lv_dots) * self.gamma_p_i.reshape(-1,1), axis = 1).reshape(2,*z.shape)
            
            r = (np.array((self.V_inf.real, self.V_inf.imag)).reshape(-1,1) + np.sum( self._V_tech(z, self.disc_dots) *
                    self.Gi.reshape(-1,1), axis = 1)).reshape((2,*z.shape))


            t = np.zeros((2,*z.shape))
            for i in range(self.Lv_dots.shape[0]):
                t1 = self._V_tech(z,self.Lv_dots[i])
                t = t+ np.array([np.sum(t1[0]*self.gamma_p_i[i].reshape(-1,1),axis=0),
                        np.sum(t1[1]*self.gamma_p_i[i].reshape(-1,1), axis=0)]).reshape((2,*z.shape))
            return r+t
    
    def Phi_t(self,z):
        """
        публичная функция для вычисления потенциала в текущий момент в точках z
        """
        if self.n == 0:
            z_m, d_m = np.meshgrid(z,self.disc_dots)
            return z.real*self.V_inf.real + z.imag*self.V_inf.imag + np.sum(np.multiply( np.arctan2((z_m.imag - d_m.imag),
                        (z_m.real - d_m.real)).T/(2*np.pi), self.Gi), axis=1).reshape(z.shape)
        else:
            z_lv_m, lv_m = np.meshgrid(z,self.Lv_dots)
            z_m, d_m = np.meshgrid(z,self.disc_dots)
            return (z.real*self.V_inf.real + z.imag*self.V_inf.imag + np.sum(np.multiply( np.arctan2((z_m.imag - d_m.imag),
                    (z_m.real - d_m.real)).T/(2*np.pi), self.Gi), axis=1).reshape(z.shape)+\
                        np.sum(np.arctan2((z_lv_m.imag - lv_m.imag), (z_lv_m.real - lv_m.real))/\
                            (2*np.pi) * self.gamma_p_i.reshape(-1,1), axis = 0).reshape(z.shape)).real
            

    
    def init_Gi(self):
        """
        вычисление начального значения Gi
        """
        n_m = np.meshgrid(self.kol_normals,self.disc_dots)[0]
        vj = self._V_tech(self.kol_dots, self.disc_dots)
        n_m = np.array([n_m.real, n_m.imag])

        lhs=  np.sum(n_m * vj, axis= 0).T 
        lhs = np.vstack((lhs, np.ones(lhs.shape[-1])))
        rhs = -1*(self.kol_normals.real*self.V_inf.real + self.kol_normals.imag*self.V_inf.imag)
        # rhs= rhs  + np.sum(np.multiply(self._V_tech(self.kol_dots, self.Lv_dots).reshape(
        #                 self.Lv_dots.shape[0]*self.Lv_dots.shape[1],-1),
        #                     self.gamma_p_i.reshape(-1,1)) * self.kol_normals, axis= 0 )

        # G0 = -1*np.sum(np.sum(self.gamma_p_i, axis=1), axis= 0)
        # print(G0)
        # print(rhs.shape)
        rhs = np.concatenate( [rhs, [-1*np.sum(np.sum(self.gamma_p_i, axis=1), axis= 0)]])

        return np.linalg.solve(lhs, rhs)


    def _update_Gi(self):
        """
        Обновление коэфф. Gi
        """
        n_m = np.meshgrid(self.kol_normals,self.disc_dots)[0]
        vj = self._V_tech(self.kol_dots, self.disc_dots)
        n_m = np.array([n_m.real, n_m.imag])

        lhs=  np.sum(n_m * vj, axis= 0).T 
        lhs = np.vstack((lhs, np.ones(lhs.shape[-1])))
        rhs = -1*(self.kol_normals.real*self.V_inf.real + self.kol_normals.imag*self.V_inf.imag)

        ###
        test = np.zeros(rhs.shape)
        for i in range(self.Lv_dots.shape[0]):
            tv1 = self._V_tech(self.kol_dots, self.Lv_dots[i])
            n_m1 = np.meshgrid(self.kol_dots, self.Lv_dots[i])[0]
            test = test + np.sum((tv1[0]*n_m1 + tv1[1]*n_m1)*self.gamma_p_i[i].reshape(-1,1),axis = 0)
            print(f'tv1.shape {tv1.shape}')
            print(f'n_m1.shape {n_m1.shape}')
            


        # 
        print(self._V_tech(self.kol_dots, self.Lv_dots).shape)
        print(self.gamma_p_i.reshape(-1,1).shape)
        # rhs= rhs - np.sum(np.sum(np.multiply(self._V_tech(self.kol_dots, self.Lv_dots),
        #             self.gamma_p_i.reshape(-1,1)),axis=1) * self.kol_normals, axis= 0 )

        rhs = rhs- test

        
        rhs = np.concatenate( [rhs, [-1*np.sum(np.sum(self.gamma_p_i, axis=1), axis= 0)]])

        self.Gi = np.linalg.solve(lhs, rhs)


    def _update_Lv(self):
        """
            Обновление l_v dots and коэфф. gamma_p_i и tay
        """
        V_lv = self.V_t(self.Lv_dots)
        V_lv = V_lv[0] + 1j*V_lv[1]
        print(f'V_lv.shape {V_lv.shape}')
        print(f'V_lv {V_lv}')

        V_pd = self.V_t(self.p_dots)
        V_pd = V_pd[0] + 1j*V_pd[1]


        #!!!!!!
        self.tay = self.delta / np.max(np.abs(V_pd))

        self.Lv_dots = self.Lv_dots + self.tay * V_lv

        new_lv_dots=  (self.p_dots + self.tay*V_pd).reshape(-1,1)

        print(f'new_lv_dots {new_lv_dots}')
        print(f'Lv_dots.shape {self.Lv_dots.shape}')

        
        # print(new_lv_dots)
        self.Lv_dots= np.hstack((new_lv_dots, self.Lv_dots))

        new_gamma_p_i = np.array([self.Gi[i] for i in self.index_p_dots]).reshape(-1,1)

        print(f'new_gamma_p_i {new_gamma_p_i}')
        self.gamma_p_i = np.hstack((new_gamma_p_i,self.gamma_p_i))


    def update(self):
        """
        шаг времени
        """
        self._update_Lv()

        self._update_Gi()
        print(f'Gi {self.Gi}')
        self.n+=1
        self.t+= self.tay
        
        print(f'LV dots {self.Lv_dots}')










if __name__ == "__main__":
    # V_inf= np.cos(np.pi/4) + 1j*np.sin(np.pi/4)
    V_inf = 1 + 0j
    eng = Engine(V_inf = V_inf,obstacle_func=create_obstacle_plate)
    print(eng.Gi)
    for i in range(2):
        print(f'{i+1}-----------------------------------')
        eng.update()
    # eng.update()
    size = 1.5
    step = 50
    # fig, ax = plt.subplots()
    scopes  = ((-size, size),(-size, size))
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    
    fig = plt.figure()
    ax = plt.axes(xlim=scopes[0], ylim=scopes[1])

    # матрица комплексных чисел
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1] 

    obstacle = eng.obstacle
    ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')

    for i in range(len(eng.p_dots)):
        ax.plot(eng.Lv_dots[i].real, eng.Lv_dots[i].imag)
        ax.scatter(eng.Lv_dots[i].real, eng.Lv_dots[i].imag)

    V = eng.V_t(z)
    

    # print(eng.V_t(eng.p_dots))

    # phi = eng.Phi_t(z)
    # # print(phi[10])
    

    # cmap= 'seismic'
    # levels = 50
    # if levels is None:
    #     cs= ax.contourf(x, y,phi,cmap=plt.get_cmap(cmap))
    # else:
    #     cs= ax.contourf(x, y,phi, levels = levels,cmap=plt.get_cmap(cmap))
    
    # cbar= plt.colorbar(cs, extendfrac='auto')
    ax.quiver(x,y,V[0],V[1])
    plt.show()

    