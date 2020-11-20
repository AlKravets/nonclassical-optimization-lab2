import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




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

        # Вычисление нормалей в точках дискретных особенностях. 
        # Нормали в уголовых точках -- это нормализованная сумма соседних нормалей
        self.__normals_in_disc_dots = np.concatenate((self.kol_normals,[self.kol_normals[-1]]))
        for index in self.index_p_dots[1:-1]:
            self.__normals_in_disc_dots[index] = (self.__normals_in_disc_dots[index-1] + self.__normals_in_disc_dots[index+1])
            self.__normals_in_disc_dots[index] /= np.abs(self.__normals_in_disc_dots[index])
        # print(f'__normals_in_disc_dots {self.__normals_in_disc_dots}')
        
    
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
            
    
    def Phi_t(self,z):
        """
        публичная функция для вычисления потенциала в текущий момент в точках z
        Не работает
        # """
        # if self.n == 0:
        #     z_m, d_m = np.meshgrid(z,self.disc_dots)
        #     return z.real*self.V_inf.real + z.imag*self.V_inf.imag + np.sum(np.multiply( np.arctan2((z_m.imag - d_m.imag),
        #                 (z_m.real - d_m.real)).T/(2*np.pi), self.Gi), axis=1).reshape(z.shape)
        # else:
        #     z_lv_m, lv_m = np.meshgrid(z,self.Lv_dots)
        #     z_m, d_m = np.meshgrid(z,self.disc_dots)
        #     return (z.real*self.V_inf.real + z.imag*self.V_inf.imag + np.sum(np.multiply( np.arctan2((z_m.imag - d_m.imag),
        #             (z_m.real - d_m.real)).T/(2*np.pi), self.Gi), axis=1).reshape(z.shape)+\
        #                 np.sum(np.arctan2((z_lv_m.imag - lv_m.imag), (z_lv_m.real - lv_m.real))/\
        #                     (2*np.pi) * self.gamma_p_i.reshape(-1,1), axis = 0).reshape(z.shape)).real
        pass
            

    
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

        ### тестовый вариант
        # test = np.zeros(rhs.shape)
        # for i in range(self.Lv_dots.shape[0]):
        #     tv1 = self._V_tech(self.kol_dots, self.Lv_dots[i])
        #     n_m1 = np.meshgrid(self.kol_normals, self.Lv_dots[i])[0]
        #     test = test + np.sum((tv1[0]*n_m1.real + tv1[1]*n_m1.imag)*self.gamma_p_i[i].reshape(-1,1),axis = 0)
        #     print(f'tv1.shape {tv1.shape}')
        #     print(f'n_m1.shape {n_m1.shape}')
        # rhs = rhs- test     
        

        ## Вычитание вихревых точек 
        rhs= rhs - np.sum(np.sum(np.multiply(self._V_tech(self.kol_dots, self.Lv_dots),
                    self.gamma_p_i.reshape(-1,1)),axis=1) * np.array((self.kol_normals.real,self.kol_normals.imag)).reshape(2,-1), axis= 0 )     

        
        rhs = np.concatenate( [rhs, [-1*np.sum(np.sum(self.gamma_p_i, axis=1), axis= 0)]])

        self.Gi = np.linalg.solve(lhs, rhs)



    def _update_Lv(self):
        """
            Обновление l_v dots and коэфф. gamma_p_i и tay
        """
        V_lv = self.V_t(self.Lv_dots)
        
        V_lv = V_lv[0] + 1j*V_lv[1]
        
        # print(f'Lv_dots.shape {self.Lv_dots.shape}')
        # print(f'V_lv.shape {V_lv.shape}')


        V_pd = self.V_t(self.p_dots)
        V_pd = V_pd[0] + 1j*V_pd[1]

        



        #!!!!!!
        self.tay = self.delta / np.max(np.abs(V_pd))
        ###выор шага tay второй вариант делает шаг tay меньше.
        # self.tay = self.delta / np.max(np.abs(np.concatenate([V_lv.reshape(-1),V_pd])))

        
        
        new_lv_dots = self.Lv_dots + self.tay * V_lv
        # self.Lv_dots = new_lv_dots


        #костыль Это рабочая непроницаемость для пластинки (plate)
        # self.Lv_dots = kostil(self.Lv_dots,2*self.delta)

        # Условие непроницаемости
        self.Lv_dots = self._neproniknist1(new_lv_dots)

        # тут new_lv_dots -- это новые точки из p_dots
        new_lv_dots=  (self.p_dots + self.tay*V_pd).reshape(-1,1)
        self.Lv_dots= np.hstack((new_lv_dots,self.Lv_dots))

        new_gamma_p_i = np.array([self.Gi[i] for i in self.index_p_dots]).reshape(-1,1)

        self.gamma_p_i = np.hstack((new_gamma_p_i,self.gamma_p_i))


    def update(self):
        """
        шаг времени
        """
        self._update_Lv()

        self._update_Gi()

        self.n+=1
        self.t+= self.tay
        



    def __list_with_close_points(self,dot):
        """
        Возвращает список индексов точек из self.disc_dots, которые ближе к dot ( complex) на 2*delta
        """
        res = []
        for i in range(len(self.disc_dots)):
            if np.abs(self.disc_dots[i] - dot) < 2*self.delta:
                res.append(i)
        return res
    def _neproniknist1(self, new_lv_dots):
        """
        Функция, которая обеспечивает непроницамость границы
        возвращает массив новых точек,
        не изменяет точки, которые оторвутся на этом шаге        
        """

        shape = self.Lv_dots.shape


        old_dots = self.Lv_dots.reshape(-1)
        new_lv_dots = new_lv_dots.reshape(-1)
        
        size_dots = len(new_lv_dots)

        for i in range(size_dots):
            close_dots = self.__list_with_close_points(new_lv_dots[i])
            if len(close_dots):
                # print(f'dot_number {i}')
                # print(f'close_dots {close_dots}')
                # print(f'koodrs close dots {[self.disc_dots[i] for i in close_dots]}')
                new_lv_dots[i] = self.__find_first_new_dot(new_lv_dots[i], old_dots[i], close_dots)
        return new_lv_dots.reshape(shape)


    def __find_first_new_dot(self,new_dot, old_dot, close_dots):
        """
        Поиск новой точки (проекция новой точки на сферу в одной из точек close_dots по нормале в этой точке)
        """

        index_nearest_dot, flag = self.__find_nearest_dot(close_dots,new_dot)

        nearest_dot = self.disc_dots[index_nearest_dot]
        # print(f'index_nearest_dot {index_nearest_dot}')

        normal = self.__normals_in_disc_dots[index_nearest_dot]

        # эти условия должны обрабатывать случай, когда точка лежит в области угловой точки
        # и нужно, чтобы точка не проникала за границы, которые создают эту угловую точку.
        if flag:
            before_normal = self.__normals_in_disc_dots[index_nearest_dot-1]
            after_normal = self.__normals_in_disc_dots[index_nearest_dot+1]
            flag = np.sign(before_normal.imag*after_normal.real - before_normal.real* after_normal.imag )
            # print(f'flag = {flag}')
            if self.__sign_on_normal(old_dot,nearest_dot,normal)<0 and flag ==1:
                if self.__sign_on_normal(old_dot,nearest_dot,before_normal)>0:
                    normal = before_normal
                elif self.__sign_on_normal(old_dot,nearest_dot,after_normal)>0:
                    normal = after_normal
                else:
                    normal = self.__normals_in_disc_dots[index_nearest_dot]

            if self.__sign_on_normal(old_dot,nearest_dot,normal)>0 and flag == -1:
                if self.__sign_on_normal(old_dot,nearest_dot,before_normal)<0:
                    normal = before_normal
                elif self.__sign_on_normal(old_dot,nearest_dot,after_normal)<0:
                    normal = after_normal
                else:
                    normal = self.__normals_in_disc_dots[index_nearest_dot]
        
        # знаки старой и новой точки относительно нормали ближ. точки
        sign_old= self.__sign_on_normal(old_dot,nearest_dot,normal)
        sign_new = self.__sign_on_normal(new_dot,nearest_dot,normal)

        # print(f'sign {sign_old}')
        # print(f'sign new dot {self.__sign_on_normal(new_dot,nearest_dot,normal)}')
        # print(f'normal {normal}')

        # Вичисление коэф. на который будем умнодать нормаль
        if sign_old == sign_new:
            koef = sign_old*(2*self.delta - np.abs(nearest_dot - new_dot))
        else:
            koef = sign_old*(2*self.delta + np.abs(nearest_dot - new_dot))
        # print(f'realnew sign {self.__sign_on_normal(new_dot+koef*normal,nearest_dot,normal)}')
        return new_dot+koef*normal
        

    def __find_nearest_dot(self,  close_dots, new_dot):
        """
        Выбираю ближайшую точку из массива индексов точек
        возвращает индекс ближайшей точки
        flag -- это bool, который указывает, если точка находится рядом с угловой вершиной
        """
        flag = False
        for item in close_dots:
            if item in self.index_p_dots[1:-1]:
                flag = True
        return min(close_dots, key = lambda index: np.abs(self.disc_dots[index] - new_dot)), flag


    def __sign_on_normal(self,dot, nearest_dot, normal):
        """
        Возвращает знак точки относительно нормали ближайшей точки
        """
        return np.sign(normal.real*(dot-nearest_dot).real + normal.imag*(dot-nearest_dot).imag)        





def kostil(lv_dots, two_delta):
    """
    костиль для непроникності платівки
    """
    shape = lv_dots.shape
    lv_dots = lv_dots.reshape(-1)
    for i in range(len(lv_dots)):
        if -0.5 <= lv_dots[i].imag <= 0.5 and lv_dots[i].real < two_delta:
            lv_dots[i] = two_delta + 1j*lv_dots[i].imag
            

    return lv_dots.reshape(shape)


def main_animation_plate():
    """
    Тестоввая анимация, работает только для пластинки
    """
    name = "result_plate1.gif"
    frames = 200
    size = 1.5
    step = 50
    
    scopes  = ((-1*size, 15*size),(-5*size, 5*size))
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1]

    V_inf= np.cos(np.pi/4) + 1j*np.sin(np.pi/4)
    # V_inf = 1 + 0j
    eng = Engine(V_inf = V_inf,obstacle_func=create_obstacle_plate)
    
    fig = plt.figure()
    ax = plt.axes(xlim=scopes[0], ylim=scopes[1])
    
    scatter1 = ax.scatter([], [],c= 'r')
    scatter2 = ax.scatter([], [],c= 'b')
    V = eng.V_t(z)
    qv = ax.quiver(x, y, V[0], V[1])


    def init():
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {eng.t}, n = {eng.n}")
        obstacle = eng.obstacle
        ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')
        
        return scatter1, scatter2, qv,
        

    def animate(frame):
        eng.update()
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {round(eng.t,3)}, n = {eng.n}")
        
        scatter1.set_offsets(np.c_[eng.Lv_dots[0].real,eng.Lv_dots[0].imag])
        scatter2.set_offsets(np.c_[eng.Lv_dots[1].real,eng.Lv_dots[1].imag])
        
        return scatter1, scatter2, qv,
        

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames,interval=100, blit=True)
 
    # Сохраняем анимацию как gif файл
    anim.save(name)

def main_animation():
    """
    """
    name = "result_plate1.gif"
    frames = 200
    size = 1.5
    step = 50
    
    scopes  = ((-1*size, 4*size),(-1*size, 4*size))
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1]

    V_inf= np.cos(np.pi/4) + 1j*np.sin(np.pi/4)
    # V_inf = 1 + 0j
    eng = Engine(V_inf = V_inf,obstacle_func=create_obstacle_1, M =  20)
    
    fig = plt.figure()
    ax = plt.axes(xlim=scopes[0], ylim=scopes[1])
    
    scatter_plots = []

    for _ in range(len(eng.p_dots)):
        scatter_plots.append(ax.scatter([], []))

    V = eng.V_t(z)
    qv = ax.quiver(x, y, V[0], V[1])


    def init():
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {eng.t}, n = {eng.n}")
        obstacle = eng.obstacle
        ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')
        
        return qv,
        

    def animate(frame):
        eng.update()
        print(f'step ={frame+1}')
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {round(eng.t,3)}, n = {eng.n}")
        
        for i in range(len(eng.p_dots)):
            scatter_plots[i].set_offsets(np.c_[eng.Lv_dots[i].real,eng.Lv_dots[i].imag])
        
        return qv,
        

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames,interval=100, blit=True)
 
    # Сохраняем анимацию как gif файл
    anim.save(name)

def main_animation_with_params(name = "result_plate.gif",frames = 200, start_anim_from_moment = 0,scopes  = ((-1*1.5, 1*1.5),(-1*1.5, 1*1.5)), step = 50, figsize = None,show_steps = True,
                                V_inf = 1 + 0j, obstacle_func = create_obstacle_plate, x0 = 0, y0 = 0, M = 20, delta = 5*10**-2, reverse = True):
    """
    Функция, которая созраняет анимацию и получает все параметры.
    """

    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1]

    eng = Engine(V_inf, obstacle_func, x0, y0, M, delta, reverse)
    for i in range(start_anim_from_moment-1):
        eng.update()
        if show_steps:
            print(f'before animation: n ={eng.n}')
    
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    ax = plt.axes(xlim=scopes[0], ylim=scopes[1])
    
    scatter_plots = []

    for _ in range(len(eng.p_dots)):
        scatter_plots.append(ax.scatter([], []))

    V = eng.V_t(z)
    qv = ax.quiver(x, y, V[0], V[1])


    def init():
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {eng.t}, n = {eng.n}")
        obstacle = eng.obstacle
        ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')
        
        return qv,
        

    def animate(frame):
        eng.update()
        if show_steps:
            print(f'frame ={frame+1}, n = {eng.n}')
        V = eng.V_t(z)
        qv.set_UVC(V[0],V[1])
        ax.set_title(f"t = {round(eng.t,3)}, n = {eng.n}")
        
        for i in range(len(eng.p_dots)):
            scatter_plots[i].set_offsets(np.c_[eng.Lv_dots[i].real,eng.Lv_dots[i].imag])
        
        return qv,
        

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames,interval=100, blit=True)
 
    # Сохраняем анимацию как gif файл
    anim.save(name)


if __name__ == "__main__":
    
    # Под комментариями код, который создавал картинку в выбранный момент времени
    # size = 1.5
    # step = 50
    
    # scopes  = ((-size, 1*size),(-size, 1*size))
    # x = np.linspace(scopes[0][0], scopes[0][1], step)
    # y = np.linspace(scopes[1][0], scopes[1][1], step)
    
    
    # fig = plt.figure()
    # ax = plt.axes(xlim=scopes[0], ylim=scopes[1])



    # # V_inf= np.cos(np.pi/4) + 1j*np.sin(np.pi/4)
    # V_inf = 1 + 0j
    # eng = Engine(V_inf = V_inf,obstacle_func=create_obstacle_1)
    # print(eng.Gi)
    # print(f'eng.index_p_dots {eng.index_p_dots}')
    # for i in range(20):
    #     print(f'{i+1}-----------------------------------')
    #     eng.update()
        
    # xy = np.meshgrid(x, y)
    # z = xy[0] + 1j*xy[1] 

    # obstacle = eng.obstacle
    # ax.plot(obstacle[0], obstacle[1], linewidth= 2, color = 'black')

    # for i in range(len(eng.p_dots)):
    #     ax.plot(eng.Lv_dots[i].real, eng.Lv_dots[i].imag)
    #     ax.scatter(eng.Lv_dots[i].real, eng.Lv_dots[i].imag)

    # V = eng.V_t(z)
    # # V = eng.V_t(eng.p_dots)
    # # V = eng.V_t(eng.Lv_dots.reshape(-1))
    
    # # ax.quiver(eng.Lv_dots.reshape(-1).real,eng.Lv_dots.reshape(-1).imag,V[0],V[1])
    # ax.quiver(x,y,V[0], V[1])
    # # ax.quiver(eng.p_dots.real,eng.p_dots.imag,V[0], V[1])
    # # plt.show()
    

    size =1.5
    scopes = ((-1*size, 10*size),(-2*size, 2*size))
    main_animation_with_params(frames=200, start_anim_from_moment=400,scopes=scopes, obstacle_func=create_obstacle_1, figsize=(16,9))