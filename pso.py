import copy
import math
import random


class Particle:
    """
    粒子个体类
    """

    def __init__(self, n_dimensions, bounds):
        """
        初始化粒子
        :param n_dimensions: 维度
        :param bounds: 边界
        """
        self.position = [] # 位置
        self.velocity = [] # 速度
        self.best_position = [] # 最佳位置
        self.best_fitness = 1000000 # 适应度即函数值, 初始值设为一个很大的数，因为要求两个函数的最小值

        # 初始化边界，位置和速度
        for dim in range(n_dimensions):
            # 边界，有多少个维度边界就有多少，每个维度都有min和max
            min_bound, max_bound = bounds[dim]
            # 位置的初始位置设定为边界内任意位置
            self.position.append(random.uniform(min_bound, max_bound))
            # 速度的范围设定为边界的10%
            max_velocity = (max_bound - min_bound) / 10.0
            # 速度的初始值设定为范围内任意值
            self.velocity.append(random.uniform(-max_velocity, max_velocity))
        #开始时，最佳位置就是初始位置
        self.best_position = copy.deepcopy(self.position)

class PSO:
    """
    粒子群优化算法核心类
    """

    def __init__(self, objective_func, n_particles, n_dimensions, bounds, w, c1, c2, max_iter):
        """
        初始化粒子群
        :param objective_func: 目标函数
        :param n_particles: 粒子数量
        :param n_dimensions: 函数维度，同时也是粒子的维度
        :param bounds: 搜索边界
        :param w: 惯性权重，即速度方向的影响因子
        :param c1: 个体学习因子
        :param c2: 社会学习因子
        :param max_iter: 最大迭代次数
        """
        self.objective_func = objective_func
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        self.bounds = bounds or [(-10, 10)] * n_dimensions # 默认边界为[-10, 10]

        self.particles = [] # 粒子群
        self.global_best_position = None # 全局最优位置
        self.global_best_fitness = 1000000 # 全局最优适应度
        self.history = [] # 记录每一次迭代的最优位置

        #初始化粒子群
        self.init_particles()

    def init_particles(self):
        """
        初始化粒子群
        """
        # 创建指定数量的粒子
        self.particles = [
            Particle(self.n_dimensions, self.bounds) # 创建单个粒子
            for _ in range(self.n_particles) # 循环n_particles次
        ]

        # 初始化全局最优解
        for particle in self.particles:
            # 计算当前粒子的适应度
            fitness = self.objective_func(particle.position)
            particle.best_fitness = fitness
            # 更新全局最优解
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

        #记录每次迭代的最优位置
        self.history.append(self.global_best_position)

    def optimize(self):
        """
        执行优化
        :return: 最优位置
        """
        for iteration in range(self.max_iter):
            self.update_particles()
        return self.global_best_position, self.global_best_fitness

    def update_particles(self):
        """
        更新粒子群
        """
        for particle in self.particles:
            # 位置和速度
            self.update_particle(particle)
            #计算适应度
            current_fitness = self.objective_func(particle.position)
            #更新个体最优
            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()
                #更新全局最优
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = particle.position.copy()

        # 记录本次迭代的最佳适应度
        self.history.append(self.global_best_position)

    def update_particle(self, particle):
        """
        更新单个粒子的位置和速度
        :param particle: 要更新的粒子对象
        """
        r1, r2 = random.random(), random.random()

        for dim in range(self.n_dimensions):
            min_bound, max_bound = self.bounds[dim] # 当前维度的边界，因为边界可能会更新

            #惯性权重
            inertia = self.w * particle.velocity[dim]
            #个体权重
            cognitive = self.c1 * r1 * (particle.best_position[dim] - particle.position[dim])
            #社会权重
            social = self.c2 * r2 * (self.global_best_position[dim] - particle.position[dim])

            #更新速度
            new_velocity = inertia + cognitive + social
            particle.velocity[dim] = new_velocity

            #更新位置
            new_position = particle.position[dim] + new_velocity

            #处理边界，此处只是粗略处理，后面可以改为反弹机制
            if new_position < min_bound:
                new_position = min_bound
            elif new_position > max_bound:
                new_position = max_bound

            particle.position[dim] = new_position

class Function:
    """
    目标函数集合
    """
    @staticmethod
    def griewank(x):
        sum_part = sum(x_i**2 for x_i in x)/4000.0
        prod_part = 1.0
        for i, x_i in enumerate(x):
            prod_part *= math.cos(x_i/math.sqrt(i+1))
        return 1.0 + sum_part - prod_part

    @staticmethod
    def rastrigin(x):
        return 10*len(x) + sum(x_i*x_i - 10*math.cos(2*math.pi*x_i) for x_i in x)

    @staticmethod
    def get_func(name):
        if name == 'griewank':
            return Function.griewank
        elif name == 'rastrigin':
            return Function.rastrigin
        else:
            return None


if __name__ == '__main__':
    """
    主函数
    """

    griewank_func = Function.get_func('griewank')
    rastrigin_func = Function.get_func('rastrigin')

    # 优化griewank函数
    pso_griewank = PSO(griewank_func,30,2,None,0.7,1.49,1.49,300)
    g_best_p = pso_griewank.optimize()[0]
    g_best_f = pso_griewank.optimize()[1]
    print("Griewank函数的最小值：")
    print("坐标", g_best_p)
    print("函数值", g_best_f)

    # 优化rastrigin函数
    pso_rastrigin = PSO(rastrigin_func,30,2,None,0.7,1.49,1.49,300)

    g_best_p = pso_rastrigin.optimize()[0]
    g_best_f = pso_rastrigin.optimize()[1]
    print("Rastrigin函数的最小值：")
    print("坐标", g_best_p)
    print("函数值", g_best_f)
