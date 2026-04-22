import random


# 节点距离矩阵
distance = [
    [0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 3],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 7, 0, 6],
    [0, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 4],
    [4, 0, 6, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 8, 2, 0, 2, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 5, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 3, 0],
    [4, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 1, 2, 0, 4, 4, 0, 0, 0, 0],
    [0, 7, 0, 2, 0, 0, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
    [3, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


class GeneticAlgorithm:
    def __init__(self, population_size, generations, distance_matrix, path_length=10):
        """
        初始化遗传算法
        population_size: 种群大小
        generations: 迭代次数
        distance_matrix: 城市距离矩阵
        path_length: 路径长度（节点数量）
        population: 种群
        best_individual: 最优个体
        best_fitness: 最优适应度
        """
        self.population_size = population_size
        self.generations = generations
        self.distance_matrix = distance_matrix
        self.path_length = path_length
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')

    def _get_feasible_neighbors(self, current_node, visited):
        """
        获取当前节点的可行邻居节点
        current_node: 当前节点
        visited: 已访问节点集合
        """
        neighbors = []
        for neighbor in range(len(self.distance_matrix)):
            if (self.distance_matrix[current_node][neighbor] > 0 and
                    neighbor != current_node and
                    neighbor not in visited):
                neighbors.append(neighbor)
        return neighbors

    def _generate_feasible_path(self):
        """
        从随机起点开始，使用贪心策略生成可行路径
        """
        start_node = random.randint(0, len(self.distance_matrix) - 1)
        path = [start_node]
        visited = set([start_node])

        while len(path) < self.path_length:
            current_node = path[-1]
            neighbors = self._get_feasible_neighbors(current_node, visited)

            if not neighbors:
                break

            # 贪心选择：选择距离最近的邻居
            next_node = min(neighbors, key=lambda n: self.distance_matrix[current_node][n])
            path.append(next_node)
            visited.add(next_node)

        return path if len(path) == self.path_length else None

    def _generate_random_feasible_path(self):
        """
        备用方法：随机生成可行路径（当贪心法失败时使用）
        """
        max_attempts = 100

        for _ in range(max_attempts):
            path = []
            visited = set()
            current_node = random.randint(0, len(self.distance_matrix) - 1)
            path.append(current_node)
            visited.add(current_node)

            while len(path) < self.path_length:
                neighbors = self._get_feasible_neighbors(path[-1], visited)
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                path.append(next_node)
                visited.add(next_node)

            if len(path) == self.path_length:
                return path

        return None

    def generate_population(self):
        """
        贪心算法生成初始种群 - 保证路径可行性
        策略：从随机起点开始，每次选择最近的未访问节点
        """
        self.population = []

        for _ in range(self.population_size):
            individual = self._generate_feasible_path()
            if individual and len(individual) == self.path_length:
                self.population.append(individual)
            else:
                individual = self._generate_random_feasible_path()
                if individual:
                    self.population.append(individual)

        return self.population

    def calculate_fitness(self, individual):
        """
        计算个体适应度 - 路径总距离
        注意：处理不可行连接（距离为0）
        """
        total_distance = 0
        for i in range(len(individual) - 1):
            from_node = individual[i]
            to_node = individual[i + 1]
            dist = self.distance_matrix[from_node][to_node]
            if dist == 0:
                # 不可行连接，返回极大惩罚值
                return float('inf')
            total_distance += dist
        return total_distance

    def select(self, tournament_size=3):
        """
        选择操作 - 锦标赛选择
        tournament_size: 锦标赛规模
        """
        selected = []
        for _ in range(self.population_size):
            # 随机选择 tournament_size 个个体进行锦标赛
            tournament = random.sample(list(zip(self.population,
                                                [self.calculate_fitness(ind) for ind in self.population])),
                                       tournament_size)
            # 选择适应度最好的个体（距离最小的）
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        """
        顺序交叉(OX) - 保持路径可行性
        parent1, parent2: 父代个体
        """
        size = len(parent1)
        child = [-1] * size

        # 选择两个交叉点
        start, end = sorted(random.sample(range(size), 2))

        # 从parent1复制片段
        child[start:end + 1] = parent1[start:end + 1]

        # 从parent2填充剩余位置
        parent2_pos = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[parent2_pos] in child:
                    parent2_pos += 1
                    if parent2_pos >= size:
                        parent2_pos = 0
                child[i] = parent2[parent2_pos]
                parent2_pos += 1

        return child

    def mutate(self, individual, mutation_rate=0.1):
        """
        变异操作 - 交换变异
        individual: 要变异的个体
        mutation_rate: 变异概率
        """
        if random.random() < mutation_rate:
            # 随机选择两个不同的位置进行交换
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def repair_individual(self, individual):
        """
        修复不可行个体 - 确保路径连续可通行
        """
        # 检查并修复不可行连接
        for i in range(len(individual) - 1):
            if self.distance_matrix[individual[i]][individual[i + 1]] == 0:
                # 找到可行的替代节点
                visited = set(individual[:i + 1])
                neighbors = self._get_feasible_neighbors(individual[i], visited)
                if neighbors:
                    # 随机选择一个可行邻居替换
                    individual[i + 1] = random.choice(neighbors)
                else:
                    # 如果找不到可行邻居，这个个体无法修复
                    return None
        return individual

    def _apply_elitism(self, new_population):
        """
        精英保留策略 - 保留当代最优个体到下一代
        new_population: 新种群
        """
        if self.best_individual is not None:
            # 用最优个体替换新种群中最差的个体
            worst_index = max(range(len(new_population)),
                              key=lambda i: self.calculate_fitness(new_population[i]))
            new_population[worst_index] = self.best_individual
        return new_population

    def _is_valid_path(self, individual):
        """
        验证路径可行性
        """
        if len(individual) != self.path_length:
            return False

        # 检查是否有重复节点
        if len(set(individual)) != self.path_length:
            return False

        # 检查所有连接是否可行
        for i in range(len(individual) - 1):
            if self.distance_matrix[individual[i]][individual[i + 1]] == 0:
                return False
        return True

    def run(self):
        """
        运行遗传算法主流程
        """
        # 1. 初始化种群
        print("生成初始种群...")
        self.generate_population()

        # 2. 评估初始种群
        fitnesses = [self.calculate_fitness(ind) for ind in self.population]
        current_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])

        if fitnesses[current_best_idx] < self.best_fitness:
            self.best_fitness = fitnesses[current_best_idx]
            self.best_individual = self.population[current_best_idx][:]

        print(f"初始最优距离: {self.best_fitness}")

        # 3. 迭代进化
        for generation in range(self.generations):
            # 3.1 选择
            selected_population = self.select(tournament_size=3)

            # 3.2 交叉
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % self.population_size]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.extend([child1, child2])

            # 确保种群大小不变
            new_population = new_population[:self.population_size]

            # 3.3 变异
            for i in range(len(new_population)):
                new_population[i] = self.mutate(new_population[i], mutation_rate=0.1)

            # 3.4 修复
            repaired_population = []
            for individual in new_population:
                repaired = self.repair_individual(individual)
                if repaired and self._is_valid_path(repaired):
                    repaired_population.append(repaired)
                else:
                    # 如果无法修复，使用随机生成的新个体
                    new_ind = self._generate_random_feasible_path()
                    if new_ind:
                        repaired_population.append(new_ind)

            self.population = repaired_population

            # 3.5 评估
            fitnesses = [self.calculate_fitness(ind) for ind in self.population]
            current_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])

            # 更新全局最优
            if fitnesses[current_best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[current_best_idx]
                self.best_individual = self.population[current_best_idx][:]
                print(f"第{generation + 1}代发现新最优: {self.best_fitness}")

            # 3.6 精英保留
            self.population = self._apply_elitism(self.population)

        # 4. 返回最优解
        print(f"\n最终最优距离: {self.best_fitness}")
        return self.best_individual, self.best_fitness


def adjust_node_numbers(path):
    """
    将路径中的节点编号加1
    """
    return [node + 1 for node in path]


def main():
    """
    主函数
    """
    ga = GeneticAlgorithm(
        population_size=50,
        generations=100,
        distance_matrix=distance,
        path_length=10
    )
    best_path, best_distance = ga.run()

    # 调整节点编号（加1）
    adjusted_path = adjust_node_numbers(best_path)

    print('最优路径：', adjusted_path)
    print('最优距离：', best_distance)

    # 验证路径（使用调整后的节点编号）
    print("\n路径验证:")
    for i in range(len(adjusted_path) - 1):
        from_node = adjusted_path[i] - 1  # 减1以访问原始矩阵
        to_node = adjusted_path[i + 1] - 1  # 减1以访问原始矩阵
        dist = distance[from_node][to_node]
        print(f"节点 {adjusted_path[i]} → 节点 {adjusted_path[i + 1]}: 距离 = {dist}")


if __name__ == '__main__':
    main()