import random

# 城市距离矩阵
distance = [
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 3, 4],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 7, 0, 6, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6],
    [5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 8, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 5, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 3, 0, 0],
    [4, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 1, 2, 0, 4, 4, 0, 0, 0, 0, 0],
    [0, 7, 0, 2, 0, 0, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
    [3, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 6, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


class AntColony:
    def __init__(self, distance_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, rho=0.5, q=100):
        """
        初始化蚁群算法
        :param distance_matrix: 距离矩阵
        :param n_ants: 蚂蚁数量
        :param n_iterations: 迭代次数
        :param alpha: 信息素重要程度
        :param beta: 启发式信息重要程度
        :param rho: 信息素挥发系数
        :param q: 信息素强度
        """
        self.distances = distance_matrix
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        self.n_cities = len(distance_matrix) # 城市数量
        self.pheromone = [[1.0 for _ in range(self.n_cities)] for _ in range(self.n_cities)] # 信息素矩阵

        # 获取所有有效城市（排除城市5）
        self.valid_cities = [i for i in range(self.n_cities) if any(distance_matrix[i])]

    def get_possible_next_cities(self, current_city, visited):
        """获取当前城市可以前往的下一个城市"""
        possible = []
        # 城市在有效城市列表中，且未访问过，且距离不为0，则加入候选列表
        for next_city in self.valid_cities:
            if (next_city not in visited and
                    self.distances[current_city][next_city] > 0):
                possible.append(next_city)
        return possible

    def calculate_probabilities(self, current_city, possible_cities):
        """计算选择各个城市的概率"""
        probabilities = []
        total = 0.0

        for next_city in possible_cities:
            # 计算信息素和启发式信息
            pheromone = self.pheromone[current_city][next_city] ** self.alpha
            heuristic = (1.0 / self.distances[current_city][next_city]) ** self.beta
            probability = pheromone * heuristic
            probabilities.append(probability)
            total += probability

        # 归一化概率
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # 如果所有概率都为0，平均分配
            probabilities = [1.0 / len(possible_cities)] * len(possible_cities)

        return probabilities

    def select_next_city(self, probabilities, possible_cities):
        """根据概率选择下一个城市"""
        rand = random.random()
        cumulative_prob = 0.0

        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return possible_cities[i]

        return possible_cities[-1]

    def construct_solution(self):
        """构建一个完整的路径解"""
        # 随机选择起点
        start_city = random.choice(self.valid_cities)
        path = [start_city]
        visited = set([start_city])
        total_distance = 0

        current_city = start_city

        # 构建完整路径
        while len(visited) < len(self.valid_cities):
            possible_cities = self.get_possible_next_cities(current_city, visited)

            if not possible_cities:
                break

            probabilities = self.calculate_probabilities(current_city, possible_cities)
            next_city = self.select_next_city(probabilities, possible_cities)

            total_distance += self.distances[current_city][next_city]
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city

        # 回到起点（如果可能）
        if self.distances[current_city][start_city] > 0:
            total_distance += self.distances[current_city][start_city]
            path.append(start_city)

        return path, total_distance

    def update_pheromone(self, solutions):
        """更新信息素"""
        # 信息素挥发
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.pheromone[i][j] *= (1.0 - self.rho)

        # 增加新的信息素
        for path, distance in solutions:
            if distance > 0:  # 避免除零错误
                pheromone_to_add = self.q / distance
                for i in range(len(path) - 1):
                    city_i = path[i]
                    city_j = path[i + 1]
                    self.pheromone[city_i][city_j] += pheromone_to_add
                    self.pheromone[city_j][city_i] += pheromone_to_add  # 对称更新

    def solve(self):
        """运行蚁群算法求解"""
        best_path = None
        best_distance = float('inf')

        print("开始蚁群算法求解...")
        print(f"有效城市数量: {len(self.valid_cities)}")
        print(f"城市列表: {[city + 1 for city in self.valid_cities]}")

        for iteration in range(self.n_iterations):
            solutions = []

            # 所有蚂蚁构建解
            for ant in range(self.n_ants):
                path, distance = self.construct_solution()
                solutions.append((path, distance))

                # 更新最优解
                if distance < best_distance and len(path) == len(self.valid_cities) + 1:
                    best_distance = distance
                    best_path = path

            # 更新信息素
            self.update_pheromone(solutions)

            if iteration % 20 == 0:
                print(f"迭代 {iteration}: 当前最优距离 = {best_distance}")

        return best_path, best_distance


def main():
    # 运行蚁群算法
    colony = AntColony(
        distance_matrix=distance,
        n_ants=30,
        n_iterations=100,
        alpha=1.0,
        beta=3.0,
        rho=0.4,
        q=100
    )

    best_path, best_distance = colony.solve()

    print("\n" + "=" * 50)
    print("最终结果:")
    print(f"最优路径: {[city + 1 for city in best_path]}")
    print(f"最短距离: {best_distance}")
    print(f"访问城市数量: {len(best_path) - 1}")



if __name__ == "__main__":
    main()