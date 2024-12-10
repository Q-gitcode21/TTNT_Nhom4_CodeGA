import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ma trận khoảng cách
distance_matrix = [
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
]

# Tọa độ thành phố để vẽ đồ thị
city_coordinates = [(0, 0), (2, 0), (1, 2), (3, 3)]

# Hàm tính quãng đường của một lộ trình
def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    distance += distance_matrix[route[-1]][route[0]]  # Quay về thành phố ban đầu
    return distance

# Khởi tạo quần thể ngẫu nhiên
def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

# Tính toán độ thích nghi (fitness) của từng cá thể
def evaluate_population(population):
    fitness_scores = []
    for route in population:
        fitness_scores.append(1 / calculate_distance(route))  # Fitness càng lớn càng tốt
    return fitness_scores

# Lựa chọn dựa trên vòng quay roulette
def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

# Lai ghép thứ tự (Order Crossover - OX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size

    # Sao chép đoạn giữa từ cha
    child[start:end + 1] = parent1[start:end + 1]

    # Điền các thành phố còn lại từ mẹ
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city

    return child

# Đột biến: Hoán đổi hai thành phố ngẫu nhiên
def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Hàm để vẽ đồ thị
def plot_route(route, title, generation):
    plt.figure(figsize=(6, 6))
    for i in range(len(route)):
        city1 = city_coordinates[route[i]]
        city2 = city_coordinates[route[(i + 1) % len(route)]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'bo-', lw=2)

    for idx, coord in enumerate(city_coordinates):
        plt.text(coord[0], coord[1], f"City {idx}", fontsize=10, ha='right')

    plt.title(f"{title} (Generation: {generation})")
    plt.grid(True)
    plt.show()

# Hàm chạy thuật toán GA và tạo hoạt ảnh
def genetic_algorithm(num_generations, pop_size, num_cities, mutation_rate):
    population = initialize_population(pop_size, num_cities)
    all_routes = []
    all_distances = []

    for generation in range(num_generations):
        fitness = evaluate_population(population)
        new_population = []

        # Tạo thế hệ mới
        for _ in range(pop_size // 2):
            parents = select_parents(population, fitness)
            child1 = mutate(crossover(parents[0], parents[1]), mutation_rate)
            child2 = mutate(crossover(parents[1], parents[0]), mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
        fitness = evaluate_population(population)
        best_route = population[np.argmax(fitness)]
        best_distance = calculate_distance(best_route)

        all_routes.append(best_route)
        all_distances.append(best_distance)
        print(f"Thế hệ {generation + 1}, Lộ trình tốt nhất: {best_route}, Quãng đường: {best_distance}")

    # Hoạt ảnh
    def update(frame):
        plt.cla()
        best_route = all_routes[frame]
        for i in range(len(best_route)):
            city1 = city_coordinates[best_route[i]]
            city2 = city_coordinates[best_route[(i + 1) % len(best_route)]]
            plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'bo-', lw=2)
        plt.title(f"Generation {frame + 1}, Distance: {all_distances[frame]:.2f}")
        for idx, coord in enumerate(city_coordinates):
            plt.text(coord[0], coord[1], f"City {idx}", fontsize=10, ha='right')

    fig = plt.figure(figsize=(6, 6))
    ani = FuncAnimation(fig, update, frames=len(all_routes), repeat=False)
    plt.show()

    return best_route, best_distance

# Chạy GA với vẽ đồ thị và hoạt ảnh
best_route, best_distance = genetic_algorithm(num_generations=50, pop_size=6, num_cities=4, mutation_rate=0.1)
print("\nLộ trình tối ưu:", best_route, "Quãng đường:", best_distance)
