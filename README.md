# genetic-programming-tutorial
A genetic programming implementation with detailed tutorials for each component

# Genetic Programming Tutorial

This project demonstrates a genetic programming implementation to approximate a target function. The code includes explanations and examples for each part.

---

## **1. Project Setup**
To run this code, install the required libraries:
```bash
pip install matplotlib numpy

# Configuration
POP_SIZE = 300
MAX_DEPTH = 7
GENERATIONS = 100
MUTATION_RATE = 0.3
NUM_CONSTANTS = 10
ELITE_SIZE = 5

def target_function(x):
    c1, c2 = 0, 5
    if x < c1:
        return x**2
    elif c1 <= x <= c2:
        return -x - 7
    else:
        return math.log(x + 1) if x > 0 else 0

FUNCTION_SET = [
    {'name': 'add', 'arity': 2, 'func': lambda a, b: a + b},
    {'name': 'sub', 'arity': 2, 'func': lambda a, b: a - b},
    ...
]

TERMINALS = ['x'] + [round(random.uniform(-10, 10), 2) for _ in range(NUM_CONSTANTS)]

class Node:
    def __init__(self, is_function, value, children=None):
        self.is_function = is_function
        self.value = value
        self.children = children if children else []
def generate_random_tree(max_depth, function_set, terminals):
    if max_depth == 0:
        return Node(is_function=False, value=random.choice(terminals))
    ...
def fitness(individual, train_x, train_y, alpha=0.001):
    mse = 0
    for x, y in zip(train_x, train_y):
        pred = evaluate_tree(individual, x)
        mse += (pred - y) ** 2
    mse /= len(train_x)
    return mse + alpha * tree_size(individual)
def tournament_selection(population, fitnesses, k=7):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    ...
def mutation(individual, function_set, terminals, max_depth):
    ...
for gen in range(GENERATIONS):
    fitnesses = [fitness(ind, TRAIN_X, TRAIN_Y) for ind in population]
    ...

plt.plot(range(GENERATIONS), generation_best_fitness, ...)



