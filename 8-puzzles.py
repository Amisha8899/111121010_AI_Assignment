# ----------------------------------------
# 8-Puzzle Problem Solved Using Simulated Annealing and Genetic Algorithm
# ----------------------------------------

# Description:
# The 8-puzzle problem is a sliding puzzle where you need to arrange numbers in a 3x3 grid
# in a specific order. The goal is to solve the puzzle using two heuristic algorithms:
# 1. Simulated Annealing (SA)
# 2. Genetic Algorithm (GA)

# Shared helper functions for both algorithms are provided first, followed by the
# individual implementations of Simulated Annealing and Genetic Algorithm.

import random
import math

# ----------------------------------------
# Helper Functions for the 8-Puzzle Problem
# ----------------------------------------

# A helper function to check if the puzzle is solvable
# The number of inversions (pair of tiles that are reversed from the goal) must be even for the puzzle to be solvable
def is_solvable(state):
    one_d_state = sum(state, [])
    inversions = 0
    for i in range(len(one_d_state)):
        for j in range(i + 1, len(one_d_state)):
            if one_d_state[i] > one_d_state[j] != 0:
                inversions += 1
    return inversions % 2 == 0

# Manhattan distance heuristic - calculates how far each tile is from its goal position
def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:  # Ignore the blank space
                goal_x, goal_y = divmod(goal_state.index(value), 3)
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

# Get neighbors: Generate all possible states by moving the blank tile (0) in the puzzle
def get_neighbors(state):
    neighbors = []
    zero_pos = [(row_i, col_i) for row_i, row in enumerate(state) for col_i, num in enumerate(row) if num == 0][0]
    row, col = zero_pos
    moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]  # Up, down, left, right
    
    for r, c in moves:
        if 0 <= r < 3 and 0 <= c < 3:
            new_state = [list(row) for row in state]  # Copy current state
            new_state[row][col], new_state[r][c] = new_state[r][c], new_state[row][col]  # Swap blank space with neighbor
            neighbors.append(new_state)
    
    return neighbors

# Flatten the 2D list to 1D (required for Genetic Algorithm)
def flatten(state):
    return sum(state, [])

# Convert a flattened 1D list to a 3x3 matrix
def unflatten(flat_state):
    return [flat_state[i:i+3] for i in range(0, len(flat_state), 3)]

# ----------------------------------------
# Simulated Annealing (SA) Algorithm
# ----------------------------------------

# Pseudocode for Simulated Annealing:
# 1. Start with an initial configuration of the puzzle.
# 2. Set an initial temperature and define a cooling schedule.
# 3. Randomly select a neighboring configuration (valid move).
# 4. If the new configuration is better (lower cost), accept it.
# 5. If the new configuration is worse, accept it with a probability based on the temperature.
# 6. Gradually reduce the temperature.
# 7. Stop when the temperature is very low or the goal is reached.

def simulated_annealing(initial_state, goal_state, initial_temperature=100, cooling_rate=0.99, min_temperature=0.1):
    current_state = initial_state
    T = initial_temperature  # Initial temperature
    
    # Define a cost function (using Manhattan distance)
    def cost(state):
        return manhattan_distance(state, flatten(goal_state))
    
    while T > min_temperature:
        # Check if current state is the goal state
        if current_state == goal_state:
            return current_state  # Solution found
        
        # Get a random neighbor state (random valid move)
        next_state = random.choice(get_neighbors(current_state))
        delta_E = cost(next_state) - cost(current_state)
        
        if delta_E < 0:
            # Accept the better state
            current_state = next_state
        else:
            # Accept the worse state with a probability that decreases with temperature
            acceptance_prob = math.exp(-delta_E / T)
            if random.random() < acceptance_prob:
                current_state = next_state
        
        # Cool down (reduce temperature)
        T *= cooling_rate
    
    return current_state  # Return the best state found if no exact solution was found

# ----------------------------------------
# Genetic Algorithm (GA) for 8-puzzle
# ----------------------------------------

# Pseudocode for Genetic Algorithm:
# 1. Initialize a population of random puzzle configurations (chromosomes).
# 2. Evaluate the fitness of each individual (how close to the goal state).
# 3. Select parents from the population based on their fitness.
# 4. Use crossover (combine parts of parents) to generate offspring.
# 5. Apply random mutations to some offspring to maintain diversity.
# 6. Replace the old population with new offspring.
# 7. Repeat for a number of generations or until a solution is found.

def genetic_algorithm(goal_state, population_size=100, generations=1000, mutation_rate=0.05):
    # Generate a random initial population
    def random_state():
        state = random.sample(range(9), 9)
        return unflatten(state)
    
    population = [flatten(random_state()) for _ in range(population_size)]
    
    # Fitness function based on Manhattan distance
    def fitness(individual):
        return manhattan_distance(unflatten(individual), flatten(goal_state))
    
    # Selection: Choose parents based on their fitness
    def selection(population):
        sorted_population = sorted(population, key=lambda ind: fitness(ind))
        return sorted_population[:population_size//2]
    
    # Crossover: Combine two parents to create two children
    def crossover(parent1, parent2):
        cross_point = random.randint(1, len(parent1) - 2)
        return parent1[:cross_point] + parent2[cross_point:], parent2[:cross_point] + parent1[cross_point:]
    
    # Mutation: Randomly swap two tiles in the puzzle
    def mutate(individual):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    
    for generation in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [fitness(ind) for ind in population]
        
        # Check if any individual is the goal state (fitness = 0)
        if min(fitness_scores) == 0:
            return population[fitness_scores.index(min(fitness_scores))]
        
        # Selection: Keep the top 50% of the population
        selected_population = selection(population)
        
        # Crossover to create new individuals
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        
        # Apply mutation
        for individual in offspring:
            if random.random() < mutation_rate:
                mutate(individual)
        
        # Replace the old population with new offspring
        population = offspring
    
    return population[0]  # Return the best found individual if no solution found

# ----------------------------------------
# Main Program to Run the 8-Puzzle Solver
# ----------------------------------------

if __name__ == "__main__":
    # Example initial state (can be changed)
    initial_state = [
        [7, 2, 4],
        [5, 0, 6],
        [8, 3, 1]
    ]
    
    # Goal state for the puzzle
    goal_state = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    # Check if the puzzle is solvable
    if not is_solvable(initial_state):
        print("This puzzle configuration is not solvable.")
    else:
        # Solve with Simulated Annealing
        print("Solving with Simulated Annealing...")
        sa_solution = simulated_annealing(initial_state, goal_state)
        print("Simulated Annealing Solution:")
        for row in sa_solution:
            print(row)

        # Solve with Genetic Algorithm
        print("\nSolving with Genetic Algorithm...")
        ga_solution = genetic_algorithm(goal_state)
        print("Genetic Algorithm Solution:")
        for row in unflatten(ga_solution):
            print(row)

# Explanation of the Code:

#     Helper Functions:
#         These functions provide necessary utilities like checking if the puzzle is solvable, calculating the Manhattan distance, generating neighbor states, and converting between 1D and 2D puzzle representations.

#     Simulated Annealing (SA):
#         The algorithm probabilistically explores states by adjusting a temperature parameter that controls the acceptance of worse solutions. The cost function used is the Manhattan distance between the current state and the goal state.

#     Genetic Algorithm (GA):
#         The algorithm evolves a population of solutions using crossover, mutation, and selection based on fitness (Manhattan distance). The population evolves over multiple generations to find a solution.

#     Main Program:
#         It allows the user to run both algorithms on a specific initial state and goal state. The results from both the Simulated Annealing and Genetic Algorithm are printed.

# How to Use:

#     Run the program. It will first attempt to solve the 8-puzzle using Simulated Annealing and then using the Genetic Algorithm.
#     You can modify the initial state and goal state in the if __name__ == "__main__" block.