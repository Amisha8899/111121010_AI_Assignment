Explanation of the Code:

    Helper Functions:
        These functions provide necessary utilities like checking if the puzzle is solvable, calculating the Manhattan distance, generating neighbor states, and converting between 1D and 2D puzzle representations.

    Simulated Annealing (SA):
        The algorithm probabilistically explores states by adjusting a temperature parameter that controls the acceptance of worse solutions. The cost function used is the Manhattan distance between the current state and the goal state.

    Genetic Algorithm (GA):
        The algorithm evolves a population of solutions using crossover, mutation, and selection based on fitness (Manhattan distance). The population evolves over multiple generations to find a solution.

    Main Program:
        It allows the user to run both algorithms on a specific initial state and goal state. The results from both the Simulated Annealing and Genetic Algorithm are printed.

How to Use:

    Run the program. It will first attempt to solve the 8-puzzle using Simulated Annealing and then using the Genetic Algorithm.
    You can modify the initial state and goal state in the if __name__ == "__main__" block.
