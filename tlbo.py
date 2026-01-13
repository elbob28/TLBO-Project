import numpy as np
from typing import Tuple, List, Callable

"""
Teaching-Learning-Based Optimization (TLBO) Algorithm
Solves engineering optimization problems using TLBO metaheuristic
"""

import matplotlib.pyplot as plt


class TLBO:
    """Teaching-Learning-Based Optimization Algorithm"""

    def __init__(
        self,
        objective_func: Callable,
        num_variables: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 30,
        max_iterations: int = 100,
    ):
        """
        Initialize TLBO optimizer.

        Args:
            objective_func: Function to minimize
            num_variables: Number of decision variables
            bounds: List of (min, max) tuples for each variable
            population_size: Number of learners (population)
            max_iterations: Maximum number of iterations
        """
        self.objective_func = objective_func
        self.num_variables = num_variables
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations

        # Initialize population and fitness
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

    def _initialize_population(self) -> None:
        """Initialize population randomly within bounds"""
        self.population = np.zeros((self.population_size, self.num_variables))
        for i in range(self.num_variables):
            min_val, max_val = self.bounds[i]
            self.population[:, i] = np.random.uniform(min_val, max_val, self.population_size)

    def _evaluate_fitness(self) -> None:
        """Evaluate fitness for all population members"""
        self.fitness = np.array([self.objective_func(individual) for individual in self.population])

        # Track best solution
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def _teacher_phase(self, iteration: int) -> None:
        """
        Teacher Phase: Best solution (teacher) guides the population
        """
        teacher = self.population[np.argmin(self.fitness)]

        # Mean of population
        mean_population = np.mean(self.population, axis=0)

        # Teaching factor (decreases with iterations)
        tf = np.random.rand(self.num_variables)

        # Update population based on teacher
        new_population = self.population.copy()
        for i in range(self.population_size):
            new_population[i] = self.population[i] + tf * (teacher - mean_population)

        # Apply bounds
        self._apply_bounds(new_population)
        self.population = new_population

    def _learner_phase(self) -> None:
        """
        Learner Phase: Learners interact with random peers to improve
        """
        new_population = self.population.copy()

        for i in range(self.population_size):
            # Select random learner
            j = np.random.randint(0, self.population_size)

            if self.fitness[j] < self.fitness[i]:
                # Learner i learns from learner j
                new_population[i] = self.population[i] + np.random.rand(self.num_variables) * (
                    self.population[j] - self.population[i]
                )
            else:
                # Learner i learns in opposite direction
                new_population[i] = self.population[i] + np.random.rand(self.num_variables) * (
                    self.population[i] - self.population[j]
                )

        # Apply bounds
        self._apply_bounds(new_population)
        self.population = new_population

    def _apply_bounds(self, population: np.ndarray) -> None:
        """Ensure population stays within bounds"""
        for i in range(self.num_variables):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.clip(population[:, i], min_val, max_val)

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run TLBO optimization.

        Returns:
            best_solution: Optimal solution found
            best_fitness: Fitness value of best solution
            convergence_curve: Fitness values over iterations
        """
        print("Initializing population...")
        self._initialize_population()
        self._evaluate_fitness()

        print("Starting optimization...")
        for iteration in range(self.max_iterations):
            # Teacher and Learner phases
            self._teacher_phase(iteration)
            self._evaluate_fitness()

            self._learner_phase()
            self._evaluate_fitness()

            # Track convergence
            self.convergence_curve.append(self.best_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations} - Best Fitness: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.convergence_curve


def objective_function(x: np.ndarray) -> float:
    """
    Example objective function: Sphere function
    Minimize f(x, y) = x^2 + y^2
    Global minimum at (0, 0) with value 0
    """
    return np.sum(x**2)


def visualize_results(convergence_curve: List[float], best_solution: np.ndarray, best_fitness: float) -> None:
    """Plot convergence curve"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence curve
    axes[0].plot(convergence_curve, linewidth=2, color='blue')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Fitness Value')
    axes[0].set_title('TLBO Convergence Curve')
    axes[0].grid(True, alpha=0.3)

    # Log scale for better visualization
    axes[1].semilogy(convergence_curve, linewidth=2, color='red')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Fitness Value (log scale)')
    axes[1].set_title('TLBO Convergence (Log Scale)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
    print("\nConvergence plot saved as 'convergence_plot.png'")
    plt.show()