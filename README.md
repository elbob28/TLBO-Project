# TLBO-Project

Teaching-Learning-Based Optimization (TLBO) Algorithm Implementation in Python

## Description

This project implements the Teaching-Learning-Based Optimization (TLBO) algorithm, a population-based metaheuristic optimization technique inspired by the teaching-learning process in classrooms. The algorithm is designed to solve complex engineering optimization problems.

### Features

- Pure Python implementation
- Easy to use API
- Visualization of convergence curves
- Configurable population size and iterations
- Support for custom objective functions and constraints

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TLBO-Project.git
cd TLBO-Project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0

## Usage

### Basic Example

```python
from tlbo import TLBO

# Define your objective function
def objective_function(x):
    return sum(x**2)  # Sphere function

# Define problem parameters
num_variables = 2
bounds = [(-5, 5), (-5, 5)]  # Search space bounds

# Initialize and run TLBO
tlbo = TLBO(
    objective_func=objective_function,
    num_variables=num_variables,
    bounds=bounds,
    population_size=30,
    max_iterations=100
)

best_solution, best_fitness, convergence_curve = tlbo.optimize()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### Running the Example

To run the provided example:

```bash
python main.py
```

This will optimize the sphere function and display the results along with convergence plots.

## Algorithm Overview

The TLBO algorithm consists of two main phases:

1. **Teacher Phase**: The best solution (teacher) guides the population towards better solutions.
2. **Learner Phase**: Learners interact with each other to improve their knowledge.

The algorithm iteratively applies these phases until convergence or maximum iterations are reached.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```
Teaching-Learning-Based Optimization Algorithm Implementation
```