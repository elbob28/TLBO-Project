from tlbo import TLBO, objective_function, visualize_results


def main():
    """Main function to run TLBO optimization"""
    print("=" * 60)
    print("Teaching-Learning-Based Optimization (TLBO) Algorithm")
    print("=" * 60)

    # Problem setup
    num_variables = 2
    bounds = [(-5, 5), (-5, 5)]  # Search space

    # Initialize TLBO
    tlbo = TLBO(
        objective_func=objective_function,
        num_variables=num_variables,
        bounds=bounds,
        population_size=30,
        max_iterations=100
    )

    # Run optimization
    best_solution, best_fitness, convergence_curve = tlbo.optimize()

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best Solution Found: {best_solution}")
    print(f"Best Fitness Value: {best_fitness:.10f}")
    print(f"Expected Optimal: [0, 0]")
    print(f"Expected Fitness: 0.0")
    print("=" * 60)

    # Visualize
    visualize_results(convergence_curve, best_solution, best_fitness)


if __name__ == "__main__":
    main()