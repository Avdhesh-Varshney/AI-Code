def steepest_ascent_hill_climbing(problem):
  
    current = problem.initial_state()
  
    while True:
        neighbors = problem.get_all_neighbors(current)
        if not neighbors:
            break  # No neighbors found
          
        next_state = max(neighbors, key=problem.evaluate)
        if problem.evaluate(next_state) <= problem.evaluate(current):
            break  # No better neighbor found
          
        current = next_state
    return current
