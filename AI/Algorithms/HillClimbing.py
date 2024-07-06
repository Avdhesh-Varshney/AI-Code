def hill_climbing(f, x0):
    x = x0  # initial solution
  
    while True:
        neighbors = generate_neighbors(x)  # generate neighbors of x
      
        # find the neighbor with the highest function value
        best_neighbor = max(neighbors, key=f)
      
        if f(best_neighbor) <= f(x):  # if the best neighbor is not better than x, stop
            return x
        x = best_neighbor  # otherwise, continue with the best neighbor
