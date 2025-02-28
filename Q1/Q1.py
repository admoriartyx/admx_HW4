# This is the .py file for Question 1 of Problem Set 4

# Part a

# The problem asks for a Python script that tests the different r values given but also outputs the fixed
# points for the given quadratic. However, we already know the general form of the fixed points with 
# respect to r, as seen in the code. Noting this just in case my solution appears incomplete in that regard.

import numpy as np

def stability_test():
    r_vals = [1, 2, 3, 4]
    for r in r_vals:
        print(f'r={r}')
        for x0 in [0, (r-1)/r]: # these are the fixed points we are certain of
            print(f'Fixed point: {x0}')
            f_prime = r * (1 - 2 * x0) 
            stability = "Stable" if f_prime < 0 else "Unstable"
            print('Derivative f\'(x):', f_prime, '->', stability)

stability_test()

# The only stable fixed points correspond to the derivatives of r = 3, 4. The rest returned unstable.


# Part b

def iterations(r_vals, init_x, max_its=10000, tol=1e-6):
    results = {}
    for r in r_vals:
        x = init_x
        converged = False
        history = [x]
        
        for i in range(max_its):
            x_next = r * x * (1 - x) # recursion
            history.append(x_next)

            if abs(x_next - x) < tol:
                converged = True
                break
            
            x = x_next
        
        results[r] = {
            'Convergence': converged,
            'Iterations': i,
        }
    return results

r_vals = [2, 3, 3.5, 3.8, 4.0]
x0 = 0.2
iterations_output = iterations(r_vals, x0)

for r, data in iterations_output.items():
    print(f"r = {r}:")
    print(f"  Converged: {data['Convergence']} after {data['Iterations']} iterations")
    print()

# We observe convergence for r=2 and r=4. The former converged quickly after 5 iterations and the latter
# took longer to converge at 360 iterations.

# Part c

import matplotlib.pyplot as plt

def log_map_iterations(r, init_x, max_its=1000):
    x_vals = np.zeros(max_its)
    x_vals[0] = init_x
    for i in range(1, max_its):
        x_vals[i] = r * x_vals[i-1] * (1 - x_vals[i-1])
    return x_vals

r_vals = [2, 3, 3.5, 3.8, 4.0]
initial_conditions = [0.1, 0.3, 0.5]

plt.figure(figsize=(12, 10))
for i, r in enumerate(r_vals):
    plt.subplot(len(r_vals), 1, i + 1)
    for x0 in initial_conditions:
        x_vals = log_map_iterations(r, x0)
        plt.plot(x_vals, label=f'x0={x0}')
    plt.title(f'Logistic Map Time Series for r={r}')
    plt.xlabel('Iteration')
    plt.ylabel('x(n)')
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Q1_part_c.png')

# The plot shows that most of the maps with different initial conditions behave similarly, often just 
# a bit out of phase with one another. However, when the mapping becomes chaotic, it seems to be so for all
# given initial conditions.

# Part d

def logistic_map(r, x, its, last_k_its):
    results = []
    for i in range(its):
        x = r * x * (1 - x)
        if i >= its - last_k_its:
            results.append(x)
    return results

r_vals = np.linspace(0.1, 4, 8000)
its = 2000 
last_k_its = 200
x_vals = []

for i in r_vals:
    x0 = 0.2
    x_vals = logistic_map(i, x0, its, last_k_its)
    x_vals.extend([(i, x) for x in x_vals])

r_vals, x_vals = zip(*x_vals)
plt.figure(figsize=(12, 8))
plt.scatter(r_vals, x_vals, s=0.01, color='blue')
plt.title('Bifurcation Diagram of the Logistic Map')
plt.xlabel('Growth Rate r')
plt.ylabel('Population x')
plt.xlim(2, 4)
plt.ylim(0, 1)
plt.show()
plt.savefig('Q1_part_d.png')

# First bifurcation appears at around r=3
# Second around 3.4
# Chaotic section around r=3.57
# Temporary periodic behavior around r=3.83
# Total chaos by r=4


# Part e

gammas = np.linspace(0.5, 1.5, 100)
bi_points = []

for gamma in gammas:
	bi_start = False
	appended = False
	for r in np.linspace(1.00, 6, 250):
		x = 0.2
		trajectory = []
		for i in range(500):
			if i > 450:
				trajectory.append(round(x, decimals=6))
			x = r*x*(1-x**gamma)
		if len(np.unique(trajectory)) == 2:
			if bifurcation_start:
				bi_points.append(r)
				appended = True
				break
			bifurcation_start = True
		else:
			bifurcation_start = False
	if not appended:
		bi_points.append(np.nan)
plt.plot(gammas, bi_points)
plt.xlabel(r'$\gamma$')
plt.ylabel('Smallest bifurcation $r$')
plt.title('Smallest Bifurcation $r$ value against Gamma')
plt.show()
plt.savefig('Q1_part_3.png')