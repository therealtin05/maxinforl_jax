import os
import numpy as np
import matplotlib.pyplot as plt

def load_folder(folder):
    all_steps = []
    all_fitness = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            # Load data from text file
            data = np.loadtxt(os.path.join(folder, fname))
            # If shape is (N, 2), assume columns: steps, fitness
            if data.shape[1] == 2:
                steps, fitness = data[:, 0], data[:, 1]
            else:
                raise ValueError(f"Unexpected data shape in {fname}: {data.shape}")
            all_steps.append(steps)
            all_fitness.append(fitness)
    # Align by steps (assume all runs have the same steps)
    steps = all_steps[0]
    fitness_arr = np.stack(all_fitness)
    return steps, fitness_arr

folder1 = 'logs/walker-run/maxinfosac'
folder2 = 'logs/walker-run/sac'

steps1, fitness1 = load_folder(folder1)
steps2, fitness2 = load_folder(folder2)

mean1 = fitness1.mean(axis=0)
std1 = fitness1.std(axis=0)
mean2 = fitness2.mean(axis=0)
std2 = fitness2.std(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(steps1, mean1, label='maxinfosac', color='blue')
plt.fill_between(steps1, mean1-std1, mean1+std1, color='blue', alpha=0.1)
plt.plot(steps2, mean2, label='sac', color='red')
plt.fill_between(steps2, mean2-std2, mean2+std2, color='red', alpha=0.1)
plt.xlabel('Steps')
plt.ylabel('Fitness')
plt.title('Comparison of Fitness Over Steps on Walker-run')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()