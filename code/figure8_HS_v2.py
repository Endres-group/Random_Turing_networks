import pandas as pd
import numpy as np
import os
from scipy.signal import argrelextrema
import time
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt

def gen_matrix_F(n, vr):
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)
    A = -I + B
    D = np.zeros((n, n))
    return A, D

def process_tuple(tuplet):
    p, Dx, Dy = tuplet
    D = gen_matrix_F(p[0], p[1])[1]
    D[0, 0] = Dx
    D[1, 1] = Dy
    k = np.arange(0, 10, 0.2)

    count_t1a = 0
    count_t1b = 0
    count_t2a = 0
    count_t2b = 0

    for _ in range(1000):  # 1,000 matrices
        m = gen_matrix_F(p[0], p[1])[0]
        ev = np.linalg.eigvals(m)
        if np.max(np.real(ev)) < 0:  # if matrix is stable
            Em = []
            Emi = []
            for ki in k:
                R = m - D * (ki ** 2)
                eigval = np.linalg.eigvals(R)
                Em.append(np.max(np.real(eigval)))
                idx_max = np.argmax(np.real(eigval))
                Emi.append(np.imag(eigval[idx_max]))
            a = np.max(Em)
            index = np.argmax(Em)
            nEm = np.array(Em)
            if a > 0:
                if Emi[index] == 0:
                    numZeroCrossing = np.count_nonzero(np.diff(np.sign(Em)))  # Count zero crossings
                    numpositivelocalmaxima = np.sum(nEm[argrelextrema(nEm, np.greater)] > 0) > 0
                    if numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 0:
                        count_t1a += 1
                    elif numpositivelocalmaxima > 0 and numZeroCrossing == 1:
                        count_t1b += 1
                    elif numpositivelocalmaxima == 0 and numZeroCrossing % 2 == 1:
                        count_t2a += 1
                    elif numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 1:
                        count_t2b += 1

    percent_turing_1 = (count_t1a + count_t1b) * 0.1
    percent_turing = (count_t1a + count_t1b + count_t2a + count_t2b) * 0.1
    return percent_turing_1, percent_turing

def main():
    dx = np.logspace(-3, 3, 100)
    dy = np.logspace(-3, 3, 100)

    parameters = [(2, 1/2), (3, 1/3), (5, 1/5), (50, 1/50)]

    dp_list = [(par, x, y) for par in parameters for x in dx for y in dy]

    start_time = time.time()

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Monitor progress
    results = []
    total_tasks = len(dp_list)
    for i, result in enumerate(pool.imap(process_tuple, dp_list), 1):
        results.append(result)
        if i % 100 == 0 or i == total_tasks:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (total_tasks - i) / i
            print(f'Processed {i}/{total_tasks} tasks... ({(i/total_tasks)*100:.2f}%)')
            print(f'Elapsed time: {elapsed_time:.2f} seconds')
            print(f'Estimated remaining time: {remaining_time:.2f} seconds')

    pool.close()
    pool.join()

    end_time = time.time()

    percent_turing_1 = [res[0] for res in results]
    percent_turing = [res[1] for res in results]

    df_data = pd.DataFrame({
        'N': [x[0][0] for x in dp_list],
        'Dx': [x[1] for x in dp_list],
        'Dy': [x[2] for x in dp_list],
        'Percentage_Turing_1': percent_turing_1,
        'Percentage_Turing': percent_turing
    })
    df_data.to_csv(os.path.join('./', 'heatmap_fig8.csv'), index=False)
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    # Plotting
    data = pd.read_csv('heatmap_fig8.csv')

    # Compute log10 of Dx and Dy
    data['log_Dx'] = np.log10(data['Dx'])
    data['log_Dy'] = np.log10(data['Dy'])

    def plot_heatmap_with_colorbar(data, percentage_column, title, first_panel_title):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        for i, N in enumerate([2, 3, 5, 50]):
            df_n = data[data['N'] == N]
            per = np.array(df_n[percentage_column])
            log_dx = np.array(df_n['log_Dx'])
            log_dy = np.array(df_n['log_Dy'])
            
            l = len(np.unique(log_dx))
            percentage = [per[i:i + l] for i in range(0, len(per), l)]

            row, col = divmod(i, 2)
            Y, X = np.meshgrid(np.unique(log_dy), np.unique(log_dx))
            ax = axs[row, col]
            colormap = ax.pcolormesh(X, Y, percentage, cmap='viridis', shading='auto')
            
            if row == 0 and col == 0:
                ax.set_title(first_panel_title, fontweight='bold', style='italic', fontsize=22)
            else:
                ax.set_title(f'$N = {N}$', fontweight='bold', style='italic', fontsize=22)
            
            ax.plot(X, X, color='white', linestyle='--')
            ax.set_xlim(min(log_dx), max(log_dx))
            ax.set_ylim(min(log_dy), max(log_dy))
            ax.set_aspect('equal')  # Ensure square plots
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=12)

        # Add color bar next to the last subplot
        cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.35])  
        fig.colorbar(colormap, cax=cbar_ax)

        fig.text(0.04, 0.5, r'$log_{10}$$\it{D_{2}}$', va='center', fontsize=28, rotation='vertical', fontweight='bold')
        fig.text(0.51, 0.03, r'$log_{10}$$\it{D_{1}}$', ha='center', fontsize=28, fontweight='bold')
        plt.savefig(f'/mnt/data/heatmap_{title.lower()}.png')
        plt.show()

    # Plot the heatmaps for 'Percentage_Turing_1' and 'Percentage_Turing'
    plot_heatmap_with_colorbar(data, 'Percentage_Turing_1', 'Turing 1', 'Turing I for N = 2')
    plot_heatmap_with_colorbar(data, 'Percentage_Turing', 'Turing', 'Turing for N = 2')

if __name__ == '__main__':
    main()
