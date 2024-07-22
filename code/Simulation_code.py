# This code is for simulating the data used in Figures 3 and 4.
# When running from terminal, it has different arguments:
# -s: number of samples to be used.
# -t: number of threads to be used.
# -o: output directory where to save all results.
# -m: model to be used, either '2-nodeA', '2-nodeB', '3-node'or '4-node'.
# -j: whether to compute Jacobian and save it.
# -p: whether to perform a parameter analysis (sensitivity analysis).

# Check for required libraries
def check_dependencies():
    try:
        import numpy as np
    except ImportError:
        print("numpy is not installed.")
        return False

    try:
        from scipy.optimize import root
    except ImportError:
        print("scipy is not installed.")
        return False

    try:
        from scipy.stats import loguniform
    except ImportError:
        print("scipy.stats is not installed.")
        return False

    try:
        from pyDOE import lhs
    except ImportError:
        print("pyDOE is not installed.")
        return False

    try:
        import pandas as pd
    except ImportError:
        print("pandas is not installed.")
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed.")
        return False

    try:
        import multiprocessing
    except ImportError:
        print("multiprocessing is not installed.")
        return False

    try:
        import argparse
    except ImportError:
        print("argparse is not installed.")
        return False

    try:
        import os
    except ImportError:
        print("os is not installed.")
        return False

    try:
        import re
    except ImportError:
        print("re is not installed.")
        return False

    try:
        import psutil
    except ImportError:
        print("psutil is not installed.")
        return False

    try:
        import ast
    except ImportError:
        print("ast is not installed.")
        return False

    return True


class TuringRobustnessClass:

    def __init__(self):
        self.args = self.parse_arguments()
        self.model = self.args.model
        self.jacobian_bool = self.args.jacobian_bool
        self.param_bool = self.args.param_analysis
        self.output_dir = self.args.output
        # if model == False:
        #     if adjacency_matrix == False:
        #         raise ValueError(
        #             'If model is not defined, the adjacency matrix needs to be given')
        #     else:
        #         self.adjacencymatrix = np.array(adjacency_matrix)
        if self.model == "2-nodeA":
            self.adjacencymatrix = np.array([[1, -1], [1, -1]])
            self.n_dimensions = 2
        if self.model == "2-nodeB":
            self.adjacencymatrix = np.array([[1, -1], [1, 0]])
            self.n_dimensions = 2
        if self.model == "3-node":
            self.adjacencymatrix = np.array([[0, -1, 0], [1, 0, -1], [-1, -1, 1]])
            self.n_dimensions = 3
        if self.model == "4-node":
            self.adjacencymatrix = np.array(
                [[0, -1, 0, -1], [1, 0, -1, -1], [-1, -1, 1, 0], [-1, 1, 0, 0]]
            )
            self.n_dimensions = 4

    def run_Turing_search(self):
        os.makedirs(self.output_dir, exist_ok=True)
        param_combinations = self.setup_params(self.args.samples)

        # Divide parameter sets among the number of threads
        chunk_size = len(param_combinations) // self.args.threads
        param_chunks = [
            param_combinations[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.args.threads)
        ]

        if len(param_combinations) % self.args.threads != 0:
            param_chunks[-1].extend(
                param_combinations[self.args.threads * chunk_size :]
            )

        # Run each chunk in a separate process with manual memory profiling
        with multiprocessing.Pool(self.args.threads) as pool:
            pool.starmap(
                self.profile_task,
                [
                    (chunk, f"{self.args.output}/thread_{i}")
                    for i, chunk in enumerate(param_chunks)
                ],
            )

        self.post_process()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Run Turing Robustness Analysis.")
        parser.add_argument(
            "-s",
            "--samples",
            type=int,
            default=1000,
            help="Number of parameter sets to generate.",
        )
        parser.add_argument(
            "-t",
            "--threads",
            type=int,
            default=4,
            help="Number of threads to use for parallel processing.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="results",
            help="Output directory for results and plots.",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="3-node",
            help="Model to be used (defines adjacency matrix).",
        )
        parser.add_argument(
            "-j",
            "--jacobian_bool",
            type=bool,
            default=False,
            help="Whether to save the Jacobian for all points or not (Boolean)",
        )
        parser.add_argument(
            "-p",
            "--param_analysis",
            type=bool,
            default=False,
            help="Whether to carry out parameter analysis or not (Boolean)",
        )
        return parser.parse_args()

    def generate_param_space_latin_hypercube(self, num_samples, bounds):
        num_params = sum(np.prod(bounds[key][2]) for key in bounds)
        lhs_samples = lhs(num_params, samples=num_samples)
        parameter_sets = []
        for i in range(num_samples):
            params = {}
            index = 0
            for param, (min_bound, max_bound, shape) in bounds.items():
                count = np.prod(shape)
                samples = lhs_samples[i, index : index + count]
                scaled_samples = loguniform.ppf(samples, a=min_bound, b=max_bound)
                params[param] = scaled_samples.reshape(shape)
                index += count
                if param == "K":
                    params[param] *= np.abs(self.adjacencymatrix)
            parameter_sets.append(params)
        return parameter_sets

    def setup_params(self, num_samples):
        bounds = {
            "V": (0.1, 10, (self.n_dimensions,)),
            "K": (0.01, 1, (self.n_dimensions, self.n_dimensions)),
            "B": (0.001, 0.1, (self.n_dimensions,)),
            "Mu": (0.01, 1, (self.n_dimensions,)),
        }
        return self.generate_param_space_latin_hypercube(num_samples, bounds)

    def ODE(self, x, A, V, k, n, b, mu):
        dxdt = np.zeros(len(A))
        for i in range(len(A)):
            regulation = V[i]
            for j in range(len(A)):
                regulation *= (
                    1
                    / (
                        1
                        + max((x[j] / (k[i, j] + 1e-10)) ** (-A[i, j] * n[i, j]), 1e-10)
                    )
                ) ** abs(A[i, j])
            dxdt[i] = b[i] + regulation - mu[i] * x[i]
        return dxdt

    def find_unique_steady_states(self, A, V, k, n, b, mu, num_guesses=10):
        unique_states = set()
        for _ in range(num_guesses):
            x_guess = np.random.rand(len(A)) * 10
            sol = root(self.ODE, x_guess, args=(A, V, k, n, b, mu), method="hybr")
            if sol.success:
                rounded_state = tuple(np.round(sol.x, decimals=5))
                unique_states.add(rounded_state)
        return unique_states

    def linear_stability_analysis(self, steady_state, A, D, V, k, n, b, mu):
        J = self.jacobian(steady_state, A, V, k, n, b, mu)
        jac_list = []
        jac_list.append(J)
        dispersion = []
        for kn in self.kn_values:
            Dk2 = np.diag(D) * kn**2
            J_mod = J - Dk2
            eigenvalues = np.linalg.eigvals(J_mod)
            max_real_part = np.max(eigenvalues.real)
            dispersion.append(max_real_part)
        return dispersion, jac_list

    def jacobian(self, x, A, V, k, n, b, mu, eps=1e-6):
        """Numerical approximation of the Jacobian matrix at point x."""
        jacob = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            x_eps = np.array(x, copy=True)
            x_eps[i] += eps
            dx = self.ODE(x_eps, A, V, k, n, b, mu) - self.ODE(x, A, V, k, n, b, mu)
            jacob[:, i] = dx / eps
        return jacob

    def save_dispersion_plot(self, result, kn_values, output_dir):
        plt.figure()
        plt.plot(kn_values, result["dispersion"], label="Max Real Part of Eigenvalues")
        plt.title(f"Turing Dispersion Relation for Params: {result['params']}")
        plt.xlabel("Wave Number k")
        plt.ylabel("Max Real Part of Eigenvalues")
        plt.axhline(0, color="red", linestyle="--")
        plt.grid(True)
        plt.legend()
        plot_filename = os.path.join(
            output_dir, f"dispersion_plot_{np.random.randint(100000)}.png"
        )
        plt.savefig(plot_filename)
        plt.close()

    def profile_task(self, params, output_dir):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # Memory in MiB
        self.turing_task(params, output_dir)
        mem_after = process.memory_info().rss / (1024**2)  # Memory in MiB
        print(
            f"Memory usage for {output_dir}: Before: {mem_before} MiB, After: {mem_after} MiB, Difference: {mem_after - mem_before} MiB"
        )

    def turing_task(self, params, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.A = self.adjacencymatrix
        self.n = 2 * np.ones((self.n_dimensions, self.n_dimensions))
        self.D = np.zeros(self.n_dimensions)
        self.D[0] = 1
        self.D[1] = 10
        self.kn_values = np.linspace(0, 10, 1000)

        total_Turing = 0
        total_states = 0
        results = []
        jac_els_turing = []
        jac_els_no_turing = []

        for param in params:
            V = param["V"]
            k = param["K"]
            b = param["B"]
            mu = param["Mu"]
            unique_states = self.find_unique_steady_states(self.A, V, k, self.n, b, mu)

            for state in unique_states:
                total_states += 1
                dispersion, jacobian = self.linear_stability_analysis(
                    state, self.A, self.D, V, k, self.n, b, mu
                )
                max_real_parts_np = np.array(dispersion)
                Turing = int(
                    max_real_parts_np[0] < 0 and any(max_real_parts_np[1:] > 0)
                )
                total_Turing += Turing

                if Turing:
                    change_mat = self.parameter_change(param, state, dispersion)
                    result = {
                        "params": param,
                        "steady_state": state,
                        "dispersion": dispersion,
                        "Turing": Turing,
                        "Change_params": change_mat,
                    }
                    results.append(result)
                    self.save_dispersion_plot(result, self.kn_values, output_dir)
                    jac_els_turing.append(jacobian)
                elif self.jacobian_bool:
                    jac_els_no_turing.append(jacobian)

        # Save results to a file
        results_file_path = os.path.join(output_dir, "turing_results.txt")
        with open(results_file_path, "w") as f:
            for result in results:
                f.write(f"{result}\n")
        if self.jacobian_bool:
            results_jac_file_path = os.path.join(output_dir, "jac_turing_results.txt")
            with open(results_jac_file_path, "w") as f:
                for result in jac_els_turing:
                    f.write(f"{result};)\n")

            results_jac_file_path = os.path.join(
                output_dir, "jac_no_turing_results.txt"
            )
            with open(results_jac_file_path, "w") as f:
                for result in jac_els_no_turing:
                    f.write(f"{result};)\n")
        # Output summary of Turing patterns
        summary = f"Summary test: number of Turing I = {total_Turing} out of {total_states} unique steady states --> fraction = {total_Turing / total_states if total_states else 0}\n"
        print(output_dir, summary)
        summary_file_path = os.path.join(output_dir, "summary.txt")
        with open(summary_file_path, "w") as f:
            f.write(summary)

    def parameter_change(self, param, state, old_dispersion):
        V = param["V"]
        k = param["K"]
        b = param["B"]
        mu = param["Mu"]
        dpar = 0.00001
        dpar = 0.0005
        dlambda_max_dV = np.zeros_like(V)
        dlambda_max_dk = np.zeros_like(k)
        dlambda_max_db = np.zeros_like(b)
        dlambda_max_dmu = np.zeros_like(mu)
        dkc_dV = np.zeros_like(V)
        dkc_dk = np.zeros_like(k)
        dkc_db = np.zeros_like(b)
        dkc_dmu = np.zeros_like(mu)

        for i in range(self.n_dimensions):
            Vnew = V
            knew = k
            bnew = b
            munew = mu
            Vnew[i] += dpar
            dlambda, dkc = self.difference_dispersion(
                state, Vnew, knew, bnew, munew, old_dispersion
            )
            dlambda_max_dV[i] = dlambda / dpar
            dkc_dV[i] = dkc / dpar

        for i in range(self.n_dimensions):
            for j in range(self.n_dimensions):
                Vnew = V
                knew = k
                bnew = b
                munew = mu
                knew[i, j] += dpar
                dlambda, dkc = self.difference_dispersion(
                    state, Vnew, knew, bnew, munew, old_dispersion
                )
                dlambda_max_dk[i, j] = dlambda / dpar
                dkc_dk[i, j] = dkc / dpar

        for i in range(self.n_dimensions):
            Vnew = V
            knew = k
            bnew = b
            munew = mu
            bnew[i] += dpar
            dlambda, dkc = self.difference_dispersion(
                state, Vnew, knew, bnew, munew, old_dispersion
            )
            dlambda_max_db[i] = dlambda / dpar
            dkc_db[i] = dkc / dpar

        for i in range(self.n_dimensions):
            Vnew = V
            knew = k
            bnew = b
            munew = mu
            munew[i] += dpar
            dlambda, dkc = self.difference_dispersion(
                state, Vnew, knew, bnew, munew, old_dispersion
            )
            dlambda_max_dmu[i] = dlambda / dpar
            dkc_dmu[i] = dkc / dpar
        return [
            dlambda_max_dV,
            dlambda_max_dk,
            dlambda_max_db,
            dlambda_max_dmu,
            dkc_dV,
            dkc_dk,
            dkc_db,
            dkc_dmu,
        ]

    def difference_dispersion(self, state, Vnew, knew, bnew, munew, old_dispersion):
        dispersion1 = old_dispersion
        max_real_parts_np_1 = np.array(dispersion1)
        sol2 = root(
            self.ODE,
            state,
            args=(self.adjacencymatrix, Vnew, knew, self.n, bnew, munew),
            method="hybr",
        )
        if sol2.success:
            rounded_state = tuple(np.round(sol2.x, decimals=5))
        else:
            print("The second steady state could not be found, using original")
            rounded_state = state
        dispersion2, _ = self.linear_stability_analysis(
            rounded_state, self.adjacencymatrix, self.D, Vnew, knew, self.n, bnew, munew
        )
        max_real_parts_np_2 = np.array(dispersion2)
        i_max_1 = np.argmax(max_real_parts_np_1)
        i_max_2 = np.argmax(max_real_parts_np_2)
        dlambda_max = np.max(max_real_parts_np_2) - np.max(max_real_parts_np_1)
        dkc = self.kn_values[i_max_2] - self.kn_values[i_max_1]
        return dlambda_max, dkc

    def analysis_of_parameters(self):
        results_list = ""
        with open(self.results_file_path_total, "r") as f:
            for line in f:
                results_list += line.strip()

        # Define custom start and end delimiters
        start_delim = "{'params"
        # Split the string using the start delimiter and then reassemble it
        parts = results_list.split(start_delim)
        dictionary = [start_delim + i for i in parts[1:]]
        self.dictionaries = []
        for dict_string in dictionary:

            dict_string = dict_string.replace("array([", "[").replace("])", "]")
            self.dictionaries.append(ast.literal_eval(dict_string))
        # Now we have the dictionaries as we want, next we evaluate the parameters obtained

        self.n = len(self.dictionaries[0]["params"]["V"])
        # Then we might want to add all data into a single array (maybe per parameter) so that we
        # can easily plot it

        self.V_matrix_kc = np.zeros((len(self.dictionaries), self.n))
        self.V_matrix_lmax = np.zeros((len(self.dictionaries), self.n))
        for s in range(len(self.dictionaries)):
            for i in range(self.n):
                self.V_matrix_lmax[s, i] = self.dictionaries[s]["Change_params"][0][i]
                self.V_matrix_kc[s, i] = self.dictionaries[s]["Change_params"][4][i]

        self.k_matrix_kc = np.zeros((len(self.dictionaries), self.n, self.n))
        self.k_matrix_lmax = np.zeros((len(self.dictionaries), self.n, self.n))
        for s in range(len(self.dictionaries)):
            for i in range(self.n):
                for j in range(self.n):
                    self.k_matrix_lmax[s, i, j] = np.array(
                        self.dictionaries[s]["Change_params"][1]
                    )[i, j]
                    self.k_matrix_kc[s, i, j] = np.array(
                        self.dictionaries[s]["Change_params"][5]
                    )[i, j]

        self.B_matrix_kc = np.zeros((len(self.dictionaries), self.n))
        self.B_matrix_lmax = np.zeros((len(self.dictionaries), self.n))
        for s in range(len(self.dictionaries)):
            for i in range(self.n):
                self.B_matrix_lmax[s, i] = self.dictionaries[s]["Change_params"][2][i]
                self.B_matrix_kc[s, i] = self.dictionaries[s]["Change_params"][6][i]

        self.mu_matrix_kc = np.zeros((len(self.dictionaries), self.n))
        self.mu_matrix_lmax = np.zeros((len(self.dictionaries), self.n))
        for s in range(len(self.dictionaries)):
            for i in range(self.n):
                self.mu_matrix_lmax[s, i] = self.dictionaries[s]["Change_params"][3][i]
                self.mu_matrix_kc[s, i] = self.dictionaries[s]["Change_params"][7][i]

        # next we decide which statistical analysis we want to carry out. We could first make a list and rank the parameters
        self.V_matrix_lmax_mean = np.mean(self.V_matrix_lmax, axis=0)
        self.V_matrix_kc_mean = np.mean(self.V_matrix_kc, axis=0)

        self.k_matrix_lmax_mean = np.mean(self.k_matrix_lmax, axis=0)
        self.k_matrix_kc_mean = np.mean(self.k_matrix_kc, axis=0)

        self.B_matrix_lmax_mean = np.mean(self.B_matrix_lmax, axis=0)
        self.B_matrix_kc_mean = np.mean(self.B_matrix_kc, axis=0)

        self.mu_matrix_lmax_mean = np.mean(self.mu_matrix_lmax, axis=0)
        self.mu_matrix_kc_mean = np.mean(self.mu_matrix_kc, axis=0)
        # Now we will rank these parameters by the lmax mean and kc mean values

    def parameter_rank(self, top_n=5):
        parameter_rank_lmax = pd.DataFrame(np.zeros((self.n**2 + 3 * self.n, 2)))
        parameter_rank_kc = pd.DataFrame(np.zeros((self.n**2 + 3 * self.n, 2)))

        index = 0
        # V
        for i in range(self.n):
            parameter_rank_lmax.iloc[index, 0] = self.V_matrix_lmax_mean[i]
            parameter_rank_kc.iloc[index, 0] = self.V_matrix_kc_mean[i]
            parameter_rank_lmax.iloc[index, 1] = parameter_rank_kc.iloc[index, 1] = (
                f"V_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            index += 1
        # k
        for i in range(self.n):
            for j in range(self.n):
                parameter_rank_lmax.iloc[index, 0] = self.k_matrix_lmax_mean[i, j]
                parameter_rank_kc.iloc[index, 0] = self.k_matrix_kc_mean[i, j]
                parameter_rank_lmax.iloc[index, 1] = parameter_rank_kc.iloc[
                    index, 1
                ] = (
                    f"k_{i,j}".replace("0", "A")
                    .replace("1", "B")
                    .replace("2", "C")
                    .replace("3", "D")
                )
                index += 1
        # B
        for i in range(self.n):
            parameter_rank_lmax.iloc[index, 0] = self.B_matrix_lmax_mean[i]
            parameter_rank_kc.iloc[index, 0] = self.B_matrix_kc_mean[i]
            parameter_rank_lmax.iloc[index, 1] = parameter_rank_kc.iloc[index, 1] = (
                f"B_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            index += 1
        # mu
        for i in range(self.n):
            parameter_rank_lmax.iloc[index, 0] = self.mu_matrix_lmax_mean[i]
            parameter_rank_kc.iloc[index, 0] = self.mu_matrix_kc_mean[i]
            parameter_rank_lmax.iloc[index, 1] = parameter_rank_kc.iloc[index, 1] = (
                f"mu_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            index += 1

        # And next we sort this array
        sorted_kc = np.array(
            parameter_rank_kc.sort_values(
                by=parameter_rank_kc.keys()[0], key=abs, ascending=False
            )
        )
        sorted_lmax = np.array(
            parameter_rank_lmax.sort_values(
                by=parameter_rank_lmax.keys()[0], key=abs, ascending=False
            )
        )

        top_kc = f"Top {top_n} parameters for changing k_c are:\n"
        top_lmax = f"Top {top_n} parameters for changing lambda_max are:\n"
        for i in range(top_n):
            new_top_kc = f"{i+1}. {sorted_kc[i,1]}, change = {sorted_kc[i,0]}\n"
            new_top_lmax = f"{i+1}. {sorted_lmax[i,1]}, change = {sorted_lmax[i,0]}\n"
            top_kc += new_top_kc
            top_lmax += new_top_lmax

        summary = top_kc + top_lmax
        print(self.output_dir, summary)
        summary_file_path = os.path.join(self.output_dir, "par_change_summary.txt")
        with open(summary_file_path, "w") as f:
            f.write(summary)

    def beta_approximation(self, data):
        data_min = np.min(data)
        data_max = np.max(data)
        # normalise data
        data_norm = (data - data_min) / (data_max - data_min)
        # Method of Moments
        mean = np.mean(data_norm)
        var = np.var(data_norm)
        common_factor = mean * (1 - mean) / var - 1
        alpha_mm = mean * common_factor
        beta_mm = (1 - mean) * common_factor

        # Maximum Likelihood Estimation
        def neg_log_likelihood(params, data):
            alpha_par, beta_par = params
            return -np.sum(beta.logpdf(data, alpha_par, beta_par))

        initial_guess = [alpha_mm, beta_mm]
        bounds = [(0.01, None), (0.01, None)]  # Ensure parameters are positive
        result = minimize(
            neg_log_likelihood, initial_guess, args=(data,), bounds=bounds
        )

        alpha_mle, beta_mle = result.x
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, alpha_mle, beta_mle)
        x_new = x * (data_max - data_min) + data_min
        y_new = y / (data_max - data_min)
        plt.plot(x_new, y_new)
        # return alpha_mle, beta_mle, data_min, data_max

    def jacobian_plot(self, log_s=False, turing=False, beta_only=False):
        if turing:
            results_file_path = os.path.join(
                self.output_dir, "jac_results_turing_total.txt"
            )
            hist_plots_path = os.path.join(
                self.output_dir, "histograms_turing_jacobian"
            )

        else:
            results_file_path = os.path.join(
                self.output_dir, "jac_results_no_turing_total.txt"
            )
            hist_plots_path = os.path.join(
                self.output_dir, "histograms_no_turing_jacobian"
            )

        results_list = ""
        with open(results_file_path, "r") as f:
            for line in f:
                results_list += line.strip()

        # Define custom start and end delimiters
        end_delim = ";)"
        # Split the string using the start delimiter and then reassemble it
        dictionary = results_list.split(end_delim)
        self.dictionaries = []
        for dict_string in dictionary[:-1]:

            dict_string = dict_string.replace("array([", "[").replace("])", "]")
            self.dictionaries.append(np.array(ast.literal_eval(dict_string)))
        # Now we have the dictionaries as we want, next we evaluate the parameters obtained
        n = np.shape(self.dictionaries[0][0])[0]
        if beta_only:
            hist_plots_path += "/beta_plots"
        else:
            hist_plots_path += "/histograms"
        for i in range(n):
            for j in range(n):
                jac_element = []
                for jac_mat in self.dictionaries:
                    jac_element.append(jac_mat[0][i, j])

                n_bins = 2 * int(np.sqrt(len(jac_element)))

                # v par

                os.makedirs(hist_plots_path, exist_ok=True)
                self.beta_approximation(jac_element)
                if not beta_only:
                    plt.hist(jac_element, bins=n_bins, density=True)
                plt.xlabel(f"J_{i+1,j+1} value")
                plt.ylabel("Count")
                if log_s:
                    plt.yscale("log")
                    outnamepdf = hist_plots_path + f"/jacobian_{i+1,j+1}_log.pdf"
                    outname = hist_plots_path + f"/jacobian_{i+1,j+1}_log"
                else:
                    outnamepdf = hist_plots_path + f"/jacobian_{i+1,j+1}.pdf"
                    outname = hist_plots_path + f"/jacobian_{i+1,j+1}"
                plt.savefig(outnamepdf)
                plt.savefig(outname)
                plt.clf()

    def hist_plot(self, log_s=False):
        n_bins = 2 * int(np.sqrt(len(self.dictionaries)))
        hist_plots_path = os.path.join(self.output_dir, "histograms")

        # v par
        hist_plots_path_par = os.path.join(hist_plots_path, "V_par")
        os.makedirs(hist_plots_path_par, exist_ok=True)
        for i in range(self.n):
            plt.hist(self.V_matrix_kc[:, i], bins=n_bins)
            plt.xlabel(
                rf"k_c change for V_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/kc_V_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_V_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/kc_V_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_V_{i}".replace("0", "A").replace(
                    "1", "B"
                ).replace("2", "C").replace("3", "D")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()

            plt.hist(self.V_matrix_lmax[:, i], bins=n_bins)
            plt.xlabel(
                f"lambda_max change for V_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/lmax_V_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_V_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/lmax_V_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_V_{i}".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()
        # k
        hist_plots_path_par = os.path.join(hist_plots_path, "k_par")
        os.makedirs(hist_plots_path_par, exist_ok=True)
        for i in range(self.n):
            for j in range(self.n):
                plt.hist(self.k_matrix_kc[:, i, j], bins=n_bins)
                plt.xlabel(
                    f"k_c change for k_{i,j}".replace("0", "A")
                    .replace("1", "B")
                    .replace("2", "C")
                    .replace("3", "D")
                )
                plt.ylabel("Count")
                if log_s:
                    plt.yscale("log")
                    outnamepdf = hist_plots_path_par + f"/kc_k_{i,j}_log.pdf".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                    outname = hist_plots_path_par + f"/kc_k_{i,j}_log".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                else:
                    outnamepdf = hist_plots_path_par + f"/kc_k_{i,j}.pdf".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                    outname = hist_plots_path_par + f"/kc_k_{i,j}".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                plt.savefig(outnamepdf)
                plt.savefig(outname)
                plt.clf()

                plt.hist(self.k_matrix_lmax[:, i, j], bins=n_bins)
                plt.xlabel(
                    f"lambda_max change for k_{i,j}".replace("0", "A")
                    .replace("1", "B")
                    .replace("2", "C")
                    .replace("3", "D")
                )
                plt.ylabel("Count")
                if log_s:
                    plt.yscale("log")
                    outnamepdf = hist_plots_path_par + f"/lmax_k_{i,j}_log.pdf".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                    outname = hist_plots_path_par + f"/lmax_k_{i,j}_log".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                else:
                    outnamepdf = hist_plots_path_par + f"/lmax_k_{i,j}.pdf".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                    outname = hist_plots_path_par + f"/lmax_k_{i,j}".replace(
                        "0", "A"
                    ).replace("1", "B").replace("2", "C").replace("3", "D")
                plt.savefig(outnamepdf)
                plt.savefig(outname)
                plt.clf()

        # B
        hist_plots_path_par = os.path.join(hist_plots_path, "B_par")
        os.makedirs(hist_plots_path_par, exist_ok=True)
        for i in range(self.n):
            plt.hist(self.B_matrix_kc[:, i], bins=n_bins)
            plt.xlabel(
                f"k_c change for B_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/kc_B_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_B_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/kc_B_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_B_{i}".replace("0", "A").replace(
                    "1", "B"
                ).replace("2", "C").replace("3", "D")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()

            plt.hist(self.B_matrix_lmax[:, i], bins=n_bins)
            plt.xlabel(
                f"lambda_max change for B_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/lmax_B_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_B_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/lmax_B_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_B_{i}".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()

        # mu
        hist_plots_path_par = os.path.join(hist_plots_path, "mu_par")
        os.makedirs(hist_plots_path_par, exist_ok=True)
        for i in range(self.n):
            plt.hist(self.mu_matrix_kc[:, i], bins=n_bins)
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/kc_mu_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_mu_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/kc_mu_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/kc_mu_{i}".replace("0", "A").replace(
                    "1", "B"
                ).replace("2", "C").replace("3", "D")
            plt.xlabel(
                rf"k_c change for \mu_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()

            plt.hist(self.mu_matrix_lmax[:, i], bins=n_bins)
            plt.xlabel(
                f"lambda_max change for \mu_{i}".replace("0", "A")
                .replace("1", "B")
                .replace("2", "C")
                .replace("3", "D")
            )
            plt.ylabel("Count")
            if log_s:
                plt.yscale("log")
                outnamepdf = hist_plots_path_par + f"/lmax_mu_{i}_log.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_mu_{i}_log".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            else:
                outnamepdf = hist_plots_path_par + f"/lmax_mu_{i}.pdf".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
                outname = hist_plots_path_par + f"/lmax_mu_{i}".replace(
                    "0", "A"
                ).replace("1", "B").replace("2", "C").replace("3", "D")
            plt.savefig(outnamepdf)
            plt.savefig(outname)
            plt.clf()

    def post_process(self):
        total_turing = 0
        total_states = 0
        all_results = []
        all_jac_t = []
        all_jac_nt = []

        # Aggregate results from all thread directories
        for thread_dir in os.listdir(self.output_dir):
            thread_path = os.path.join(self.output_dir, thread_dir)
            if os.path.isdir(thread_path):
                summary_path = os.path.join(thread_path, "summary.txt")
                results_path = os.path.join(thread_path, "turing_results.txt")
                jac_path_t = os.path.join(thread_path, "jac_turing_results.txt")
                jac_path_nt = os.path.join(thread_path, "jac_no_turing_results.txt")

                # Process summary.txt
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        for line in f:
                            match = re.search(
                                r"number of Turing I = (\d+) out of (\d+)", line
                            )
                            if match:
                                turing_count = int(match.group(1))
                                state_count = int(match.group(2))
                                total_turing += turing_count
                                total_states += state_count

                # Process turing_results.txt
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        for line in f:
                            all_results.append(line.strip())
                if self.jacobian_bool:
                    if os.path.exists(jac_path_t):
                        with open(jac_path_t, "r") as f:
                            for line in f:
                                all_jac_t.append(line.strip())
                    if os.path.exists(jac_path_nt):
                        with open(jac_path_nt, "r") as f:
                            for line in f:
                                all_jac_nt.append(line.strip())

        # Write combined summary_total.txt
        summary_total_path = os.path.join(self.output_dir, "summary_total.txt")
        with open(summary_total_path, "w") as f:
            summary = f"Total number of Turing I = {total_turing} out of {total_states} unique steady states --> fraction = {total_turing / total_states if total_states else 0}\n"
            f.write(summary)
        print(summary)

        # Write combined turing_results_total.txt
        results_total_path = os.path.join(self.output_dir, "turing_results_total.txt")
        with open(results_total_path, "w") as f:
            for result in all_results:
                f.write(result + "\n")
        self.results_file_path_total = results_total_path

        # Write combined jac_results_total.txt
        if self.jacobian_bool:
            results_total_path = os.path.join(
                self.output_dir, "jac_results_turing_total.txt"
            )
            with open(results_total_path, "w") as f:
                for result in all_jac_t:
                    f.write(result + "\n")

        # Write combined jac_results_total.txt
        if self.jacobian_bool:
            results_total_path = os.path.join(
                self.output_dir, "jac_results_no_turing_total.txt"
            )
            with open(results_total_path, "w") as f:
                for result in all_jac_nt:
                    f.write(result + "\n")
        # If params analisys is to be done:
        self.jacobian_plot(log_s=False, turing=False)
        self.jacobian_plot(log_s=True, turing=False)
        self.jacobian_plot(log_s=False, turing=False, beta_only=True)
        self.jacobian_plot(log_s=True, turing=False, beta_only=True)
        if total_turing > 0:
            if self.param_bool:
                self.analysis_of_parameters()
                self.parameter_rank(10)
                self.histplot()
                self.histplot(True)
            if self.jacobian_bool:
                self.jacobian_plot(log_s=False, turing=True)
                self.jacobian_plot(log_s=True, turing=True)
                self.jacobian_plot(log_s=False, turing=True, beta_only=True)
                self.jacobian_plot(log_s=True, turing=True, beta_only=True)


if __name__ == "__main__":
    # Check if all dependencies are installed
    if not check_dependencies():
        print("One or more required libraries are not installed. Exiting.")
        exit(1)

    # Rest of your code
    import numpy as np
    from scipy.optimize import root
    from scipy.stats import loguniform
    from pyDOE import lhs
    import matplotlib.pyplot as plt
    from scipy.stats import beta
    from scipy.optimize import minimize
    import multiprocessing
    import argparse
    import os
    import re
    import psutil
    import pandas as pd
    import ast
    import time

    t_initial = time.time()
    TRC = TuringRobustnessClass()
    TRC.run_Turing_search()
    print(f"Elapsed time was: {time.time()-t_initial} seconds")
