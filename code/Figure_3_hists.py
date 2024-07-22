# This code is for making the histograms from Figure 3. Only need to change the path to the data below (input_dir).
# Arguments log_s and beta_only define whether to use a log scale and to plot only the beta distribution fit respectively.

class JacAnalysisClass:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def jacobian_plot(self, log_s=False, beta_only=False):
        def beta_approximation(data, ax, fig, colour, range_x=None):
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
            if range_x is None:
                x = np.linspace(0, 1, 100)
            else:
                x = (range_x - data_min) / (data_max - data_min)
            y = beta.pdf(x, alpha_mle, beta_mle)
            x_new = x * (data_max - data_min) + data_min
            y_new = y / (data_max - data_min)
            ynew = y_new[np.isfinite(y_new)]
            xnew = x_new[np.isfinite(y_new)]
            ax.plot(xnew, ynew, color=colour, linewidth=4)
            return x_new, ynew

        results_file_path_turing = os.path.join(
            self.input_dir, "jac_results_turing_total.txt"
        )
        hist_plots_path = os.path.join(self.output_dir, "histograms_beta_dstb")

        results_file_path_noturing = os.path.join(
            self.input_dir, "jac_results_no_turing_total.txt"
        )

        results_list_t = ""
        with open(results_file_path_turing, "r") as f:
            for line in f:
                results_list_t += line.strip()
        results_list_nt = ""
        with open(results_file_path_noturing, "r") as f:
            for line in f:
                results_list_nt += line.strip()

        # Define custom start and end delimiters
        end_delim = ";)"
        # Split the string using the start delimiter and then reassemble it
        dictionary_t = results_list_t.split(end_delim)
        dictionaries_t = []
        dictionary_nt = results_list_nt.split(end_delim)
        dictionaries_nt = []
        for dict_string in dictionary_t[:-1]:

            dict_string = dict_string.replace("array([", "[").replace("])", "]")
            dictionaries_t.append(np.array(ast.literal_eval(dict_string)))

        for dict_string in dictionary_nt[:-1]:

            dict_string = dict_string.replace("array([", "[").replace("])", "]")
            dictionaries_nt.append(np.array(ast.literal_eval(dict_string)))
        # Now we have the dictionaries as we want, next we evaluate the parameters obtained
        n = np.shape(dictionaries_t[0][0])[0]
        if beta_only:
            hist_plots_path += "/beta_plots"
        else:
            hist_plots_path += "/histograms"
            i = 0
            j = 0
        for i in range(n):
            for j in range(n):
                jac_element_t = []
                for jac_mat in dictionaries_t:
                    jac_element_t.append(jac_mat[0][i, j])
                jac_element_nt = []
                for jac_mat in dictionaries_nt:
                    jac_element_nt.append(jac_mat[0][i, j])
                if np.max(jac_element_nt) - np.min(jac_element_nt) == 0:
                    continue
                n_bins = 2 * int(np.sqrt(len(jac_element_t)))

                # v par
                fig, ax = plt.subplots(figsize=(4, 4))
                os.makedirs(hist_plots_path, exist_ok=True)
                x, ynew = beta_approximation(jac_element_t, ax, fig, "tab:blue")
                _, _ = beta_approximation(
                    jac_element_nt, ax, fig, "tab:orange"
                )  # , range_x = x)
                if log_s:
                    ax.set_yscale("log")
                spread_x = np.max(jac_element_t) - np.min(jac_element_t)
                x_lim = [
                    np.min(jac_element_t) - 1 / 20 * spread_x,
                    np.max(jac_element_t) + 1 / 20 * spread_x,
                ]
                x_lim = [-1.1, 0.9]
                ax.set_xlim(x_lim)
                # ynew = y_new[np.isfinite(y_new)]
                spread_y = np.max(ynew) - np.min(ynew)
                if log_s:
                    y_lim = [
                        np.max([np.min(ynew), 1e-5]),
                        np.max(ynew) ** 1.5,
                    ]
                else:
                    y_lim = [
                        np.max([np.min(ynew), 1e-5]) - 1 / 20 * spread_y,
                        np.max(ynew) + 1 / 20 * spread_y,
                    ]
                ax.set_ylim(y_lim)

                x_ticks = ax.get_xticks()
                y_ticks = ax.get_yticks()
                filtered_x_ticks = [
                    tick
                    for tick in x_ticks
                    if np.isfinite(tick) and x_lim[0] <= tick <= x_lim[1]
                ]
                filtered_y_ticks = [
                    tick
                    for tick in y_ticks
                    if np.isfinite(tick) and y_lim[0] <= tick <= y_lim[1]
                ]

                # Set the ticks on x-axis and y-axis to only include the minimum and maximum values
                ax.set_xticks([filtered_x_ticks[0], filtered_x_ticks[-1]])
                ax.set_yticks([filtered_y_ticks[0], filtered_y_ticks[-1]])

                plt.xticks(fontsize=20, fontname="Arial")
                plt.yticks(fontsize=20, fontname="Arial")

                if not beta_only:
                    ax.hist(
                        jac_element_t,
                        bins=n_bins,
                        density=True,
                        color="tab:blue",
                        alpha=0.5,
                    )
                    # ax.hist(
                    #     jac_element_nt,
                    #     bins=n_bins,
                    #     density=True,
                    #     color="tab:orange",
                    #     alpha=0.5,
                    # )
                # plt.xlabel(f"J_{i+1,j+1} value")
                # plt.ylabel("Count")
                # fig.tight_layout()
                if log_s:
                    # set y axis to log
                    outnamepdf = hist_plots_path + f"/jacobian_{i+1,j+1}_log.pdf"
                    outname = hist_plots_path + f"/jacobian_{i+1,j+1}_log"
                else:
                    outnamepdf = hist_plots_path + f"/jacobian_{i+1,j+1}.pdf"
                    outname = hist_plots_path + f"/jacobian_{i+1,j+1}"
                plt.savefig(outnamepdf, bbox_inches="tight")
                plt.savefig(outname, bbox_inches="tight")
                plt.clf()


if __name__ == "__main__":
    # Check if all dependencies are installed
    if not check_dependencies():
        print("One or more required libraries are not installed. Exiting.")
        exit(1)

    # Rest of your code
    import numpy as np
    from scipy.optimize import root
    from scipy.stats import loguniform
    import matplotlib.pyplot as plt
    import multiprocessing
    import argparse
    import os
    import ast
    import re
    from scipy.stats import beta
    from scipy.optimize import minimize

    input_dir = ""
    statclass = JacAnalysisClass(input_dir, input_dir)
    statclass.jacobian_plot(log_s=False, beta_only=True)
    statclass.jacobian_plot(log_s=True, beta_only=True)
    statclass.jacobian_plot(log_s=False, beta_only=False)
    statclass.jacobian_plot(log_s=True, beta_only=False)
