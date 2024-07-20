class Jac_Sample_Sum:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def plot(self):

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
        n_samples = 1000000
        sample_sum_nt = np.zeros(n_samples)
        sample_sum_t = np.zeros(n_samples)
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
                if i == j:
                    continue
                sample_sum_nt += np.random.choice(jac_element_nt, n_samples)
                sample_sum_t += np.random.choice(jac_element_t, n_samples)

                # v par
        os.makedirs(hist_plots_path, exist_ok=True)
        # n_bins_t = 2 * int(np.sqrt(len(jac_element_t)))
        # n_bins_nt = 2 * int(np.sqrt(len(jac_element_nt)))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(sample_sum_nt, bins=70, density=True, color="tab:orange", alpha=0.5)
        ax.hist(sample_sum_t, bins=18, density=True, color="tab:blue", alpha=0.5)
        plt.yscale("log")
        plt.xticks(fontsize=15, fontname="Arial")
        plt.yticks(fontsize=15, fontname="Arial")
        outnamepdf = hist_plots_path + "/jacobian_sum.pdf"
        outname = hist_plots_path + "/jacobian_sum"
        plt.savefig(outnamepdf, bbox_inches="tight")
        plt.savefig(outname, bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    import os
    import ast

    input_dir = ""
    statclass = Jac_Sample_Sum(input_dir, input_dir)
    statclass.plot()
