class JacAnalysisClass:
    def __init__(self):
        self.input_dir_2n = ""
        self.input_dir_3n = ""
        self.input_dir_4n = ""
        self.output_dir = ""

    def box_plots(self):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        # Read the data
        self.input_dir = [self.input_dir_2n, self.input_dir_3n, self.input_dir_4n]
        for input_dir in self.input_dir:
            results_file_path_turing = os.path.join(
                input_dir, "jac_results_turing_total.txt"
            )

            results_file_path_noturing = os.path.join(
                input_dir, "jac_results_no_turing_total.txt"
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
            moment1_nt = []
            moment1_t = []
            moment2_nt = []
            moment2_t = []
            moment3_nt = []
            moment3_t = []
            moment4_nt = []
            moment4_t = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    jac_element_t = []
                    for jac_mat in dictionaries_t:
                        jac_element_t.append(jac_mat[0][i, j])
                    jac_element_nt = []
                    for jac_mat in dictionaries_nt:
                        jac_element_nt.append(jac_mat[0][i, j])
                    if np.max(jac_element_nt) - np.min(jac_element_nt) == 0:
                        moment1_nt.append(0)
                        moment1_t.append(0)
                        moment2_nt.append(0)
                        moment2_t.append(0)
                        moment3_nt.append(0)
                        moment3_t.append(0)
                        moment4_nt.append(0)
                        moment4_t.append(0)
                        continue
                    moment1_nt.append(np.mean(jac_element_nt))
                    moment1_t.append(np.mean(jac_element_t))
                    moment2_nt.append(np.var(jac_element_nt))
                    moment2_t.append(np.var(jac_element_t))
                    moment3_nt.append(
                        np.mean((jac_element_nt - np.mean(jac_element_nt)) ** 3)
                        / (np.var(jac_element_nt) ** (3 / 2))
                    )
                    moment3_t.append(
                        np.mean((jac_element_t - np.mean(jac_element_t)) ** 3)
                        / (np.var(jac_element_t) ** (3 / 2))
                    )
                    moment4_nt.append(
                        np.mean((jac_element_nt - np.mean(jac_element_nt)) ** 4)
                        / (np.var(jac_element_nt) ** (4 / 2))
                    )
                    moment4_t.append(
                        np.mean((jac_element_t - np.mean(jac_element_t)) ** 4)
                        / (np.var(jac_element_t) ** (4 / 2))
                    )
                    # stat, p_value = levene(jac_element_nt, jac_element_t, center='median')  # Use 'median' for Brown-Forsythe test
                    # print(f"Levene's test statistic: {stat}")
                    # print(f"P-value: {p_value}")

                    # if p_value < 0.05:
                    #     print("The difference in variances is significant.")
                    # else:
                    #     print("The difference in variances is not significant.")

            # Plot the boxplots for the three different node systems
            # The boxplots should be 4 boxplots, one for each moment
            # Each boxplot must have three pairs of boxes, one for each node system and each pair the turing and no turing
            data1.append(moment1_t)
            data1.append(moment1_nt)
            data2.append(moment2_t)
            data2.append(moment2_nt)
            data3.append(moment3_t)
            data3.append(moment3_nt)
            data4.append(moment4_t)
            data4.append(moment4_nt)

        # Create DataFrame for both subplots
        df1 = [data1[0], data1[1], data1[2], data1[3], data1[4], data1[5]]
        df2 = [data2[0], data2[1], data2[2], data2[3], data2[4], data2[5]]

        # Create the 2x1 plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

        # Function to create the violin plot
        def create_violin_plot(ax, df):
            sns.violinplot(
                ax=ax,
                data=df,
                palette=[
                    "tab:blue",
                    "tab:orange",
                    "tab:blue",
                    "tab:orange",
                    "tab:blue",
                    "tab:orange",
                ],
            )
            new_labels = ["2 Nodes", "", "3 Nodes", "", "4 Nodes", ""]
            tick_font = {"labelsize": 20}
            ax.set_xticks(np.arange(6))
            ax.set_xticklabels(new_labels, fontsize=20, fontname="Arial")
            ax.tick_params(axis="y", **tick_font)

        def create_violin_plot(ax, df):
            sns.swarmplot(
                ax=ax,
                data=df,
                palette=[
                    "tab:blue",
                    "tab:orange",
                    "tab:blue",
                    "tab:orange",
                    "tab:blue",
                    "tab:orange",
                ],
                size=15,
            )
            new_labels = ["2 Nodes", "", "3 Nodes", "", "4 Nodes", ""]
            tick_font = {"labelsize": 20}
            ax.set_xticks(np.arange(6))
            ax.set_xticklabels(new_labels, fontsize=20, fontname="Arial")
            ax.tick_params(axis="y", **tick_font)

        axes[0].set_ylabel("Mean value", fontsize=30, fontname="Arial")
        axes[1].set_ylabel("Variance value", fontsize=30, fontname="Arial")
        # plt.yscale('log')

        # Create the plots in each subplot
        create_violin_plot(axes[0], df1)
        create_violin_plot(axes[1], df2)

        # Show the plot
        plt.xticks(fontsize=20, fontname="Arial")
        plt.yticks(fontsize=20, fontname="Arial")
        plt.tight_layout()
        plt.savefig(self.output_dir + "/swarm_plot.pdf", bbox_inches="tight")
        plt.savefig(self.output_dir + "/swarm_plot", bbox_inches="tight")
        plt.clf()

        os.makedirs(self.output_dir, exist_ok=True)
        tick_font = {"labelsize": 20}
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].boxplot(data1, positions=[1, 2, 4, 5, 7, 8])
        axs[0, 0].set_title("First moment", fontsize=25)
        axs[0, 1].boxplot(data2, positions=[1, 2, 4, 5, 7, 8])
        axs[0, 1].set_title("Second moment", fontsize=25)
        axs[1, 0].boxplot(data3, positions=[1, 2, 4, 5, 7, 8])
        axs[1, 0].set_title("Third moment", fontsize=25)
        axs[1, 1].boxplot(data4, positions=[1, 2, 4, 5, 7, 8])
        axs[1, 1].set_title("Fourth moment", fontsize=25)
        axs[0, 0].set_xticklabels(
            ["T", "NT", "T", "NT", "T", "NT"], fontsize=20, fontname="Arial"
        )
        axs[0, 1].set_xticklabels(
            ["T", "NT", "T", "NT", "T", "NT"], fontsize=20, fontname="Arial"
        )
        axs[1, 0].set_xticklabels(
            ["T", "NT", "T", "NT", "T", "NT"], fontsize=20, fontname="Arial"
        )
        axs[1, 1].set_xticklabels(
            ["T", "NT", "T", "NT", "T", "NT"], fontsize=20, fontname="Arial"
        )
        axs[0, 0].tick_params(axis="y", **tick_font)
        axs[1, 0].tick_params(axis="y", **tick_font)
        axs[0, 1].tick_params(axis="y", **tick_font)
        axs[1, 1].tick_params(axis="y", **tick_font)
        # draw a horizontal line at y=0 for all axis
        for ax in axs.flatten():
            ax.axhline(y=0, color="k", linestyle="--")

        plt.subplots_adjust(top=0.85, bottom=0.2, hspace=0.6, wspace=0.4)

        plt.savefig(self.output_dir + "/boxplots_all.pdf", bbox_inches="tight")
        plt.savefig(self.output_dir + "/boxplots_all", bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":

    import numpy as np
    from scipy.optimize import root
    from scipy.stats import loguniform
    import matplotlib.pyplot as plt
    import multiprocessing
    import argparse
    import os
    import ast
    import re
    import seaborn as sns
    from scipy.stats import beta
    from scipy.optimize import minimize
    from scipy.stats import levene

    statclass = JacAnalysisClass()
    statclass.box_plots()
    statclass.jacobian_plot(log_s=True, beta_only=True)
    statclass.jacobian_plot(log_s=False, beta_only=False)
    statclass.jacobian_plot(log_s=True, beta_only=False)
