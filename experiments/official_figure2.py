import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from swarm_visualizer.boxplot import plot_paired_boxplot
from swarm_visualizer.utility.general_utils import set_plot_properties


def plot_combined():
    Propositions = ["A", "AandB", "AandBandC"]
    colors = ["r", "g", "b"]

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=1)
        plt.plot(possible_lengths, vals, colors[idx])
        plt.xlabel("Length of Video", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)

    # set legends
    plt.title("Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc="lower left")
    plt.savefig("data/Combined_length_vs_accuracy.png")


def plot_combined_with_variance():
    Propositions = ["A", "AandB", "AandBandC", "AuntilB"]
    colors = ["r", "g", "b", "orange"]

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/_viclip_plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)
        avg = np.average(vals, axis=0)
        var = np.var(vals, axis=0)
        std = np.std(vals, axis=0)
        upper = avg + std
        lower = avg - std

        # plt.plot(possible_lengths, avg, colors[idx])
        # plt.fill_between(possible_lengths, lower, upper, color=colors[idx], alpha=0.05)
        # plt.plot(possible_lengths, vals, colors[idx])
        plt.errorbar(possible_lengths, avg, yerr=var, fmt=colors[idx])
        plt.xlabel("Length of Video", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)

    # set legends
    plt.title("Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc="lower left")
    plt.savefig("data/_viclip_Combined_length_vs_accuracy.png")


def plot_until():
    Propositions = ["AuntilB"]
    colors = ["r", "g", "b"]

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        breakpoint()
        vals = np.average(results, axis=2)
        avg = np.average(vals, axis=0)
        var = np.var(vals, axis=0)
        std = np.std(vals, axis=0)
        upper = avg + std
        lower = avg - std

        # plt.plot(possible_lengths, avg, colors[idx])
        # plt.fill_between(possible_lengths, lower, upper, color=colors[idx], alpha=0.05)
        # plt.plot(possible_lengths, vals, colors[idx])
        plt.errorbar(possible_lengths, avg, yerr=var, fmt=colors[idx])
        plt.xlabel("Length of Video", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)

    # set legends
    plt.title("Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc="lower left")
    plt.savefig("data/Until_length_vs_accuracy.png")


def plot_combined_with_boxplot():
    Propositions = ["A", "AandB", "AandBandC"]
    colors = ["r", "g", "b"]

    data = []

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)
        data.append(vals)

    # Plot the box plot
    plt.boxplot(data, labels=Propositions, patch_artist=True)

    # Add labels and title to the plot
    plt.xlabel("Proposition", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.title("Finding Existence of a Proposition in a Video using VideoLlama")

    # Set colors for the box plot
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)

    # Save and show the plot
    plt.savefig("data/Combined_Proposition_vs_accuracy.png")
    plt.show()


def plot_seaborn():
    set_plot_properties()
    df = None
    Propositions = ["A", "AandB", "AandBandC", "AuntilB"]

    # Load VideoLLama data
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        avg = np.average(results, axis=0)  # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df) == None:
                    df = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["VideoLlama"],
                        }
                    )
                else:
                    df_new = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["VideoLlama"],
                        }
                    )
                    df = pd.concat([df, df_new], ignore_index=True)

    # Load ViClip data
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/_viclip_plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        possible_lengths = [5, 10, 15, 20, 25]  # Changing possible lenghts (matching 8 to 5)
        avg = np.average(results, axis=0)  # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df) == None:
                    df = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["ViCLIP"],
                        }
                    )
                else:
                    df_new = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["ViCLIP"],
                        }
                    )
                    df = pd.concat([df, df_new], ignore_index=True)

    # Load NSVS-TL
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f"data/_nsvstl_plotdata_{Proposition}.pkl", "rb") as f:
            results, possible_lengths, repeat = pickle.load(f)

        # thresh = 0.2142
        # results[results>=thresh] = 1
        # results[results<thresh] = 0

        # # with open(f'data/_nsvstl_plotdata_{Proposition}.pkl', 'wb') as f:
        # #     pickle.dump([results, possible_lengths, repeat], f)

        possible_lengths = [5, 10, 15, 20, 25]  # Changing possible lenghts (matching 8 to 5)
        avg = np.average(results, axis=0)  # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df) == None:
                    df = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["NSVS-TL"],
                        }
                    )
                else:
                    df_new = pd.DataFrame(
                        {
                            "TL Specification": [Proposition],
                            "Accuracy": [avg[i][j]],
                            "Length": [length],
                            "Approach": ["NSVS-TL"],
                        }
                    )
                    df = pd.concat([df, df_new], ignore_index=True)
    #

    df["TL Specification"] = df["TL Specification"].replace("A", "Single\nevent")
    df["TL Specification"] = df["TL Specification"].replace("AandB", "Two concurrent\nevents")
    df["TL Specification"] = df["TL Specification"].replace("AandBandC", "Three concurrent\nevents")
    df["TL Specification"] = df["TL Specification"].replace("AuntilB", "Event A until B")
    df["Approach"] = df["Approach"].replace("NSVS-TL", "NSVS-TL (Ours)")
    #
    # Define the desired order and corresponding colors for the 'Approach' column
    desired_order = ["ViCLIP", "VideoLlama", "NSVS-TL (Ours)"]
    colors = {"ViCLIP": "g", "VideoLlama": "b", "NSVS-TL (Ours)": "#e7984a"}

    # Ensure that the 'Approach' column in the dataframe follows the desired order
    df["Approach"] = pd.Categorical(df["Approach"], categories=desired_order, ordered=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Accuracy vs. Length
    plot_paired_boxplot(
        df,
        x_var="Length",
        y_var="Accuracy",
        hue="Approach",
        title_str="Accuracy of Baselines Drop with Video Length",
        ax=axes[0],
        pal=colors,
    )
    axes[0].set_title("Accuracy of Baselines Drop with Video Length", fontsize=20)
    axes[0].set_xlabel("Length (Number of frames)", fontsize=17)
    axes[0].get_legend().remove()  # Correct way to remove the legend
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, fontsize=15)

    # Generating random data for the second plot for demonstration
    # Plot 2: Accuracy vs. TL Specification
    plot_paired_boxplot(
        df,
        x_var="TL Specification",
        y_var="Accuracy",
        hue="Approach",
        title_str="Accuracy of Baselines Drop with Proposition Complexity",
        ax=axes[1],
        pal=colors,
    )
    axes[1].set_title("Accuracy of Baselines Drop with Proposition Complexity", fontsize=20)
    axes[1].set_xlabel("TL Specification", fontsize=17)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, fontsize=15)
    plt.legend(fontsize=13, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.40))
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.10)

    # Saving the figure
    plt.savefig("data/test_box_combined.png")
    plt.close()


if __name__ == "__main__":
    # plot_combined()
    # plot_combined_with_variance()
    # plot_until()
    # plot_combined_with_boxplot()
    plot_seaborn()
