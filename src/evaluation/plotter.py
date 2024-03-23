import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse
from matplotlib.axes import Axes


LINE_FORMATTING_DATA = {
    "Baseline": {
        "label": "Random Single Edge Change",
        "color": "#d3d3d3",
        "marker": "o",
        "path": "random_single_edge_change",
    },
    "ECCS Edge": {
        "label": "ECCS - Single Edge Change",
        "color": "#7F9FBA",
        "marker": "o",
        "path": "best_single_edge_change",
    },
    "ECCS Adj Set": {
        "label": "ECCS - Single Adjustment Set Change",
        "color": "#7FBA82",
        "marker": "o",
        "path": "best_single_adjustment_set_change",
    },
    "ECCS Adj Set Naive": {
        "label": "ECCS - Single Adjustment Set Change Naive",
        "color": "#7FBA82",
        "marker": "x",
        "path": "best_single_adjustment_set_change_naive",
    },
}

FONTSIZE = 16

min_ground_truth_ate = np.inf
max_ground_truth_ate = -np.inf


def plot_edit_distance(ax: Axes, method: str, points: int, base_path: str) -> float:
    """
    Plots the edit distance for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"])

    for filename in os.listdir(path):
        if filename.endswith("edit_distance_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            if len(data) < points:
                data = np.pad(data, (0, points - len(data)), "edge")

            if accumulator is None:
                accumulator = data
            else:
                accumulator += data

            file_count += 1

    if file_count == 0:
        return 0

    elementwise_average = accumulator / file_count

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def plot_ate_diff(ax: Axes, method: str, points: int, base_path: str) -> float:
    """
    Plots the Absolute Relative ATE Error for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0
    count_zeros = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"])

    for filename in os.listdir(path):
        if filename.endswith("ate_diff_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            diff_data = np.load(filepath)

            if len(diff_data) < points:
                diff_data = np.pad(diff_data, (0, points - len(diff_data)), "edge")

            if diff_data[-1] == 0:
                count_zeros += 1

            # Load the ate data to compute ground truth ATE
            ate_filepath = filepath.replace("ate_diff", "ate")
            ate_data = np.load(ate_filepath)
            if len(ate_data) < points:
                ate_data = np.pad(ate_data, (0, points - len(ate_data)), "edge")

            ground_truth_ate = ate_data[0] - diff_data[0]

            # Compute the absolute relative ATE error
            data = np.abs(diff_data / ground_truth_ate)

            if accumulator is None:
                accumulator = data
            else:
                accumulator += data

            file_count += 1

    print(f"File count was {file_count}")
    print(f"The final ATE difference was 0 for {count_zeros} files")

    if file_count == 0:
        return 0

    elementwise_average = accumulator / file_count

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def wrapup_plot(filename: str, ax: Axes, max_val: float) -> None:
    """
    Set final formatting for the plot and save it to a file.

    Parameters:
        filename: The name of the file to save the plot to.
        ax: The axis to save.
        maxes: The maximum value of the plotted data.
    """

    ax.set_ylim(0, 1.1 * max_val)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_xlabel("User Interaction Index", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(filename)
    plt.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the results")
    args = parser.parse_args()

    # Load the experiment configuration file
    with open(os.path.join(args.path, "config.yml"), "r") as f:
        config = yaml.safe_load(f)
    num_points = 1 + config["run_eccs"]["num_steps"]

    # Create a directory for the plots
    plots_path = os.path.join(args.path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    ### Graph edit distance
    print("Plotting graph edit distance...")
    _, ax = plt.subplots()
    max1 = plot_edit_distance(ax, "Baseline", num_points, args.path)
    max2 = plot_edit_distance(ax, "ECCS Edge", num_points, args.path)
    max3 = plot_edit_distance(ax, "ECCS Adj Set", num_points, args.path)
    max4 = plot_edit_distance(ax, "ECCS Adj Set Naive", num_points, args.path)
    ax.set_ylabel("Graph Edit Distance\nfrom Ground Truth", fontsize=FONTSIZE)
    wrapup_plot(
        os.path.join(plots_path, "edit_distance.png"), ax, max(max1, max2, max3, max4)
    )

    ### ATE difference
    print("Plotting ATE difference...")
    _, ax = plt.subplots()
    max1 = plot_ate_diff(ax, "Baseline", num_points, args.path)
    max2 = plot_ate_diff(ax, "ECCS Edge", num_points, args.path)
    max3 = plot_ate_diff(ax, "ECCS Adj Set", num_points, args.path)
    max4 = plot_ate_diff(ax, "ECCS Adj Set Naive", num_points, args.path)
    ax.set_ylabel("Absolute Relative ATE Error", fontsize=FONTSIZE)
    wrapup_plot(
        os.path.join(plots_path, "ate_error.png"), ax, max(max1, max2, max3, max4)
    )


if __name__ == "__main__":
    main()
