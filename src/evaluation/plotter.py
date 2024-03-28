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
    "Oracle": {
        "label": "Oracle",
        "color": "#000000",
        "marker": "o",
        "path": "best_single_edge_change_oracle",
    },
}

FONTSIZE = 16

min_ground_truth_ate = np.inf
max_ground_truth_ate = -np.inf


def plot_edit_distance(
    ax: Axes, method: str, points: int, base_path: str, methods: list[str]
) -> float:
    """
    Plots the edit distance for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        methods: The list of methods that were run in this experiment.

    Returns:
        The maximum plotted value.
    """

    if LINE_FORMATTING_DATA[method]["path"] not in methods:
        return 0

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
                accumulator = [float(i) for i in data]
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


def plot_ate_diff(ax: Axes, method: str, points: int, base_path: str, methods: list[str]) -> float:
    """
    Plots the Absolute Relative ATE Error for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        methods: The list of methods that were run in this experiment.

    Returns:
        The maximum plotted value.
    """

    if LINE_FORMATTING_DATA[method]["path"] not in methods:
        return 0

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
                accumulator = [float(i) for i in data]
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


def plot_invocation_duration(
    ax: Axes, method: str, points: int, base_path: str, methods: list[str]
) -> float:
    """
    Plots the duration of each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        methods: The list of methods that were run in this experiment.

    Returns:
        The maximum plotted value.
    """

    if LINE_FORMATTING_DATA[method]["path"] not in methods:
        return 0

    accumulator = None
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"])

    for filename in os.listdir(path):
        if filename.endswith("invocation_duration_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            if len(data) < points:
                data = np.pad(
                    data, (0, points - len(data)), "constant", constant_values=0
                )

            if accumulator is None:
                accumulator = [float(i) for i in data]
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


def plot_edits_per_invocation(
    ax: Axes, method: str, points: int, base_path: str, methods: list[str]
) -> float:
    """
    Plots the numbert of edits for each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        methods: The list of methods that were run in this experiment.

    Returns:
        The maximum plotted value.
    """

    if LINE_FORMATTING_DATA[method]["path"] not in methods:
        return 0

    accumulator = None
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"])

    for filename in os.listdir(path):
        if filename.endswith("edits_per_invocation_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            if len(data) < points:
                data = np.pad(
                    data, (0, points - len(data)), "constant", constant_values=0
                )

            if accumulator is None:
                accumulator = [float(i) for i in data]
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
    parser.add_argument(
        "--skip", type=bool, default=False, help="Don't re-generate plots that exist"
    )
    args = parser.parse_args()

    # Load the experiment configuration file
    with open(os.path.join(args.path, "config.yml"), "r") as f:
        config = yaml.safe_load(f)
    num_points = 1 + config["run_eccs"]["num_steps"]
    methods = config["run_eccs"]["methods"]

    # Create a directory for the plots
    plots_path = os.path.join(args.path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    ### Graph edit distance
    print("Plotting graph edit distance...")
    if args.skip and os.path.exists(os.path.join(plots_path, "edit_distance.png")):
        print("Skipping edit distance plot")
    else:
        _, ax = plt.subplots()
        max1 = plot_edit_distance(ax, "Baseline", num_points, args.path, methods)
        max2 = plot_edit_distance(ax, "ECCS Edge", num_points, args.path, methods)
        max3 = plot_edit_distance(ax, "ECCS Adj Set", num_points, args.path, methods)
        max4 = plot_edit_distance(
            ax, "ECCS Adj Set Naive", num_points, args.path, methods
        )
        max5 = plot_edit_distance(ax, "Oracle", num_points, args.path, methods)
        ax.set_ylabel("Graph Edit Distance\nfrom Ground Truth", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "edit_distance.png"),
            ax,
            max(max1, max2, max3, max4, max5),
        )

    ### ATE difference
    print("Plotting ATE difference...")
    if args.skip and os.path.exists(os.path.join(plots_path, "ate_error.png")):
        print("Skipping ATE Error plot")
    else:
        _, ax = plt.subplots()
        max1 = plot_ate_diff(ax, "Baseline", num_points, args.path, methods)
        max2 = plot_ate_diff(ax, "ECCS Edge", num_points, args.path, methods)
        max3 = plot_ate_diff(ax, "ECCS Adj Set", num_points, args.path, methods)
        max4 = plot_ate_diff(ax, "ECCS Adj Set Naive", num_points, args.path, methods)
        max5 = plot_ate_diff(ax, "Oracle", num_points, args.path, methods)
        ax.set_ylabel("Absolute Relative ATE Error", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "ate_error.png"),
            ax,
            max(max1, max2, max3, max4, max5),
        )

    ### Invocation Duration
    print("Plotting Invocation Duration...")
    if args.skip and os.path.exists(
        os.path.join(plots_path, "invocation_duration.png")
    ):
        print("Skipping Invocation Duration plot")
    else:
        _, ax = plt.subplots()
        max1 = plot_invocation_duration(ax, "Baseline", num_points, args.path, methods)
        max2 = plot_invocation_duration(ax, "ECCS Edge", num_points, args.path, methods)
        max3 = plot_invocation_duration(
            ax, "ECCS Adj Set", num_points, args.path, methods
        )
        max4 = plot_invocation_duration(
            ax, "ECCS Adj Set Naive", num_points, args.path, methods
        )
        max5 = plot_invocation_duration(ax, "Oracle", num_points, args.path, methods)
        ax.set_ylabel("Invocation Duration (s)", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "invocation_duration.png"),
            ax,
            max(max1, max2, max3, max4, max5),
        )

    ### Edits per Invocation
    print("Plotting Edits per Invocation...")
    if args.skip and os.path.exists(
        os.path.join(plots_path, "edits_per_invocation.png")
    ):
        print("Skipping Edits per Invocation plot")
    else:
        _, ax = plt.subplots()
        max1 = plot_edits_per_invocation(ax, "Baseline", num_points, args.path, methods)
        max2 = plot_edits_per_invocation(
            ax, "ECCS Edge", num_points, args.path, methods
        )
        max3 = plot_edits_per_invocation(
            ax, "ECCS Adj Set", num_points, args.path, methods
        )
        max4 = plot_edits_per_invocation(
            ax, "ECCS Adj Set Naive", num_points, args.path, methods
        )
        max5 = plot_edits_per_invocation(ax, "Oracle", num_points, args.path, methods)
        ax.set_ylabel("Edits per Invocation", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "edits_per_invocation.png"),
            ax,
            max(max1, max2, max3, max4, max5),
        )


if __name__ == "__main__":
    main()
