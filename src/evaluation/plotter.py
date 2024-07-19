import numpy as np
import matplotlib as mpl
from tqdm.auto import tqdm

rc_fonts = {
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """,
}
mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt
import yaml
import os
import argparse
import json
from matplotlib.axes import Axes


LINE_FORMATTING_DATA = {
    "random_single_edge_change": {
        "label": r"\textsc{RandomEdit}",
        "color": "#d3d3d3",
        "marker": "o",
        "path": "random_single_edge_change",
    },
    "best_single_edge_change": {
        "label": r"\textsc{SingleEdit}",
        "color": "#7F9FBA",
        "marker": "o",
        "path": "best_single_edge_change",
    },
    "astar_single_edge_change": {
        "label": r"\textsc{HeuristicEdit}",
        "color": "#ba8a7f",
        "marker": "o",
        "path": "astar_single_edge_change",
    },
    "best_single_adjustment_set_change": {
        "label": r"\textsc{AdjSetEdit}",
        "color": "#7FBA82",
        "marker": "o",
        "path": "best_single_adjustment_set_change",
    },
    "best_single_adjustment_set_change_opt": {
        "label": r"\textsc{AdjSetEdit} (Opt)",
        "color": "#BA7FB7",
        "marker": "o",
        "path": "best_single_adjustment_set_change_opt",
    },
}


FONTSIZE = 20


def plot_edit_distance(
    ax: Axes, method: str, points: int, base_path: str, prefixes: list[str]
) -> float:
    """
    Plots the edit distance for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        prefixes: The prefixes of the files to consider.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.startswith(tuple(prefixes)) and filename.endswith(
            "edit_distance_trajectory.npy"
        ):
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


def plot_ate_diff(
    ax: Axes, method: str, points: int, base_path: str, prefixes: list[str]
) -> float:
    """
    Plots the Absolute Relative ATE Error for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        prefixes: The prefixes of the files to consider.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0
    count_zeros = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.startswith(tuple(prefixes)) and filename.endswith(
            "ate_diff_trajectory.npy"
        ):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            diff_data = np.load(filepath, allow_pickle=True)

            if len(diff_data) < points:
                diff_data = np.pad(diff_data, (0, points - len(diff_data)), "edge")

            if diff_data[-1] == 0:
                count_zeros += 1

            # Load the ate data to compute ground truth ATE
            ate_filepath = filepath.replace("ate_diff", "ate")
            ate_data = np.load(ate_filepath, allow_pickle=True)
            if len(ate_data) < points:
                ate_data = np.pad(ate_data, (0, points - len(ate_data)), "edge")

            ground_truth_ate = ate_data[0] - diff_data[0]

            # Compute the absolute relative ATE error
            data = np.abs(diff_data / ground_truth_ate)

            for i, v in enumerate(data):
                if v < 10e-4:
                    data[i] = 0

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
    ax: Axes, method: str, points: int, base_path: str, prefixes: list[str]
) -> float:
    """
    Plots the duration of each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        prefixes: The prefixes of the files to consider.

    Returns:
        The maximum plotted value.
    """

    accumulator = [0.0] * points
    file_counts = [0] * points

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.startswith(tuple(prefixes)) and filename.endswith(
            "invocation_duration_trajectory.npy"
        ):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            # Load the freshness data to compute rounds
            fresh_filepath = filepath.replace("invocation_duration", "fresh_edits")
            fresh_data = np.load(fresh_filepath)

            # Set latency to 0 for rounds where there were no fresh edits
            for i, v in enumerate(fresh_data):
                if v == 0:
                    data[i] = 0

            # Convert to rounds by removing all 0s from the list
            data = [x for x in data if x > 0]

            for i in range(len(data)):
                file_counts[i + 1] += 1

            if len(data) < points:
                data = np.pad(
                    data, (1, points - len(data) - 1), "constant", constant_values=0
                )

            accumulator += data

    if all(x == 0 for x in file_counts):
        return 0

    elementwise_average = [
        i / j if j != 0 else 0 for i, j in zip(accumulator, file_counts)
    ]

    retval = max(elementwise_average)

    elementwise_average[0] = None
    elementwise_average = [x for x in elementwise_average if x is None or x > 0]

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return retval


def plot_fresh_edits(
    ax: Axes, method: str, points: int, base_path: str, prefixes: list[str]
) -> float:
    """
    Plots the number of fresh edits for each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        prefixes: The prefixes of the files to consider.

    Returns:
        The maximum plotted value.
    """

    accumulator = [0] * points
    file_counts = [0] * points

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.startswith(tuple(prefixes)) and filename.endswith(
            "fresh_edits_trajectory.npy"
        ):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            # Convert to rounds by removing all 0s from the list
            data = [x for x in data if x > 0]

            for i in range(len(data)):
                file_counts[i + 1] += 1 if data[i] > 0 else 0

            if len(data) < points:
                orig_len = len(data)
                data = np.pad(
                    data, (1, points - orig_len - 1), "constant", constant_values=0
                )

            if accumulator is None:
                accumulator = data
            else:
                accumulator += data

    if all(x == 0 for x in file_counts):
        return 0

    elementwise_average = [
        i / j if j != 0 else 0 for i, j in zip(accumulator, file_counts)
    ]

    if method == "best_single_adjustment_set_change":
        print(file_counts)

    retval = max(elementwise_average)

    elementwise_average[0] = None
    elementwise_average = [x for x in elementwise_average if x is None or x > 0]

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return retval


def plot_zero_ate_diff(
    ax: Axes, method: str, points: int, base_path: str, prefixes: list[str]
) -> float:
    """
    Plots the fraction of experiments with zero ATE difference at each round.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
        prefixes: The prefixes of the files to consider.
    """

    accumulator = [0] * points
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.startswith(tuple(prefixes)) and filename.endswith(
            "ate_diff_trajectory.npy"
        ):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            diff_data = np.load(filepath, allow_pickle=True)

            if len(diff_data) < points:
                diff_data = np.pad(diff_data, (0, points - len(diff_data)), "edge")

            accumulator += (diff_data < 10e-4).astype(int)

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


def plot_invocation_duration_scaling(
    ax: Axes, method: str, base_path: str, prefix_to_data_point: dict[str, str]
) -> float:
    """
    Plots the duration of each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        base_path: The base path to the results.
        prefix_to_data_point: A mapping from file prefix to the data point into
            which the data of that file should be incorporated.

    Returns:
        The maximum plotted value.
    """

    data_points = set(prefix_to_data_point.values())

    accumulators = {data_point: 0.0 for data_point in data_points}
    invocation_counts = {data_point: 0 for data_point in data_points}

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("invocation_duration_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)
            data_point = prefix_to_data_point[filename[:12]]

            # Load the freshness data to compute rounds
            fresh_filepath = filepath.replace("invocation_duration", "fresh_edits")
            fresh_data = np.load(fresh_filepath)

            # Set latency to 0 for rounds where there were no fresh edits, and count the
            # invocation otherwise.
            for i, v in enumerate(fresh_data):
                if v == 0:
                    data[i] = 0
                else:
                    invocation_counts[data_point] += 1

            # Accumulate the right data point
            accumulators[data_point] += sum(data)

    if all(x == 0 for x in invocation_counts):
        return 0

    elementwise_average = {
        data_point: (
            accumulators[data_point] / invocation_counts[data_point]
            if invocation_counts[data_point] != 0
            else 0
        )
        for data_point in data_points
    }

    ax.plot(
        elementwise_average.keys(),
        elementwise_average.values(),
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average.values())


def wrapup_plot(
    filename: str,
    ax: Axes,
    max_val: float,
    log_y_axis: bool = False,
    x_unit: str = "Judgment",
) -> None:
    """
    Set final formatting for the plot and save it to a file. Also print stats about the plotted
    lines to another file.

    Parameters:
        filename: The name of the file to save the plot to.
        ax: The axis to save.
        max_val: The maximum value of the plotted data.
        log_y_axis: Whether to use a log scale for the y-axis.
        x_unit: The units of the x axis ("Judgment" or "Round")
    """

    # Deal with the stats
    num_points = 0
    with open(filename + ".csv", "w") as f:
        f.write("label,last_y,average_y\n")
        for line in ax.get_lines():
            y_data = line.get_ydata()
            if len(y_data) > num_points:
                num_points = len(y_data)
            y_data = [x for x in y_data if x is not None]
            f.write(f"{line.get_label()},{y_data[-1]},{np.mean(y_data)}\n")

    # Deal with the figure
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    if x_unit == "Judgment" or x_unit == "Round":
        x_unit = f"{x_unit} " + r"\#"
        ax.set_xticks(np.arange(0, num_points, 2))
    elif len(ax.get_lines()) > 0:
        ax.set_xticks(ax.get_lines()[0].get_xdata())
    ax.set_xlabel(x_unit, fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    if log_y_axis:
        ax.set_yscale("log")
        ax.set_ylim(0.001, 10 ** (1.1 * np.log10(max_val)))
    else:
        ax.set_ylim(0, 1.1 * max_val)
    if (
        False
    ):  # These are the hard-coded y limits for /home/markakis/eccs/evaluation/2024-04-05 12:10:57.816692
        if "edit_distance" in filename:
            ax.set_ylim(0, 24.432597305389223)
        elif "ate_error" in filename:
            ax.set_ylim(0, 0.6739732398970182)
        elif "invocation_duration" in filename:
            ax.set_ylim(0.001, 253.37828732457606)
        elif "fresh_edits" in filename:
            ax.set_ylim(0, 4.776765147721583)
        elif "zero_ate_diff" in filename:
            ax.set_ylim(0, 0.8694610778443115)

    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=300)
    plt.cla()


def plotter(path: str, skip: bool = False):
    """
    Plot the experiment data at `path`.

    Parameters:
        path: The path to the experiment data.
        skip: Whether to skip re-generating plots that exist.
    """

    # Load the experiment configuration file
    with open(os.path.join(path, "config.yml"), "r") as f:
        config = yaml.safe_load(f)
    num_points = 1 + config["run_eccs"]["num_steps"]

    # Create a directory for the plots
    plots_path = os.path.join(path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Figure out the ground truth dags for each parameter combination
    if type(config["gen_dag"]["num_nodes"]) != list:
        config["gen_dag"]["num_nodes"] = [config["gen_dag"]["num_nodes"]]
    if type(config["gen_dag"]["edge_prob"]) != list:
        config["gen_dag"]["edge_prob"] = [config["gen_dag"]["edge_prob"]]
    combinations = [
        (num_nodes, edge_prob)
        for num_nodes in config["gen_dag"]["num_nodes"]
        for edge_prob in config["gen_dag"]["edge_prob"]
    ]
    combination_to_ground_truth_dags = {}
    for file in os.listdir(os.path.join(path, "ground_truth_dags")):
        if file.endswith(".json"):
            name = file[:12]
            with open(os.path.join(path, "ground_truth_dags", file), "r") as f:
                params = json.load(f)
            combination_to_ground_truth_dags.setdefault(
                (params["num_nodes"], params["edge_prob"]), []
            ).append(name)

    for num_nodes, edge_prob in combinations:
        print(f"Plotting for num_nodes={num_nodes}, edge_prob={edge_prob}...")
        prefixes = combination_to_ground_truth_dags[(num_nodes, edge_prob)]

        # Create a directory for the plots
        comb_plots_path = os.path.join(
            plots_path, f"num_nodes={num_nodes}_edge_prob={edge_prob}"
        )
        os.makedirs(comb_plots_path, exist_ok=True)

        ### Graph edit distance
        print("Plotting graph edit distance...")
        if skip and os.path.exists(os.path.join(comb_plots_path, "edit_distance.png")):
            print("Skipping edit distance plot")
        else:
            _, ax = plt.subplots()
            max_y = 0
            for method in LINE_FORMATTING_DATA:
                max_y = max(
                    max_y,
                    plot_edit_distance(ax, method, num_points, path, prefixes),
                )
            ax.set_ylabel("Graph Edit Distance", fontsize=FONTSIZE)
            wrapup_plot(os.path.join(comb_plots_path, "edit_distance"), ax, max_y)

        ### ATE difference
        print("Plotting ATE difference...")
        if skip and os.path.exists(os.path.join(comb_plots_path, "ate_error.png")):
            print("Skipping ATE Error plot")
        else:
            _, ax = plt.subplots()
            max_y = 0
            for method in LINE_FORMATTING_DATA:
                max_y = max(
                    max_y, plot_ate_diff(ax, method, num_points, path, prefixes)
                )
            ax.set_ylabel("ARE_ATE", fontsize=FONTSIZE)
            wrapup_plot(os.path.join(comb_plots_path, "ate_error"), ax, max_y)

        ### Invocation Duration
        print("Plotting Invocation Duration...")
        if skip and os.path.exists(
            os.path.join(comb_plots_path, "invocation_duration.png")
        ):
            print("Skipping Invocation Duration plot")
        else:
            _, ax = plt.subplots()
            max_y = 0
            for method in LINE_FORMATTING_DATA:
                max_y = max(
                    max_y,
                    plot_invocation_duration(ax, method, num_points, path, prefixes),
                )

            ax.set_ylabel("ECCS Latency (s)", fontsize=FONTSIZE)
            wrapup_plot(
                os.path.join(comb_plots_path, "invocation_duration"),
                ax,
                max_y,
                log_y_axis=True,
                x_unit="Round",
            )

        ### Number of fresh edits
        print("Plotting Fresh Edits...")
        if skip and os.path.exists(os.path.join(comb_plots_path, "fresh_edits.png")):
            print("Skipping Fresh Edits plot")
        else:
            _, ax = plt.subplots()
            max_y = 0
            for method in LINE_FORMATTING_DATA:
                max_y = max(
                    max_y, plot_fresh_edits(ax, method, num_points, path, prefixes)
                )

            ax.set_ylabel(r"\# Fresh Edits", fontsize=FONTSIZE)
            wrapup_plot(
                os.path.join(comb_plots_path, "fresh_edits"), ax, max_y, x_unit="Round"
            )

        ### Fraction of experiments with zero ATE difference at that round
        print("Plotting Fraction of experiments with zero ATE difference...")
        if skip and os.path.exists(os.path.join(comb_plots_path, "zero_ate_diff.png")):
            print("Skipping Zero ATE Difference plot")
        else:
            _, ax = plt.subplots()
            max_y = 0
            for method in LINE_FORMATTING_DATA:
                max_y = max(
                    max_y, plot_zero_ate_diff(ax, method, num_points, path, prefixes)
                )

            ax.set_ylabel(
                "Fraction of Experiments with\nZero ARE_ATE", fontsize=FONTSIZE
            )
            wrapup_plot(
                os.path.join(comb_plots_path, "zero_ate_diff"),
                ax,
                max_y,
                x_unit="Round",
            )

    ### Latency scaling by num_nodes
    print("Plotting latency scaling by num_nodes...")
    if skip and os.path.exists(os.path.join(plots_path, "scaling_num_nodes.png")):
        print("Skipping latency scaling by num_nodes plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        prefix_to_data_point = {
            prefix: num_nodes
            for (num_nodes, _), prefixes in combination_to_ground_truth_dags.items()
            for prefix in prefixes
        }

        for method in LINE_FORMATTING_DATA:
            max_y = max(
                max_y,
                plot_invocation_duration_scaling(
                    ax, method, path, prefix_to_data_point
                ),
            )
        ax.set_ylabel("ECCS Latency (s)", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "scaling_num_nodes"), ax, max_y, x_unit="Num Nodes"
        )

    ### Latency scaling by edge_prob
    print("Plotting latency scaling by edge_prob...")
    if skip and os.path.exists(os.path.join(plots_path, "scaling_edge_prob.png")):
        print("Skipping latency scaling by edge_prob plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        prefix_to_data_point = {
            prefix: edge_prob
            for (_, edge_prob), prefixes in combination_to_ground_truth_dags.items()
            for prefix in prefixes
        }
        for method in LINE_FORMATTING_DATA:
            max_y = max(
                max_y,
                plot_invocation_duration_scaling(
                    ax, method, path, prefix_to_data_point
                ),
            )
        ax.set_ylabel("ECCS Latency (s)", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "scaling_edge_prob"),
            ax,
            max_y,
            x_unit="Edge Probability",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the results")
    parser.add_argument(
        "--skip", type=bool, default=False, help="Don't re-generate plots that exist"
    )
    args = parser.parse_args()

    plotter(args.path, args.skip)
