import numpy as np
import matplotlib.pyplot as plt
import yaml
import os


def main():
    # Load the configuration file
    with open("plotter_config.yml", "r") as f:
        config = yaml.safe_load(f)

    path = config["path"]
    true_graph_name = config["true_graph_name"]
    dataset_name = config["dataset_name"]
    starting_graph_name = config["starting_graph_name"]
    user_interactions = config["user_interactions"]

    ### Graph edit distance

    file_path_best = os.path.join(
        path,
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_best_single_edge_change_{user_interactions}_edit_distance_trajectory.npy",
    )
    file_path_random = os.path.join(
        path,
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_random_single_edge_change_{user_interactions}_edit_distance_trajectory.npy",
    )

    data_best = np.load(file_path_best, allow_pickle=True)
    data_random = np.load(file_path_random, allow_pickle=True)

    num_data_points = 10
    plt.plot(
        range(num_data_points), data_best[:num_data_points], label="Best", marker="o"
    )
    plt.plot(
        range(num_data_points),
        data_random[:num_data_points],
        label="Random",
        marker="o",
    )
    plt.xlabel("User Interaction Index")
    plt.ylabel("Graph Edit Distance from Ground Truth")
    plt.yticks(range(0, int(max(max(data_best), max(data_random))) + 1))
    plt.xlim(0, num_data_points)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_{user_interactions}_graph_edit_distance.png"
    )

    ### Absolute ATE difference

    file_path_best = os.path.join(
        path,
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_best_single_edge_change_{user_interactions}_abs_ate_diff_trajectory.npy",
    )
    file_path_random = os.path.join(
        path,
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_random_single_edge_change_{user_interactions}_abs_ate_diff_trajectory.npy",
    )

    # Load the file
    data_best = np.load(file_path_best, allow_pickle=True)
    data_random = np.load(file_path_random, allow_pickle=True)

    num_data_points = 10
    plt.cla()
    plt.plot(
        range(num_data_points), data_best[:num_data_points], label="Best", marker="o"
    )
    plt.plot(
        range(num_data_points),
        data_random[:num_data_points],
        label="Random",
        marker="o",
    )
    plt.xlabel("User Interaction Index")
    plt.ylabel("Absolute ATE Difference")
    plt.xlim(0, num_data_points)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_{user_interactions}_abs_ate_diff.png"
    )


if __name__ == "__main__":
    main()
