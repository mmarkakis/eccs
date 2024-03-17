import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the results")
    args = parser.parse_args()


    # Load the experiment configuration file
    with open(os.path.join(args.path, 'config.yml'), "r") as f:
        config = yaml.safe_load(f)
    baselines_path = os.path.join(args.path, "baseline_results")


    # Create a directory for the plots
    plots_path = os.path.join(args.path, "plots")
    os.makedirs(plots_path, exist_ok=True)


    ### Graph edit distance
    print("Graph edit distance:")
    accumulator = None
    file_count = 0

    for filename in os.listdir(baselines_path):
        if filename.endswith('edit_distance_trajectory.npy'):
            # Load the list from the file
            filepath = os.path.join(baselines_path, filename)
            data = np.load(filepath)

            if accumulator is None:
                # Initialize the accumulator with the first list
                accumulator = data
            else:
                # Add the current list to the accumulator
                accumulator += data

            file_count += 1

    if file_count > 0:
        elementwise_average = accumulator / file_count

    print(file_count)
    print(elementwise_average)


    ### Absolute ATE difference
    print("Absolute ATE difference:")
    accumulator = None
    file_count = 0

    for filename in os.listdir(baselines_path):
        if filename.endswith('abs_ate_diff_trajectory.npy'):
            # Load the list from the file
            filepath = os.path.join(baselines_path, filename)
            data = np.load(filepath)

            if accumulator is None:
                # Initialize the accumulator with the first list
                accumulator = data
            else:
                # Add the current list to the accumulator
                accumulator += data

            file_count += 1

    if file_count > 0:
        elementwise_average = accumulator / file_count

    print(file_count)
    print(elementwise_average)
    
    return


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

    plt.plot(
        range(len(data_best)), data_best, label="Best", marker="o"
    )
    plt.plot(
        range(len(data_best)),
        data_random[:len(data_best)],
        label="Random",
        marker="o",
    )
    plt.xlabel("User Interaction Index")
    plt.ylabel("Graph Edit Distance from Ground Truth")
    plt.yticks(range(0, int(max(max(data_best), max(data_random))) + 1))
    plt.xlim(0, len(data_best))
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

    plt.cla()
    plt.plot(
        range(len(data_best)), data_best, label="Best", marker="o"
    )
    plt.plot(
        range(len(data_best)),
        data_random[:len(data_best)],
        label="Random",
        marker="o",
    )
    plt.xlabel("User Interaction Index")
    plt.ylabel("Absolute ATE Difference")
    plt.xlim(0, len(data_best))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{true_graph_name}_{dataset_name}_{starting_graph_name}_{user_interactions}_abs_ate_diff.png"
    )


if __name__ == "__main__":
    main()
