import argparse
import yaml
import sys

sys.path.append("../..")
from src.generators.random_dag_generator import RandomDAGGenerator
from src.generators.random_dataset_generator import RandomDatasetGenerator
from src.evaluation.causal_discovery import CausalDiscovery
from src.evaluation.user import ECCSUser
import networkx as nx
import os
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def simulate(
    args: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
        str,
        str,
        str,
        str,
        str,
        int,
        int,
    ]
) -> None:
    """
    Similate a single run of ECCS user and save the results.

    Parameters:
        args: a tuple containing:
            data: The dataset to be used for causal analysis.
            ground_truth_dag: The ground truth DAG.
            starting_dag: The starting DAG for the ECCS user.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            method: The method to be used to suggest edits.
            results_path: The path to save the results to.
            dataset_name: The name of the dataset.
            num_steps: The number of steps to run the ECCS user for.
            i: The index of the run, if applicable.
    """

    (
        data,
        ground_truth_dag,
        starting_dag,
        treatment,
        outcome,
        method,
        results_path,
        dataset_name,
        num_steps,
        i,
    ) = args

    exp_prefix = os.path.join(
        results_path,
        f"{dataset_name}_{starting_dag['name']}_{treatment}_{outcome}_{method}_{'' if i == None else f'{i}_'}",
    )
    f = open(exp_prefix + ".log", "w")
    sys.stdout = f
    sys.stderr = f

    user = ECCSUser(
        data,
        ground_truth_dag["graph"],
        starting_dag["graph"],
        treatment,
        outcome,
    )

    user.run(num_steps, method)

    f.flush()

    np.save(
        f"{exp_prefix}ate_trajectory.npy",
        user.ate_trajectory,
    )
    np.save(
        f"{exp_prefix}ate_diff_trajectory.npy",
        user.ate_diff_trajectory,
    )
    np.save(
        f"{exp_prefix}edit_distance_trajectory.npy",
        user.edit_distance_trajectory,
    )

    f.close()


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=str,
        default="./iterative_config.yml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="../../evaluation/",
        help="Output directory path",
    )
    args = parser.parse_args()

    # Read config yml file and create bookkeeping dir structure
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    work_path = os.path.join(args.out_path, f"{datetime.now()}")
    os.makedirs(work_path, exist_ok=True)
    with open(os.path.join(work_path, "config.yml"), "w") as file:
        yaml.dump(config, file)

    # 1. Generate the ground truth dags
    print("-----------------")
    print(f"{datetime.now()} Phase 1: Generating ground truth dags")
    ground_truth_dags_path = os.path.join(work_path, "ground_truth_dags")
    os.makedirs(ground_truth_dags_path, exist_ok=True)
    ground_truth_dags = {}
    for _ in tqdm(range(config["gen_dag"]["ground_truth_dags"])):
        ret_dict = RandomDAGGenerator.generate(
            config["gen_dag"]["num_nodes"],
            config["gen_dag"]["edge_prob"],
            tuple(config["gen_dag"]["edge_weight_range"]),
            tuple(config["gen_dag"]["edge_noise_sd_range"]),
            ground_truth_dags_path,
        )
        ground_truth_dags[ret_dict["name"]] = ret_dict

    # 2. Generate the datasets
    print("-----------------")
    print(f"{datetime.now()} Phase 2: Generating datasets")
    datasets_path = os.path.join(work_path, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    dataset_names = {}
    for dag_dict in tqdm(ground_truth_dags.values()):
        dataset_names[dag_dict["name"]] = []
        for _ in range(config["gen_dataset"]["datasets_per_ground_truth_dag"]):
            dataset_dict = RandomDatasetGenerator.generate(
                dag_dict["name"],
                dag_dict["edge_matrix"],
                dag_dict["noise_matrix"],
                config["gen_dataset"]["num_points"],
                config["gen_dataset"]["min_source_val"],
                config["gen_dataset"]["max_source_val"],
                datasets_path,
            )
            dataset_names[dag_dict["name"]].append(dataset_dict["name"])

    # 3. Generate the random starting dags
    print("-----------------")
    print(f"{datetime.now()} Phase 3: Generating starting dags")
    starting_dags_path = os.path.join(work_path, "starting_dags")
    os.makedirs(starting_dags_path, exist_ok=True)
    starting_dags = {}
    for _ in tqdm(range(config["gen_starting_dag"]["starting_dags"])):
        ret_val = RandomDAGGenerator.generate(
            config["gen_starting_dag"]["num_nodes"],
            config["gen_starting_dag"]["edge_prob"],
            tuple(config["gen_starting_dag"]["edge_weight_range"]),
            tuple(config["gen_starting_dag"]["edge_noise_sd_range"]),
            starting_dags_path,
        )
        starting_dags[ret_val["name"]] = ret_val

    # 4. Run the methods and store results
    print("-----------------")
    print(f"{datetime.now()} Phase 4: Starting experiment")
    eccs_results_path = os.path.join(work_path, "eccs_results")
    baseline_results_path = os.path.join(work_path, "baseline_results")
    os.makedirs(eccs_results_path, exist_ok=True)
    os.makedirs(baseline_results_path, exist_ok=True)
    num_steps = config["run_eccs"]["num_steps"]
    method = config["run_eccs"]["method"]

    tasks = []
    pbar = tqdm(
        total=config["gen_dag"]["ground_truth_dags"]
        * config["gen_dataset"]["datasets_per_ground_truth_dag"]
        * config["gen_starting_dag"]["starting_dags"]
        * int(
            config["gen_dag"]["num_nodes"] * (config["gen_dag"]["num_nodes"] - 1) / 2
        )  # Choices of treatment / outcome
        * (1 + config["run_baseline"]["num_tries"])
    )
    # For each ground truth dag...
    for ground_truth_dag_name, datasets in dataset_names.items():
        ground_truth_dag = ground_truth_dags[ground_truth_dag_name]
        # For each dataset...
        for dataset_name in datasets:
            data = pd.read_csv(os.path.join(datasets_path, f"{dataset_name}.csv"))
            # For each starting dag...
            for starting_dag in starting_dags.values():
                # For each choice of treatment...
                for treatment_idx in range(config["gen_dag"]["num_nodes"]):
                    # For each choice of outcome...
                    for outcome_idx in range(
                        treatment_idx + 1, config["gen_dag"]["num_nodes"]
                    ):

                        treatment = f"v{treatment_idx}"
                        outcome = f"v{outcome_idx}"

                        if not nx.has_path(
                            ground_truth_dag["graph"], treatment, outcome
                        ) or not nx.has_path(starting_dag["graph"], treatment, outcome):
                            pbar.update(1 + config["run_baseline"]["num_tries"])
                            continue

                        # Run ECCS
                        tasks.append(
                            (
                                data,
                                ground_truth_dag,
                                starting_dag,
                                treatment,
                                outcome,
                                method,
                                eccs_results_path,
                                dataset_name,
                                num_steps,
                                None,
                            )
                        )

                        for i in range(config["run_baseline"]["num_tries"]):
                            tasks.append(
                                (
                                    data,
                                    ground_truth_dag,
                                    starting_dag,
                                    treatment,
                                    outcome,
                                    "random_single_edge_change",
                                    baseline_results_path,
                                    dataset_name,
                                    num_steps,
                                    i,
                                )
                            )

    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(simulate, task) for task in tasks]

        # As each future completes, update the progress bar
        for _ in as_completed(futures):
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
