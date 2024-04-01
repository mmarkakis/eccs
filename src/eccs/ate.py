from __future__ import annotations
import contextlib
import pandas as pd
from typing import Optional, Any
import networkx as nx
from dowhy import CausalModel
from datetime import datetime


class ATECalculator:
    """
    A class to calculate ATEs.
    """

    @staticmethod
    def get_ate_and_confidence(
        data: pd.DataFrame,
        treatment: str | int,
        outcome: str | int,
        graph: Optional[nx.DiGraph] = None,
        calculate_p_value: bool = False,
        calculate_std_error: bool = False,
        get_estimand: bool = False,
        bootstrap_reps: int = 10,
        bootstrap_fraction: float = 0.1,
        print_timing_info: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate the ATE of `treatment` on `outcome`, alongside confidence measures.

        Parameters:
            data: The data to be used for causal analysis.
            treatment_idx: The name or index of the treatment variable.
            outcome_idx: The name or index of the outcome variable.
            graph: The graph to be used for causal analysis. If not specified, a two-node graph with just
                `treatment` and `outcome` is used.
            calculate_p_value: Whether to calculate the P-value of the ATE.
            calculate_std_error: Whether to calculate the standard error of the ATE.
            get_estimand: Whether to return the estimand used to calculate the ATE, as part of the returned dictionary.
            print_timing_info: Whether to print timing information.

        Returns:
            A dictionary containing the ATE of `treatment` on `outcome`, alongside confidence measures. If
            `get_estimand` is True, the estimand used to calculate the ATE is also returned.
        """
        timings = []
        timings.append(datetime.now())

        treatment_name = (
            data.columns[treatment] if isinstance(treatment, int) else treatment
        )
        outcome_name = data.columns[outcome] if isinstance(outcome, int) else outcome

        if graph is None:
            graph = nx.DiGraph()
            graph.add_node(treatment_name)
            graph.add_node(outcome_name)
            graph.add_edge(treatment_name, outcome_name)

        # Use dowhy to get the ATE, P-value and standard error.
        timings.append(datetime.now())
        d = {}

        try:
            timings.append(datetime.now())
            model = CausalModel(
                data=data[list(graph.nodes)],
                treatment=treatment_name,
                outcome=outcome_name,
                graph=graph,
            )
            timings.append(datetime.now())
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )
            timings.append(datetime.now())
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=True,
            )
            timings.append(datetime.now())
            p_value = (
                estimate.test_stat_significance()["p_value"].astype(float)[0]
                if calculate_p_value
                else None
            )
            timings.append(datetime.now())
            stderr = (
                estimate.get_standard_error({'num_simulations': bootstrap_reps, 'sample_size_fraction': bootstrap_fraction}) if calculate_std_error else None
            )
            timings.append(datetime.now())
            d = {
                "ATE": float(estimate.value),
                "P-value": p_value,
                "Standard Error": stderr,
            }
            if get_estimand:
                d["Estimand"] = identified_estimand

        except:
            raise ValueError

        timings.append(datetime.now())
        if print_timing_info:
            print("\tTimings:")
            for i in range(1, len(timings)):
                print(f"\t\tStep {i}: {timings[i] - timings[i-1]}")

        return d
