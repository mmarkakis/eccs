from __future__ import annotations
import contextlib
import pandas as pd
from typing import Optional, Any
import networkx as nx
from dowhy import CausalModel
from datetime import datetime
import warnings
import sympy as sp


class ATECalculator:
    """
    A class to calculate ATEs.
    """

    ATE_cache = {}

    def _get_backdoor_estimand_expr(self, estimand):
        # estimand is IdentifiedEstimand
        for k, v in estimand.items():
            if k != "backdoor":
                continue
            sp_expr_str = sp.pretty(v["estimand"], use_unicode=True)
            return sp_expr_str
        return None

    def get_ate_and_confidence(
        self,
        data: pd.DataFrame,
        treatment: str | int,
        outcome: str | int,
        graph: Optional[nx.DiGraph] = None,
        calculate_p_value: bool = False,
        calculate_std_error: bool = False,
        get_estimand: bool = False,
        print_timing_info: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate the ATE of `treatment` on `outcome`, alongside confidence measures.

        Parameters:
            data: The data to be used for causal analysis.
            treatment: The name or index of the treatment variable.
            outcome: The name or index of the outcome variable.
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

        with open("/dev/null", "w+") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
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
                        estimand_identifier = None
                        try:
                            estimand_identifier = self._get_backdoor_estimand_expr(
                                identified_estimand
                            )
                        except:
                            estimand_identifier = None
                        if (
                            estimand_identifier is not None
                            and estimand_identifier in self.ATE_cache
                        ):
                            return self.ATE_cache[estimand_identifier]

                        timings.append(datetime.now())
                        estimate = model.estimate_effect(
                            identified_estimand,
                            method_name="backdoor.linear_regression",
                            test_significance=True,
                        )
                        timings.append(datetime.now())
                        p_value = (
                            estimate.test_stat_significance()["p_value"].astype(float)[
                                0
                            ]
                            if calculate_p_value
                            else None
                        )
                        timings.append(datetime.now())
                        stderr = (
                            estimate.get_standard_error()
                            if calculate_std_error
                            else None
                        )
                        timings.append(datetime.now())
                        d = {
                            "ATE": float(estimate.value),
                            "P-value": p_value,
                            "Standard Error": stderr,
                        }
                        if get_estimand:
                            d["Estimand"] = identified_estimand
                        if estimand_identifier is not None:
                            self.ATE_cache[estimand_identifier] = d

                except:
                    raise ValueError

        timings.append(datetime.now())
        if print_timing_info:
            print("\tTimings:")
            for i in range(1, len(timings)):
                print(f"\t\tStep {i}: {timings[i] - timings[i-1]}")

        return d
