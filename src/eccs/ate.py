import contextlib
import pandas as pd
from typing import Optional, Any
import networkx as nx
from dowhy import CausalModel


class ATECalculator:
    """
    A class to calculate ATEs.
    """

    @staticmethod
    def get_ate_and_confidence(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph: Optional[nx.DiGraph()] = None,
        calculate_p_value: bool = False,
        calculate_std_error: bool = False,
        get_estimand: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate the ATE of `treatment` on `outcome`, alongside confidence measures.

        Parameters:
            data: The data to be used for causal analysis.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            graph: The graph to be used for causal analysis. If not specified, a two-node graph with just
                `treatment` and `outcome` is used.
            calculate_p_value: Whether to calculate the P-value of the ATE.
            calculate_std_error: Whether to calculate the standard error of the ATE.
            get_estimand: Whether to return the estimand used to calculate the ATE, as part of the returned dictionary.

        Returns:
            A dictionary containing the ATE of `treatment` on `outcome`, alongside confidence measures. If
            `get_estimand` is True, the estimand used to calculate the ATE is also returned.
        """

        if graph is None:
            graph = nx.DiGraph()
            graph.add_node(treatment)
            graph.add_node(outcome)
            graph.add_edge(treatment, outcome)

        # Use dowhy to get the ATE, P-value and standard error.
        with open("/dev/null", "w+") as f:
            try:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    model = CausalModel(
                        data=data[list(graph.nodes)],
                        treatment=treatment,
                        outcome=outcome,
                        graph=nx.nx_pydot.to_pydot(graph).to_string(),
                    )
                    identified_estimand = model.identify_effect(
                        proceed_when_unidentifiable=True
                    )
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression",
                        test_significance=True,
                    )
                    p_value = (
                        estimate.test_stat_significance()["p_value"].astype(float)[0]
                        if calculate_p_value
                        else None
                    )
                    stderr = (
                        estimate.get_standard_error() if calculate_std_error else None
                    )
                    d = {
                        "ATE": float(estimate.value),
                        "P-value": p_value,
                        "Standard Error": stderr,
                    }
                    if get_estimand:
                        d["Estimand"] = identified_estimand
                    return d
            except:
                raise ValueError
