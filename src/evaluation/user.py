import networkx as nx
from ..eccs.eccs import ECCS, EdgeChange
from ..eccs.ate import ATECalculator
import pandas as pd
import numpy as np


class ECCSUser:
    """
    A class simulating a user of the ECCS system.

    The user knows the ground truth causal graph and responds to the system's suggestions
    accordingly.
    """

    # The maximum number of invocations of the ECCS system
    MAX_INVOCATIONS = 100

    def __init__(
        self,
        data_path: str,
        true_graph_path: str,
        test_graph_path: str,
        treatment: str,
        outcome: str,
    ) -> None:
        """
        Initializes the ECCSUser object.

        Parameters:
            data_path: The path to the dataset.
            true_graph_path: The path to the ground truth causal graph.
            test_graph_path: The path to the starting graph available to the user.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
        """

        self._data = pd.read_csv(data_path)
        self._true_graph = nx.DiGraph(nx.nx_pydot.read_dot(true_graph_path))
        self._test_graph = nx.DiGraph(nx.nx_pydot.read_dot(test_graph_path))
        self._treatment = treatment
        self._outcome = outcome

        self._eccs = ECCS(data_path, test_graph_path)
        self._eccs.set_treatment(treatment)
        self._eccs.set_outcome(outcome)

        self._invocations = 0
        self._ate_trajectory = [self.current_ate]

    @property
    def true_ate(self) -> float:
        """
        Returns the true ATE of the treatment on the outcome.

        Returns:
            The true ATE of the treatment on the outcome.
        """
        return ATECalculator.get_ate_and_confidence(
            self._true_graph, self._treatment, self._outcome
        )

    @property
    def current_ate(self) -> float:
        """
        Returns the current ATE of the treatment on the outcome.

        Returns:
            The ATE of the treatment on the outcome based on the current graph.
        """
        return ATECalculator.get_ate_and_confidence(
            self._eccs.graph, self._treatment, self._outcome
        )

    @property
    def current_abs_ate_diff(self) -> float:
        """
        Returns the absolute difference between the true ATE and the current ATE.

        Returns:
            The absolute difference between the true ATE and the current ATE.
        """
        return abs(self.true_ate - self.current_ate)

    @property
    def current_graph_edit_distance(self) -> int:
        """
        Returns the edit distance between the current graph and the true graph.

        Returns:
            The edit distance between the current graph and the true graph.
        """
        return nx.graph_edit_distance(
            self._true_graph,
            self._eccs.graph,
            node_match=(lambda n1, n2: n1.name == n2.name),
            edge_subst_cost=(lambda x, y: np.inf),
        )

    @property
    def can_invoke_again(self) -> bool:
        """
        Returns whether the user has reached the maximum number of ECCS invocations.

        Returns:
            Whether the user can invoke the ECCS system again.
        """
        return self._invocations < self.MAX_INVOCATIONS

    def invoke_eccs(self, method: str = None) -> None:
        """
        Invokes the ECCS system and updates the fixed and banned nodes accordingly.
        """

        if (method is None) or (method not in self._eccs.EDGE_SUGGESTION_METHODS):
            method = self._eccs.EDGE_SUGGESTION_METHODS[0]

        if not self.can_invoke_again:
            return

        # Get suggested modifications
        edits, ate = self._eccs.suggest(method)
        for tup in edits.itertuples():
            if tup.Change == EdgeChange.ADD:


            elif tup.Change == EdgeChange.REMOVE:


            elif tup.Change == EdgeChange.FLIP:
                

        # Update bookkeeping
        self._invocations += 1
        self._ate_trajectory.append(self.current_ate)
