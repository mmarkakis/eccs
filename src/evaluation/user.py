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

    def __init__(
        self,
        data: str | pd.DataFrame,
        true_graph: str | nx.DiGraph,
        test_graph: str | nx.DiGraph,
        treatment: str,
        outcome: str,
    ) -> None:
        """
        Initializes the ECCSUser object.

        Parameters:
            data: The dataset or the path to it.
            true_graph: The ground truth causal graph or the path to it.
            test_graph: The starting graph available to the user or the path to it.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
        """

        if isinstance(data, str):
            self._data = pd.read_csv(data)
        else:
            self._data = data

        if isinstance(true_graph, str):
            self._true_graph = nx.DiGraph(nx.nx_pydot.read_dot(true_graph))
        else:
            self._true_graph = true_graph

        if isinstance(test_graph, str):
            self._test_graph = nx.DiGraph(nx.nx_pydot.read_dot(test_graph))
        else:
            self._test_graph = test_graph

        self._treatment = treatment
        self._outcome = outcome

        self._eccs = ECCS(data, test_graph)
        self._eccs.set_treatment(treatment)
        self._eccs.set_outcome(outcome)

        self._invocations = 0
        self._ate_trajectory = [self.current_ate]
        self._edit_disance_trajectory = [self.current_graph_edit_distance]

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
    def invocations(self) -> int:
        """
        Returns the number of invocations of the ECCS system so far.

        Returns:
            The number of invocations of the ECCS system.
        """
        return self._invocations

    @property
    def ate_trajectory(self) -> list[float]:
        """
        Returns the trajectory of the ATE over the invocations.

        Returns:
            The trajectory of the ATE over the invocations.
        """
        return self._ate_trajectory

    @property
    def edit_distance_trajectory(self) -> list[int]:
        """
        Returns the trajectory of the graph edit distance over the invocations.

        Returns:
            The trajectory of the graph edit distance over the invocations.
        """
        return self._edit_disance_trajectory

    def invoke_eccs(self, method: str = None) -> None:
        """
        Invokes the ECCS system and updates the fixed and banned nodes accordingly.
        """

        if (method is None) or (method not in self._eccs.EDGE_SUGGESTION_METHODS):
            method = self._eccs.EDGE_SUGGESTION_METHODS[0]

        # Get suggested modifications and seelctively apply them
        edits, ate = self._eccs.suggest(method)
        for tup in edits.itertuples():
            edge = (tup.Source, tup.Target)
            if tup.Change == EdgeChange.ADD and edge not in self._eccs.graph.edges():
                if edge in self._true_graph.edges():
                    self._eccs.add_edge(tup.Source, tup.Target)
                    self._eccs.fix_edge(tup.Source, tup.Target)
                else:
                    self._eccs.ban_edge(tup.Source, tup.Target)
            elif tup.Change == EdgeChange.REMOVE and edge in self._eccs.graph.edges():
                if edge not in self._true_graph.edges():
                    self._eccs.remove_edge(tup.Source, tup.Target)
                    self._eccs.ban_edge(tup.Source, tup.Target)
                else:
                    self._eccs.fix_edge(tup.Source, tup.Target)
            elif tup.Change == EdgeChange.FLIP and edge in self._eccs.graph.edges():
                if edge[::-1] in self._true_graph.edges():
                    self._eccs.remove_edge(tup.Source, tup.Target)
                    self._eccs.add_edge(tup.Target, tup.Source)
                    self._eccs.fix_edge(tup.Target, tup.Source)
                else:
                    self._eccs.fix_edge(tup.Source, tup.Target)

        # Update bookkeeping
        self._invocations += 1
        self._ate_trajectory.append(self.current_ate)
        self._edit_disance_trajectory.append(self.current_graph_edit_distance)

    def run(self, steps: int, method: str = None) -> None:
        """
        Simulate the user for `steps` steps. In each step, the user invokes the ECCS
        system and updates the fixed and banned edges accordingly.

        Parameters:
            steps: The number of steps that the user executes
            method: The method to use for edge suggestions.
        """

        for _ in range(steps):
            self.invoke_eccs(method)
