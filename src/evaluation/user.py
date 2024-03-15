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

        print("Initialized ECCS user!")
        print(f"True ATE: {self.true_ate}")
        print(f"Initial ATE: {self.current_ate}")
        print(f"Initial ATE difference: {self.current_abs_ate_diff}")
        print(f"Initial edit distance: {self.current_graph_edit_distance}")
        print(f"An optimal edit path: {self.current_optimal_edit_path}")

    @property
    def true_ate(self) -> float:
        """
        Returns the true ATE of the treatment on the outcome.

        Returns:
            The true ATE of the treatment on the outcome.
        """
        return ATECalculator.get_ate_and_confidence(
            self._data, self._treatment, self._outcome, self._true_graph
        )["ATE"]

    @property
    def current_ate(self) -> float:
        """
        Returns the current ATE of the treatment on the outcome.

        Returns:
            The ATE of the treatment on the outcome based on the current graph.
        """
        return ATECalculator.get_ate_and_confidence(
            self._data, self._treatment, self._outcome, self._eccs.graph
        )["ATE"]

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
            self._eccs.graph,
            self._true_graph,
            node_match=(lambda n1, n2: n1["var_name"] == n2["var_name"]),
        )

    @property
    def current_optimal_edit_path(self) -> list[tuple]:
        """
        Returns the optimal edit path between the current graph and the true graph.

        Returns:
            The optimal edit path between the current graph and the true graph.
        """
        return nx.optimal_edit_paths(
            self._eccs.graph,
            self._true_graph,
            node_match=(lambda n1, n2: n1["var_name"] == n2["var_name"]),
        )[0]

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
    def abs_ate_diff_trajectory(self) -> list[float]:
        """
        Returns the trajectory of the absolute ATE difference over the invocations.

        Returns:
            The trajectory of the absolute ATE difference over the invocations.
        """
        return [abs(self.true_ate - ate) for ate in self._ate_trajectory]

    @property
    def edit_distance_trajectory(self) -> list[int]:
        """
        Returns the trajectory of the graph edit distance over the invocations.

        Returns:
            The trajectory of the graph edit distance over the invocations.
        """
        return self._edit_disance_trajectory

    def invoke_eccs(self, method: str = None) -> bool:
        """
        Invokes the ECCS system and updates the fixed and banned nodes accordingly.

        Returns:
            Whether the invocation returned any changes.
        """

        if (method is None) or (method not in self._eccs.EDGE_SUGGESTION_METHODS):
            method = self._eccs.EDGE_SUGGESTION_METHODS[0]

        # Get suggested modifications and seelctively apply them
        edits, ate = self._eccs.suggest(method)
        if len(edits) == 0:
            print(f"In iteration {self._invocations + 1} ECCS suggested no changes.")
            return False
        for tup in edits.itertuples():
            print(
                f"In iteration {self._invocations + 1} ECCS suggested: {tup.Change} {tup.Source} -> {tup.Destination}"
            )
            edge = (tup.Source, tup.Destination)
            if tup.Change == EdgeChange.ADD and edge not in self._eccs.graph.edges():
                if edge in self._true_graph.edges():
                    self._eccs.add_edge(tup.Source, tup.Destination)
                    self._eccs.fix_edge(tup.Source, tup.Destination)
                    print(
                        f"\tThis led the user to add and fix the edge {tup.Source} -> {tup.Destination}"
                    )
                else:
                    self._eccs.ban_edge(tup.Source, tup.Destination)
                    print(
                        f"\tThis led the user to ban the edge {tup.Source} -> {tup.Destination}"
                    )
            elif tup.Change == EdgeChange.REMOVE and edge in self._eccs.graph.edges():
                if edge not in self._true_graph.edges():
                    self._eccs.remove_edge(tup.Source, tup.Destination)
                    self._eccs.ban_edge(tup.Source, tup.Destination)
                    print(
                        f"\tThis led the user to remove and ban the edge {tup.Source} -> {tup.Destination}"
                    )
                else:
                    self._eccs.fix_edge(tup.Source, tup.Destination)
                    print(
                        f"\tThis led the user to fix the edge {tup.Source} -> {tup.Destination}"
                    )
            elif tup.Change == EdgeChange.FLIP and edge in self._eccs.graph.edges():
                if edge[::-1] in self._true_graph.edges():
                    self._eccs.remove_edge(tup.Source, tup.Destination)
                    self._eccs.add_edge(tup.Destination, tup.Source)
                    self._eccs.fix_edge(tup.Destination, tup.Source)
                    print(
                        f"\tThis led the user to flip the edge {tup.Source} -> {tup.Destination} and fix the edge {tup.Destination} -> {tup.Source}"
                    )
                else:
                    self._eccs.ban_edge(tup.Destination, tup.Source)
                    print(
                        f"\tThis led the user to ban the edge {tup.Destination} -> {tup.Source}"
                    )

        # Update bookkeeping
        self._invocations += 1
        self._ate_trajectory.append(self.current_ate)
        self._edit_disance_trajectory.append(self.current_graph_edit_distance)

        print(f"\tUpdated ATE: {self.current_ate}")
        print(f"\tUpdated ATE difference: {self.current_abs_ate_diff}")
        print(f"\tUpdated edit distance: {self.current_graph_edit_distance}")
        return True

    def run(self, steps: int, method: str = None) -> None:
        """
        Simulate the user for `steps` steps. In each step, the user invokes the ECCS
        system and updates the fixed and banned edges accordingly.

        Parameters:
            steps: The number of steps that the user executes
            method: The method to use for edge suggestions.
        """

        for _ in range(steps):
            suggested_edits = self.invoke_eccs(method)
            if not suggested_edits:
                print("ECCS suggested no changes. Stopping.")
                break
