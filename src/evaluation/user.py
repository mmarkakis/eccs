from __future__ import annotations
import networkx as nx
from ..eccs.eccs import ECCS
from ..eccs.edges import EdgeEditType, Edge
from ..eccs.ate import ATECalculator
from ..eccs.edge_state_matrix import EdgeState
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class ECCSUser:
    """
    A class simulating a user of the ECCS system.

    The user knows the ground truth causal graph and responds to the system's suggestions
    accordingly.
    """

    ate_calculator = ATECalculator()

    def __init__(
        self,
        data: str | pd.DataFrame,
        true_graph: str | nx.DiGraph,
        test_graph: str | nx.DiGraph,
        treatment: str,
        outcome: str,
        fixed: Optional[list[Edge]] = None,
        banned: Optional[list[Edge]] = None,
    ) -> None:
        """
        Initializes the ECCSUser object.

        Parameters:
            data: The dataset or the path to it.
            true_graph: The ground truth causal graph or the path to it.
            test_graph: The starting graph available to the user or the path to it.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            fixed: An optional list of fixed edges at the start.
            banned: An optional list of banned edges at the start.
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
        self._true_ate = None
        self._current_ate = None

        # This is the index of steps where algorithms that might suggest multiple
        # edges in one invocation got invoked, for plotting and data analysis
        # purposes
        self._eccs_algorithm_invocation_iters = []

        self._invocations = 0
        self._ate_trajectory = [self.current_ate]
        self._edit_disance_trajectory = [self.current_graph_edit_distance]
        self._invocation_duration_trajectory = []
        self._fresh_edits_trajectory = []

        if fixed:
            for edge in fixed:
                if self._eccs.get_edge_state(*edge) == EdgeState.PRESENT:
                    self._eccs.fix_edge(*edge)
                else:
                    print(
                        f"Warning: Fixed edge {edge} is not present in the graph. It has state {EdgeState(state)}. Skipping."
                    )

        if banned:
            for edge in banned:
                state = self._eccs.get_edge_state(*edge)
                if state == EdgeState.ABSENT:
                    self._eccs.ban_edge(*edge)
                elif (
                    state != EdgeState.BANNED
                ):  # May be already banned because we fixed its reverse.
                    print(
                        f"Warning: Banned edge {edge} is not absent/banned from the graph. It has state {EdgeState(state)}. Skipping."
                    )

        print("Initialized ECCS user!")
        print(f"True ATE: {self.true_ate}")
        print(f"Initial ATE: {self.current_ate}")
        print(f"Initial ATE difference: {self.current_ate_diff}")
        print(f"Initial edit distance: {self.current_graph_edit_distance}")
        print(f"An optimal edit path: {self.current_optimal_edit_path}")

    @property
    def true_ate(self) -> float:
        """
        Returns the true ATE of the treatment on the outcome.

        Returns:
            The true ATE of the treatment on the outcome, or
            None if the graph does not contain a directed path from the treatment to the outcome.
        """
        if self._true_ate is None:
            if nx.has_path(self._true_graph, self._treatment, self._outcome):
                self._true_ate = self.ate_calculator.get_ate_and_confidence(
                    self._data, self._treatment, self._outcome, self._true_graph
                )["ATE"]
            else:
                self._true_ate = 0
        return self._true_ate

    @property
    def current_ate(self) -> float:
        """
        Returns the current ATE of the treatment on the outcome.

        Returns:
            The ATE of the treatment on the outcome based on the current graph, or
            None if the graph does not contain a directed path from the treatment to the outcome.
        """
        return self._eccs.get_ate()

    @property
    def current_ate_diff(self) -> float:
        """
        Returns the difference between the current ATE and the true ATE.

        Returns:
            The difference between the current ATE and the true ATE, or
            None if either graph does not contain a directed path from the treatment to the outcome.
        """
        return self.current_ate - self.true_ate

    @property
    def current_graph_edit_distance(self) -> int:
        """
        Returns the edit distance between the current graph and the true graph.

        Returns:
            The edit distance between the current graph and the true graph.
        """

        nodes = set(self._eccs.graph.nodes()).union(set(self._true_graph.nodes()))
        edit_distance = 0

        # Increment distance for every true edge that is currently missing
        for edge in self._true_graph.edges():
            if edge not in self._eccs.graph.edges():
                edit_distance += 1

        # Increment distance for every edge that is currently present but should not be,
        # unless its reverse is present in the true graph, in which case we already counted it above
        # (an edge flip counts as a single operation)
        for edge in self._eccs.graph.edges():
            if (
                edge not in self._true_graph.edges()
                and edge[::-1] not in self._true_graph.edges()
            ):
                edit_distance += 1

        return edit_distance

    @property
    def current_optimal_edit_path(self) -> list[tuple]:
        """
        Returns the optimal edit path between the current graph and the true graph.

        Returns:
            The optimal edit path between the current graph and the true graph.
        """

        changes = []

        for edge in self._true_graph.edges():
            if edge not in self._eccs.graph.edges():
                if edge[::-1] in self._eccs.graph.edges():
                    changes.append((edge[::-1], edge, EdgeEditType.FLIP))
                else:
                    changes.append((edge, None, EdgeEditType.ADD))

        for edge in self._eccs.graph.edges():
            if (
                edge not in self._true_graph.edges()
                and edge[::-1] not in self._true_graph.edges()
            ):
                changes.append((edge, None, EdgeEditType.REMOVE))

        return changes

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
    def ate_diff_trajectory(self) -> list[float]:
        """
        Returns the trajectory of the ATE difference over the invocations.

        Returns:
            The trajectory of the ATE difference over the invocations.
        """
        return [ate - self.true_ate for ate in self._ate_trajectory]

    @property
    def edit_distance_trajectory(self) -> list[int]:
        """
        Returns the trajectory of the graph edit distance over the invocations.

        Returns:
            The trajectory of the graph edit distance over the invocations.
        """
        return self._edit_disance_trajectory

    @property
    def invocation_duration_trajectory(self) -> list[float]:
        """
        Returns the trajectory of the invocation durations over the invocations.

        Returns:
            The trajectory of the invocation durations over the invocations.
        """
        return self._invocation_duration_trajectory

    @property
    def fresh_edits_trajectory(self) -> list[int]:
        """
        Returns the trajectory of the number of fresh edits per invocation over the invocations.
        Edits are fresh when the underlying algorithm was actually invoked to produce them (as
        opposed to serving them from a cache).

        Returns:
            The trajectory of the number of fresh edits per invocation over the invocations.
        """
        return self._fresh_edits_trajectory

    def invoke_eccs(
        self, method: str = None, budget: int = None, max_judgments: int = None
    ) -> tuple[bool, int]:
        """
        Invokes the ECCS system and updates the fixed and banned nodes accordingly.

        Parameters:
            method: The method to use for edge suggestions.
            budget: The budget for the invocation. Not all methods use this.
            max_judgments: The maximum number of edits to produce a judgment for. If None,
                the system will produce a judgment for all suggested edits provided by a single
                invocation of the ECCS system.

        Returns:
            A tuple containing whether any edits were suggested and how many fresh edits were produced
            during this invocation.
        """

        if (method is None) or (method not in self._eccs.EDGE_SUGGESTION_METHODS):
            method = self._eccs.EDGE_SUGGESTION_METHODS[0]

        # Get suggested modifications and selectively apply them
        start = datetime.now()
        edits, ate, num_fresh_edits = self._eccs.suggest(
            method, budget, max_results=max_judgments
        )
        end = datetime.now()
        self._invocation_duration_trajectory.append((end - start).total_seconds())
        self._fresh_edits_trajectory.append(num_fresh_edits)
        print(
            f"In iteration {self._invocations + 1} ECCS suggested: {edits} in {self._invocation_duration_trajectory[-1]} seconds."
        )
        if len(edits) == 0:
            return (False, 0)
        for src, dst, _ in edits:
            fadd = frem = ffix = fban = False
            radd = rrem = rfix = rban = False

            # The user produces the right judgement for the pair of (src, dst) regardless of the suggested edit type.
            if (src, dst) in self._true_graph.edges():
                fadd = self._eccs.add_edge(src, dst)
                ffix = self._eccs.fix_edge(src, dst)
                rrem = self._eccs.remove_edge(dst, src, remove_isolates=False)
                rban = self._eccs.ban_edge(dst, src)
            elif (dst, src) in self._true_graph.edges():
                frem = self._eccs.remove_edge(src, dst, remove_isolates=False)
                fban = self._eccs.ban_edge(src, dst)
                radd = self._eccs.add_edge(dst, src)
                rfix = self._eccs.fix_edge(dst, src)
            else:
                frem = self._eccs.remove_edge(src, dst, remove_isolates=False)
                fban = self._eccs.ban_edge(src, dst)
                rrem = self._eccs.remove_edge(dst, src, remove_isolates=False)
                rban = self._eccs.ban_edge(dst, src)

            self._current_ate = None

            print(
                f"\tUser judgement for edge {src} -> {dst}: "
                + ("Add " if fadd else "")
                + ("Remove " if frem else "")
                + ("Fix " if ffix else "")
                + ("Ban " if fban else "")
            )
            print(
                f"\tUser judgement for edge {dst} -> {src}: "
                + ("Add " if radd else "")
                + ("Remove " if rrem else "")
                + ("Fix " if rfix else "")
                + ("Ban " if rban else "")
            )

        # Update bookkeeping
        self._invocations += 1
        self._ate_trajectory.append(self.current_ate)
        self._edit_disance_trajectory.append(self.current_graph_edit_distance)

        print(f"\tUpdated ATE: {self.current_ate} (from {self.ate_trajectory[-2]})")
        print(
            f"\tUpdated ATE difference: {self.current_ate_diff} (from {self.ate_diff_trajectory[-2]})"
        )
        print(
            f"\tUpdated edit distance: {self.current_graph_edit_distance} (from {self.edit_distance_trajectory[-2]})"
        )
        return (True, num_fresh_edits)

    def run(self, steps: int, method: str = None, budget: int = None) -> None:
        """
        Simulate the user for `steps` steps. In each step, the user invokes the ECCS
        system and updates the fixed and banned edges accordingly.

        Parameters:
            steps: The number of steps that the user executes
            method: The method to use for edge suggestions.
            budget: The budget for each invocation. Not all methods use this.
        """

        for i in range(steps):
            print(f"Running iteration {i + 1}")
            suggested_edits, num_fresh_edits = self.invoke_eccs(
                method, budget, max_judgments=1
            )
            if num_fresh_edits > 0:
                self._eccs_algorithm_invocation_iters.append(i)
            if not suggested_edits:
                print(
                    "ECCS suggested no changes. Stopping. ",
                    "Total fresh edits produced over time: ",
                    sum(self._fresh_edits_trajectory),
                    "Total algorithm invocations: ",
                    len(self._eccs_algorithm_invocation_iters),
                    "Final edit distance: ",
                    self.current_graph_edit_distance,
                    "Final ATE: ",
                    self.current_ate,
                    "Final ATE difference: ",
                    self.current_ate_diff,
                )
                break
        print(
            "The specific ECCS Algorithm was invoked in the following steps: ",
            self._eccs_algorithm_invocation_iters,
        )
