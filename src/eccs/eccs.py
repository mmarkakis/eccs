from __future__ import annotations
from typing import Any, List, Optional, Tuple
import pandas as pd
import networkx as nx
from networkx.algorithms.d_separation import minimal_d_separator
from .ate import ATECalculator
from .edge_state_matrix import EdgeState, EdgeStateMatrix
from .edits import EdgeEditType, EdgeEdit
from .graph_renderer import GraphRenderer
from .heuristic_search import AStarSearch

from itertools import combinations
from tqdm.auto import tqdm
import multiprocessing
import numpy as np


class ECCS:
    """
    A class for managing the Exposing Critical Causal Structures (ECCS) algorithm.
    """

    EDGE_SUGGESTION_METHODS = [
        "best_single_edge_change",
        "best_single_adjustment_set_change",
        "astar_single_edge_change",
        "random_single_edge_change",
    ]

    def __init__(self, data: str | pd.DataFrame, graph: str | nx.DiGraph):
        """
        Initialize the ECCS object.

        Parameters:
            data: The dataset or the path to it.
            graph: The causal DAG, or a path to a file containing the graph in DOT format.
        """

        # Load data appropriately
        if isinstance(data, str):
            self._data = pd.read_csv(data)
        else:
            self._data = data

        self._num_vars = self._data.shape[1]
        self._edge_decisions_matrix = EdgeStateMatrix(list(self._data.columns))

        # Load graph appropriately
        if isinstance(graph, str):
            self._graph = nx.DiGraph(nx.nx_pydot.read_dot(graph))
        else:
            self._graph = graph
        graph.add_nodes_from((n, {"var_name": n}) for n in graph.nodes)
        for edge in graph.edges():
            self.add_edge(edge[0], edge[1])

        # Ban self edges.
        for i in range(self._num_vars):
            self.ban_edge(self._data.columns[i], self._data.columns[i])

    def set_treatment(self, treatment: str) -> None:
        """
        Set the treatment variable.

        Parameters:
            treatment: The name of the treatment variable.
        """
        self._treatment = treatment
        self._treatment_idx = self._data.columns.get_loc(treatment)

    def set_outcome(self, outcome: str) -> None:
        """
        Set the outcome variable.

        Parameters:
            outcome: The name of the outcome variable.
        """
        self._outcome = outcome
        self._outcome_idx = self._data.columns.get_loc(outcome)

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the data.
        """
        return self._data

    @property
    def banlist_df(self) -> pd.DataFrame:
        """
        Returns the banlist as a dataframe, except for self edges
        """
        banlist = self._edge_decisions_matrix.ban_list
        banlist_df = pd.DataFrame(banlist, columns=["Source", "Destination"])
        banlist_df = banlist_df[banlist_df["Source"] != banlist_df["Destination"]]
        return banlist_df

    @property
    def treatment(self) -> str:
        """
        Returns the treatment variable.
        """
        return self._treatment

    @property
    def outcome(self) -> str:
        """
        Returns the outcome variable.
        """
        return self._outcome

    @property
    def vars(self) -> list[str]:
        """
        Returns the list of variables in the data.
        """
        return list(self._data.columns)

    @property
    def num_vars(self) -> int:
        """
        Calculate the number of variables in the data.

        Returns:
            The number of variables in the data.
        """
        return len(self.vars)

    @property
    def graph(self) -> nx.DiGraph:
        """
        Returns the graph.
        """
        return self._graph

    def _is_acceptable(self, graph: Optional[nx.DiGraph]) -> bool:
        """
        Check if graph is acceptable. A graph is acceptable if it satisfies the following conditions:
        - It is a directed acyclic graph.
        - It includes the treatment and outcome variables.
        - There is a directed path from the treatment to the outcome.
        - It includes no banned edges.
        - It includes all fixed edges.
        The conditions are checked in order of expense, so that the most expensive checks are only performed if the
        less expensive checks pass.

        Parameters:
            graph: The graph to check. If None, self.graph is checked.

        Returns:
            True if the graph is acceptable, False otherwise.
        """
        if graph is None:
            graph = self.graph

        is_acceptable = (
            self.treatment in graph.nodes  # It includes the treatment.
            and self.outcome in graph.nodes  # It includes the outcome.
            and all(  # It includes no banned edges.
                not self._edge_decisions_matrix.is_edge_banned(src, dst)
                for src, dst in graph.edges
            )
            and all(  # It includes all fixed edges.
                graph.has_edge(src, dst)
                for src, dst in self._edge_decisions_matrix.fixed_list
            )
            and nx.has_path(  # There is a directed path from the treatment to the outcome.
                graph, self.treatment, self.outcome
            )
            and nx.is_directed_acyclic_graph(graph)  # It is a directed acyclic graph.
        )

        return is_acceptable

    def clear_graph(self, clear_edge_states: bool = True) -> None:
        """
        Clear the graph and possibly edge states.

        Parameters:
            clear_edge_states: Whether to also clear the edge states.
        """
        self._graph = nx.DiGraph()
        if clear_edge_states:
            self._edge_states = EdgeStateMatrix(list(self._data.columns))

    def display_graph(self) -> None:
        """
        Display the current graph.
        """
        GraphRenderer.display_graph(self._graph, self._edge_decisions_matrix)

    def draw_graph(self) -> str:
        """
        Draw the current graph.

        Returns:
            A base64-encoded string representation of the graph.
        """
        return GraphRenderer.draw_graph(self._graph, self._edge_decisions_matrix)

    def save_graph(self, filename: str) -> None:
        """
        Save the current graph to a file.

        Parameters:
            filename: The name of the file to save to.
        """
        GraphRenderer.save_graph(self._graph, self._edge_decisions_matrix, filename)

    def add_edge(self, src: str, dst: str, is_suggested: bool = False) -> None:
        """
        Add an edge to the graph and mark it as present, or as suggested if `is_suggested`
        is True. Can only add an edge if its current state is absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
            is_suggested: Whether the edge addition is suggested by the system, as opposed
                to being manually added by the user.
        """
        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.ABSENT):
            self._graph.add_node(src, var_name=src)
            self._graph.add_node(dst, var_name=dst)
            self._graph.add_edge(src, dst)
            target_state = EdgeState.SUGGESTED if is_suggested else EdgeState.PRESENT
            self._edge_decisions_matrix.mark_edge(src, dst, target_state)

    def remove_edge(self, src: str, dst: str, remove_isolates: bool = True) -> None:
        """
        Remove an edge from the graph and then optionally remove any nodes with degree zero.
        Can only remove an edge if its current state is present.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
            remove_isolates: Whether to remove any nodes with degree zero after removing the edge.
        """
        if self._edge_decisions_matrix.is_edge_in_state(
            src, dst, EdgeState.PRESENT
        ) or self._edge_decisions_matrix.is_edge_in_state(
            src, dst, EdgeState.SUGGESTED
        ):
            self._graph.remove_edge(src, dst)
            if remove_isolates:
                self._graph.remove_nodes_from(list(nx.isolates(self._graph)))
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.ABSENT)

    def fix_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as fixed and mark its reverse as banned.
        Can only fix an edge if its current state is present.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """

        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.PRESENT):
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.FIXED)
            self._edge_decisions_matrix.mark_edge(dst, src, EdgeState.BANNED)

    def ban_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as banned.
        Can only ban an edge if its current state is absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.ABSENT):
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.BANNED)

    def unban_edge(self, src: str, dst: str) -> None:
        """
        Mark a banned edge as absent, as long as its reverse is not fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.is_edge_in_state(
            src, dst, EdgeState.BANNED
        ) and not self._edge_decisions_matrix.is_edge_in_state(
            dst, src, EdgeState.FIXED
        ):
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.ABSENT)

    def unfix_edge(self, src: str, dst: str) -> None:
        """
        Mark a fixed edge as present and its (previously banned) reverse as absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.FIXED):
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.PRESENT)
            self._edge_decisions_matrix.mark_edge(dst, src, EdgeState.ABSENT)

    def is_edge_fixed(self, src: str, dst: str) -> bool:
        """
        Check if an edge is fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is fixed, False otherwise.
        """
        return self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.FIXED)

    def is_edge_banned(self, src: str, dst: str) -> bool:
        """
        Check if an edge is banned.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is banned, False otherwise.
        """
        return self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.BANNED)

    def get_ate(
        self,
        graph: Optional[nx.DiGraph] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> float:
        """
        Calculate the average treatment effect (ATE) of `treatment` on `outcome` given `graph`.
        If any of these parameters are not provided, the corresponding instance variables are used.

        Parameters:
            graph: The graph to use for the calculation.
            treatment: The treatment variable.
            outcome: The outcome variable.

        Returns:
            The ATE.
        """

        if graph is None:
            graph = self._graph
        if treatment is None:
            treatment = self._treatment
        if outcome is None:
            outcome = self._outcome

        return ATECalculator.get_ate_and_confidence(
            self.data, treatment=treatment, outcome=outcome, graph=graph
        )["ATE"]

    def suggest(self, method: str) -> tuple[list[EdgeEdit], float]:
        """
        Suggest a modification to the graph that yields a maximally different ATE,
        compared to the current ATE. The modification should not edit edges that are
        currently fixed or banned. The method used for suggestion is specified by `method`.

        Parameters:
            method: The method to use for suggestion. Must be in ECCS.EDGE_SUGGESTION_METHODS.

        Returns:
          A tuple containing a list of the suggested edge edit(s) and the resulting ATE.

        Raises:
            ValueError: If `method` is not in ECCS.EDGE_SUGGESTION_METHODS.
        """

        if method not in ECCS.EDGE_SUGGESTION_METHODS:
            raise ValueError(f"Invalid method: {method}")

        if method == "best_single_edge_change":
            return self._suggest_best_single_edge_change()
        elif method == "best_single_adjustment_set_change":
            return self._suggest_best_single_adjustment_set_addition()
        elif method == "random_single_edge_change":
            return self._suggest_random_single_edge_change()
        elif method == "astar_single_edge_change":
            return self._suggest_best_single_edge_change_heuristic()

    def _edit_and_get_ate(self, edits: list[EdgeEdit]) -> Optional[float]:
        """
        Edit a copy of the current self.graph according to `edits` and evaluate the ATE.
        Does not modify self.graph.

        Parameters:
            edits: A list of edits to be made to the graph.

        Returns:
            The ATE, or None if the graph is not acceptable.
        """

        graph = self._graph.copy()

        # Edit graph
        for src, dst, edit_type in edits:
            if edit_type == EdgeEditType.ADD:
                graph.add_edge(src, dst)
            elif edit_type == EdgeEditType.REMOVE:
                graph.remove_edge(src, dst)
            elif edit_type == EdgeEditType.FLIP:
                graph.remove_edge(src, dst)
                graph.add_edge(dst, src)

        # Compute the ATE if the graph is acceptable
        if self._is_acceptable(graph):
            return self.get_ate(graph, self.treatment, self.outcome)

        return None

    def _edit_and_draw(self, edits: list[EdgeEdit]) -> Optional[str]:
        """
        Edit a copy of the current self.graph according to `edits` and draw the graph.
        Parameters:
            edits: A list of edits to be made to the graph.

        Returns:
            A base64-encoded string representation of the graph, or None if the graph is not acceptable.
        """

        graph = self._graph.copy()
        edge_decisions_matrix = self._edge_decisions_matrix.copy()

        # Edit graph
        for src, dst, edit_type in edits:
            if edit_type == EdgeEditType.ADD:
                graph.add_edge(src, dst)
                edge_decisions_matrix.mark_edge(src, dst, EdgeState.SUGGESTED)
            elif edit_type == EdgeEditType.REMOVE:
                graph.remove_edge(src, dst)
                edge_decisions_matrix.mark_edge(src, dst, EdgeState.ABSENT)
            elif edit_type == EdgeEditType.FLIP:
                graph.remove_edge(src, dst)
                edge_decisions_matrix.mark_edge(src, dst, EdgeState.ABSENT)
                graph.add_edge(dst, src)
                edge_decisions_matrix.mark_edge(dst, src, EdgeState.SUGGESTED)

        # Draw the graph if the graph is acceptable
        if not self._is_acceptable(graph):
            return None

        return GraphRenderer.draw_graph(graph, edge_decisions_matrix)

    def _suggest_best_single_edge_change(
        self,
    ) -> Tuple[list[EdgeEdit], float]:
        """
        Suggest the best_single_edge_change that maximally changes the ATE.

        Returns:
            A tuple containing a list of the suggested edge edit(s) and the resulting ATE.
        """
        base_ate = self.get_ate()
        furthest_ate = self.get_ate()
        best_ate_diff = 0
        best_edits = []

        def maybe_update_best(ate, edits):
            nonlocal best_ate_diff
            nonlocal furthest_ate
            nonlocal best_edits

            if ate is None:
                return
            ate_diff = abs(ate - base_ate)
            if ate_diff > best_ate_diff:
                best_ate_diff = ate_diff
                furthest_ate = ate
                best_edits = edits

        pairs = list(combinations(range(self._num_vars), 2))

        # Iterate over all unordered pairs of variables and compute ates
        for i, j in pairs:

            # Extract edge endpoints and if none of the two are in the graph, skip
            e1 = self._data.columns[i]
            e2 = self._data.columns[j]

            # Extract edge states
            f_state = self._edge_decisions_matrix.get_edge_state(e1, e2)
            r_state = self._edge_decisions_matrix.get_edge_state(e2, e1)

            # Apply the change and evaluate the ATE
            if f_state == EdgeState.ABSENT and (
                r_state == EdgeState.ABSENT or r_state == EdgeState.BANNED
            ):
                # Try adding in the "forward" direction
                ate = self._edit_and_get_ate([EdgeEdit(e1, e2, EdgeEditType.ADD)])
                maybe_update_best(ate, [EdgeEdit(e1, e2, EdgeEditType.ADD)])
            if r_state == EdgeState.ABSENT and (
                f_state == EdgeState.ABSENT or f_state == EdgeState.BANNED
            ):
                # Try adding in the "reverse" direction
                ate = self._edit_and_get_ate([EdgeEdit(e2, e1, EdgeEditType.ADD)])
                maybe_update_best(ate, [EdgeEdit(e2, e1, EdgeEditType.ADD)])
            if f_state == EdgeState.PRESENT:
                # Try removing the edge
                ate = self._edit_and_get_ate([EdgeEdit(e1, e2, EdgeEditType.REMOVE)])
                maybe_update_best(ate, [EdgeEdit(e1, e2, EdgeEditType.REMOVE)])
                if r_state == EdgeState.ABSENT:  # As opposed to banned
                    # Try flipping the edge
                    ate = self._edit_and_get_ate([EdgeEdit(e1, e2, EdgeEditType.FLIP)])
                    maybe_update_best(ate, [EdgeEdit(e1, e2, EdgeEditType.FLIP)])
            if r_state == EdgeState.PRESENT:
                # Try removing the edge
                ate = self._edit_and_get_ate([EdgeEdit(e2, e1, EdgeEditType.REMOVE)])
                maybe_update_best(ate, [EdgeEdit(e2, e1, EdgeEditType.REMOVE)])
                if f_state == EdgeState.ABSENT:  # As opposed to banned
                    # Try flipping the edge
                    ate = self._edit_and_get_ate([EdgeEdit(e2, e1, EdgeEditType.FLIP)])
                    maybe_update_best(ate, [EdgeEdit(e2, e1, EdgeEditType.FLIP)])

        return (best_edits, furthest_ate)

    def _suggest_best_single_edge_change_heuristic(
        self,
    ) -> Tuple[list[EdgeEdit], float]:
        """
        Suggest the best single edge change based on A star

        Returns:
            A tuple containing a list of the suggested edge edit(s) and the resulting ATE.
        """
        a_star = AStarSearch(
            self._graph, self._treatment_idx, self._outcome_idx.self._data
        )
        return a_star.astar()  ## TODO: Enforce correct return type

    def _suggest_best_single_adjustment_set_change(
        self,
    ) -> Tuple[list[EdgeEdit], float]:
        """
        Suggest the best_single_adjustment_set_changes that maximally changes the ATE.

        Changes are translated to edge changes in a rudimentary manner.

        Returns:
            A tuple containing a list of the suggested edge edit(s) and the resulting ATE.
        """
        base_ate = self.get_ate()
        furthest_ate = self.get_ate()
        best_ate_diff = 0
        best_edits = []

        def maybe_update_best(ate, edits):
            nonlocal best_ate_diff
            nonlocal furthest_ate
            nonlocal best_edits

            if ate is None:
                return
            ate_diff = abs(ate - base_ate)
            if ate_diff > best_ate_diff:
                best_ate_diff = ate_diff
                furthest_ate = ate
                best_edits = edits

        base_adj_set = nx.algorithms.minimal_d_separator(
            self._graph, self.treatment, self.outcome
        )
        vars_not_in_adj_set = [
            v
            for v in self.vars
            if v not in base_adj_set and v != self.treatment and v != self.outcome
        ]

        # Try adding each of the addable
        for v in vars_not_in_adj_set:
            edits = [
                EdgeEdit(v, self.treatment, EdgeEditType.ADD),
                EdgeEdit(v, self.outcome, EdgeEditType.ADD),
            ]
            ate = self._edit_and_get_ate(edits, base_ate, best_ate_diff)
            maybe_update_best(ate, edits)

        # Try removing each of the removable
        for v in base_adj_set:
            edits = [
                EdgeEdit(self.treatment, v, EdgeEditType.ADD),
            ]
            ate = self._edit_and_get_ate(edits, base_ate, best_ate_diff)
            maybe_update_best(ate, edits)

        return (best_edits, furthest_ate)

    def _suggest_random_single_edge_change(
        self,
    ) -> Tuple[list[EdgeEdit], float]:
        """
        Suggest a random_single_edge_change.

        Returns:
            A tuple containing a list of the suggested edge edit(s) and the resulting ATE.
        """

        while True:
            # Pick edge endpoints at random
            i = np.random.randint(self._num_vars)
            j = np.random.randint(self._num_vars)

            if i == j:
                continue

            # Extract edge endpoints
            e1 = self._data.columns[i]
            e2 = self._data.columns[j]

            # Extract edge states
            f_state = self._edge_decisions_matrix.get_edge_state(e1, e2)
            r_state = self._edge_decisions_matrix.get_edge_state(e2, e1)

            if (
                f_state == EdgeState.FIXED
                or r_state == EdgeState.FIXED
                or (f_state == EdgeState.BANNED and r_state == EdgeState.BANNED)
            ):
                continue

            edit = ()

            if f_state == EdgeState.BANNED:
                # Toggle the state of the inverse edge
                if r_state == EdgeState.ABSENT:
                    edit = EdgeEdit(e2, e1, EdgeEditType.ADD)
                else:
                    edit = EdgeEdit(e2, e1, EdgeEditType.REMOVE)

            if r_state == EdgeState.BANNED:
                # Toggle the state of the forward edge
                if f_state == EdgeState.ABSENT:
                    edit = EdgeEdit(e1, e2, EdgeEditType.ADD)
                else:
                    edit = EdgeEdit(e1, e2, EdgeEditType.REMOVE)

            # At this point neither is fixed and neither is banned.
            if f_state == EdgeState.PRESENT:
                edit = EdgeEdit(e1, e2, EdgeEditType.REMOVE)
            elif r_state == EdgeState.PRESENT:
                edit = EdgeEdit(e2, e1, EdgeEditType.REMOVE)
            elif np.random.choice([True, False]):
                edit = EdgeEdit(e1, e2, EdgeEditType.ADD)
            else:
                edit = EdgeEdit(e2, e1, EdgeEditType.ADD)

            ate = self._edit_and_get_ate([edit])

            if ate == None:
                continue

            return (
                [edit],
                ate,
            )
