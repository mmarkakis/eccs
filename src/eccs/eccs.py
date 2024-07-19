from __future__ import annotations
from typing import Any, List, Optional, Tuple, Callable
import pandas as pd
import networkx as nx
from .ate import ATECalculator
from .edge_state_matrix import EdgeState, EdgeStateMatrix
from .edges import Edge, EdgeEditType, EdgeEdit
from .graph_renderer import GraphRenderer
from .heuristic_search import AStarSearch
from .map_adj_set_to_graph import MapAdjSetToGraph

from itertools import combinations
import numpy as np


class ECCS:
    """
    A class for managing the Exposing Critical Causal Structures (ECCS) algorithm.
    """

    EDGE_SUGGESTION_METHODS = [
        "best_single_edge_change",
        "best_single_adjustment_set_change_opt",
        "best_single_adjustment_set_change",
        "astar_single_edge_change",
        "random_single_edge_change",
    ]
    A_STAR_NUM_SUGGESTIONS_PER_INVOCATION = 2

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
        self._edge_states = EdgeStateMatrix(list(self._data.columns))

        self.ate_calculator = ATECalculator()
        self._cached_edit_options = []
        self._cached_furthest_ate = 0
        self._cached_acceptance_test = None
        self._ate_cache = {}  # maps hashes of adjacency matrices to ate values

        # Load graph appropriately
        if isinstance(graph, str):
            graph = nx.DiGraph(nx.nx_pydot.read_dot(graph))

        self._graph = nx.DiGraph()
        self._graph.add_nodes_from((n, {"var_name": n}) for n in graph.nodes)
        for edge in graph.edges():
            self.add_edge(edge[0], edge[1])

        # Ban self edges.
        for i in range(self._num_vars):
            self.ban_edge(self._data.columns[i], self._data.columns[i])

        print("Initialized ECCS!")
        print(
            f"The graph has {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges."
        )
        num_fixed_edges = len(self._edge_states.fixed_list)
        num_banned_edges = len(self._edge_states.ban_list)
        print(
            f"Of the {self._graph.number_of_edges()} edges in the graph, {num_fixed_edges} are fixed."
        )
        print(
            f"Of the {self._num_vars**2 - self._graph.number_of_edges()} edges not in the graph, {num_banned_edges} are banned."
        )
        print(
            f"The number of modifiable edges is {self._num_vars**2 - num_fixed_edges - num_banned_edges}."
        )

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
        banlist = self._edge_states.ban_list
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
                not self._edge_states.is_edge_banned(src, dst)
                for src, dst in graph.edges
            )
            and all(  # It includes all fixed edges.
                graph.has_edge(src, dst) for src, dst in self._edge_states.fixed_list
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
        GraphRenderer.display_graph(self._graph, self._edge_states)

    def draw_graph(self) -> str:
        """
        Draw the current graph.

        Returns:
            A base64-encoded string representation of the graph.
        """
        return GraphRenderer.draw_graph(self._graph, self._edge_states)

    def save_graph(self, filename: str) -> None:
        """
        Save the current graph to a file.

        Parameters:
            filename: The name of the file to save to.
        """
        GraphRenderer.save_graph(self._graph, self._edge_states, filename)

    def add_edge(self, src: str, dst: str, is_suggested: bool = False) -> bool:
        """
        Add an edge to the graph and mark it as present, or as suggested if `is_suggested`
        is True. Can only add an edge if its current state is absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
            is_suggested: Whether the edge addition is suggested by the system, as opposed
                to being manually added by the user.

        Returns:
            True if the edge was indeed added, False otherwise.
        """
        if self._edge_states.is_edge_in_state(src, dst, EdgeState.ABSENT):
            self._graph.add_node(src, var_name=src)
            self._graph.add_node(dst, var_name=dst)
            self._graph.add_edge(src, dst)
            target_state = EdgeState.SUGGESTED if is_suggested else EdgeState.PRESENT
            self._edge_states.mark_edge(src, dst, target_state)
            return True
        return False

    def remove_edge(self, src: str, dst: str, remove_isolates: bool = False) -> bool:
        """
        Remove an edge from the graph and then optionally remove any nodes with degree zero.
        Can only remove an edge if its current state is present.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
            remove_isolates: Whether to remove any nodes with degree zero after removing the edge.

        Returns:
            True if the edge was indeed removed, False otherwise.
        """
        if self._edge_states.is_edge_in_state(
            src, dst, EdgeState.PRESENT
        ) or self._edge_states.is_edge_in_state(src, dst, EdgeState.SUGGESTED):
            self._graph.remove_edge(src, dst)
            if remove_isolates:
                self._graph.remove_nodes_from(list(nx.isolates(self._graph)))
            self._edge_states.mark_edge(src, dst, EdgeState.ABSENT)
            return True
        return False

    def get_edge_state(self, src: str, dst: str) -> EdgeState:
        """
        Get the state of an edge.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            The state of the edge.
        """
        return self._edge_states.get_edge_state(src, dst)

    def fix_edge(self, src: str, dst: str) -> bool:
        """
        Mark an edge as fixed.
        Can only fix an edge if its current state is present.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge was indeed fixed, False otherwise.
        """

        if self._edge_states.is_edge_in_state(src, dst, EdgeState.PRESENT):
            self._edge_states.mark_edge(src, dst, EdgeState.FIXED)
            return True
        return False

    def ban_edge(self, src: str, dst: str) -> bool:
        """
        Mark an edge as banned.
        Can only ban an edge if its current state is absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge was indeed banned, False otherwise.
        """
        if self._edge_states.is_edge_in_state(src, dst, EdgeState.ABSENT):
            self._edge_states.mark_edge(src, dst, EdgeState.BANNED)
            return True
        return False

    def unban_edge(self, src: str, dst: str) -> None:
        """
        Mark a banned edge as absent, as long as its reverse is not fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_states.is_edge_in_state(
            src, dst, EdgeState.BANNED
        ) and not self._edge_states.is_edge_in_state(dst, src, EdgeState.FIXED):
            self._edge_states.mark_edge(src, dst, EdgeState.ABSENT)

    def unfix_edge(self, src: str, dst: str) -> None:
        """
        Mark a fixed edge as present and its (previously banned) reverse as absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_states.is_edge_in_state(src, dst, EdgeState.FIXED):
            self._edge_states.mark_edge(src, dst, EdgeState.PRESENT)
            self._edge_states.mark_edge(dst, src, EdgeState.ABSENT)

    def is_edge_fixed(self, src: str, dst: str) -> bool:
        """
        Check if an edge is fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is fixed, False otherwise.
        """
        return self._edge_states.is_edge_in_state(src, dst, EdgeState.FIXED)

    def is_edge_banned(self, src: str, dst: str) -> bool:
        """
        Check if an edge is banned.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is banned, False otherwise.
        """
        return self._edge_states.is_edge_in_state(src, dst, EdgeState.BANNED)

    @staticmethod
    def hash_graph(graph: nx.DiGraph) -> int:
        """
        Hash a graph.

        Parameters:
            graph: The graph to hash.

        Returns:
            The hash of the graph.
        """
        adj_matrix = nx.adjacency_matrix(graph)
        return hash(
            (
                tuple(adj_matrix.data),
                tuple(adj_matrix.indices),
                tuple(adj_matrix.indptr),
            )
        )

    def get_ate(
        self,
        graph: Optional[nx.DiGraph] = None,
    ) -> float:
        """
        Calculate the average treatment effect (ATE) of `self._treatment` on `self._outcome` given `graph`.
        If graph is not provided, `self._graph` is used.

        Parameters:
            graph: The graph to use for the calculation.

        Returns:
            The ATE.
        """

        if graph is None:
            graph = self._graph
        graph_hash = ECCS.hash_graph(graph)
        if graph_hash not in self._ate_cache:
            self._ate_cache[graph_hash] = self.ate_calculator.get_ate_and_confidence(
                self.data, treatment=self._treatment, outcome=self._outcome, graph=graph
            )["ATE"]
        return self._ate_cache[graph_hash]

    def suggest(
        self, method: str, budget: Optional[int] = None, max_results: int = None
    ) -> tuple[list[EdgeEdit], float, int]:
        """
        Suggest a modification to the graph that yields a maximally different ATE,
        compared to the current ATE. The modification should not edit edges that are
        currently fixed or banned. The method used for suggestion is specified by `method`.

        Parameters:
            method: The method to use for suggestion. Must be in ECCS.EDGE_SUGGESTION_METHODS.
            budget: The budget for finding a suggestion. Not all methods use this.
            max_results: The maximum number of edits to return. The rest, if any, may be cached.
                If None, all suggested edits are returned.

        Returns:
          A tuple containing a list of the suggested edge edit(s), the resulting ATE and,
          if the underlying algorithm was invoked anew, the total number of edits it produced.

        Raises:
            ValueError: If `method` is not in ECCS.EDGE_SUGGESTION_METHODS.
        """

        if method not in ECCS.EDGE_SUGGESTION_METHODS:
            raise ValueError(f"Invalid method: {method}")

        if method == "best_single_edge_change":
            return self.suggest_best_single_edge_change()
        elif method == "best_single_adjustment_set_change":
            return self.suggest_best_single_adjustment_set_change(
                max_results=max_results, use_optimized=False
            )
        elif method == "best_single_adjustment_set_change_opt":
            return self.suggest_best_single_adjustment_set_change(
                max_results=max_results, use_optimized=True
            )
        elif method == "random_single_edge_change":
            return self.suggest_random_single_edge_change()
        elif method == "astar_single_edge_change":
            return self.suggest_best_single_edge_change_heuristic(
                budget, max_results=max_results
            )

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
            print("Applying edit: ", src, dst, edit_type)
            if edit_type == EdgeEditType.ADD:
                if not graph.has_edge(src, dst):
                    graph.add_edge(src, dst)
            elif edit_type == EdgeEditType.REMOVE:
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)
            elif edit_type == EdgeEditType.FLIP:
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)
                if not graph.has_edge(dst, src):
                    graph.add_edge(dst, src)

        print("Applied edits successfully")

        # Compute the ATE if the graph is acceptable
        if self._is_acceptable(graph):
            print("Graph is acceptable after edits: ", edits)
            ate = self.get_ate(graph)
            print("Got back ATE: ", ate)
            return ate

        print("Graph is not acceptable after edits: ", edits)
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
        edge_decisions_matrix = self._edge_states.copy()

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

    def suggest_best_single_edge_change(
        self,
    ) -> Tuple[list[EdgeEdit], float, int]:
        """
        Suggest the best_single_edge_change that maximally changes the ATE.

        Returns:
            A tuple containing a list of the suggested edge edit(s), the resulting ATE and,
                if the underlying algorithm was invoked anew, the total number of edits it produced.
        """
        base_ate = self.get_ate()
        furthest_ate = base_ate
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
            f_state = self.get_edge_state(e1, e2)
            r_state = self.get_edge_state(e2, e1)

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

        return (best_edits, furthest_ate, 1)

    @staticmethod
    def _pop_n(l: list[Any], n: int) -> list[Any]:
        """
        Pop `n` elements from the start of list `l` and return them as a list.

        Parameters:
            l: The list to pop from.
            n: The number of elements to pop.

        Returns:
            The popped elements.
        """
        elements = l[:n]
        l[:] = l[n:]
        return elements

    def suggest_best_single_edge_change_heuristic(
        self, budget: Optional[int] = None, max_results: int = None
    ) -> Tuple[list[EdgeEdit], float, int]:
        """
        Suggest the best single edge change based on A star

        Parameters:
            budget: The budget for the search.
            max_results: The maximum number of edits to return. The rest, if any, are cached.
                If None, all suggested edits are returned.

        Returns:
            A tuple containing a list of the suggested edge edit(s), the resulting ATE and,
                if the underlying algorithm was invoked anew, the total number of edits it produced.
        """
        if len(self._cached_edit_options) > 0:
            edits = ECCS._pop_n(self._cached_edit_options, max_results)
            return (edits, self._edit_and_get_ate(edits), 0)
        a_star = AStarSearch(
            self._graph,
            self._treatment,
            self._outcome,
            self._data,
            self._edge_states,
            computational_budget=budget,
        )
        edits = a_star.astar()
        self._cached_edit_options = edits[: self.A_STAR_NUM_SUGGESTIONS_PER_INVOCATION]
        edits = ECCS._pop_n(self._cached_edit_options, max_results)
        res = (
            edits,
            self._edit_and_get_ate(edits),
            self.A_STAR_NUM_SUGGESTIONS_PER_INVOCATION,
        )

        # return ([edit], self._edit_and_get_ate([edit]), True) # TODO: FIXME
        return res

    @staticmethod
    def _find_adjustment_set(
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """
        Find an adjustment set between the treatment and outcome variables in the graph.
        We achieve this by only leaving backdoor paths in the graph, by temporarily removing
        all edges that are directed out from the treatment variable.

        Parameters:
            graph: The graph to search.
            treatment: The treatment variable.
            outcome: The outcome variable.

        Returns:
            The adjustment set, or None if no adjustment set exists.
        """
        temp_removed_edges = list(graph.out_edges(treatment))
        for edge in temp_removed_edges:
            graph.remove_edge(edge[0], edge[1])
        adjset = nx.algorithms.find_minimal_d_separator(graph, treatment, outcome)
        for edge in temp_removed_edges:
            graph.add_edge(edge[0], edge[1])
        return adjset

    def suggest_best_single_adjustment_set_change(
        self, max_results: int = None, use_optimized: bool = True
    ) -> Tuple[list[EdgeEdit], float, int]:
        """
        Suggest the best_single_adjustment_set_changes that maximally changes the ATE.

        Parameters:
            max_results: The maximum number of edits to return. The rest, if any, are cached.
                If None, all suggested edits are returned.
            use_optimized: Whether to use the optimized version of the algorithm.

        Returns:
            A tuple containing a list of the suggested edge edit(s), the resulting ATE and,
                if the underlying algorithm was invoked anew, the total number of edits it produced.
        """
        print("Computing and suggesting best single adjustment set change")
        print(
            f"The current cache size is {len(self._cached_edit_options)} and the acceptance test returns {self._cached_acceptance_test}"
        )

        if len(self._cached_edit_options) > 0 and self._cached_acceptance_test():
            print(
                "Serving previous edge suggestion(s) based on best single adjustment set change"
            )
            edits = ECCS._pop_n(self._cached_edit_options, max_results)
            self._cached_acceptance_test = self._check_if_edits_were_accepted(edits)
            return (edits, self._cached_furthest_ate, 0)

        ranking = self.get_adj_set_changes_ranking(use_optimized=use_optimized, k=1)

        print("Done evaluating options")
        if len(ranking) == 0:
            return ([], self.get_ate(), 0)

        _, best_edits, furthest_ate, _ = ranking[0]
        num_best_edits = len(best_edits)

        if num_best_edits == 0:
            return ([], furthest_ate, 0)
        elif (max_results is None) or (
            num_best_edits <= max_results
        ):  # No need to cache edits
            self._cached_furthest_ate = furthest_ate
            self._cached_edit_options = []
            self._cached_acceptance_test = None
            return (best_edits, furthest_ate, num_best_edits)
        else:  # Must cache some edits
            self._cached_furthest_ate = furthest_ate
            self._cached_edit_options = best_edits
            edits_to_return = ECCS._pop_n(self._cached_edit_options, max_results)
            self._cached_acceptance_test = self._check_if_edits_were_accepted(
                edits_to_return
            )
            return (edits_to_return, furthest_ate, num_best_edits)

    def get_adj_set_changes_ranking(
        self,
        use_optimized: bool = True,
        k: Optional[int] = None,
        min_ate_ratio: Optional[float] = None,
    ) -> list[tuple[str, list[EdgeEdit], float]]:
        """
        Get a ranking of single-variable adjustment set changes, based on their
        impact on the ATE. Can filter by number of variables to return (`k`) and/or
        by the minimum relative change in ATE.

        Parameters:
            use_optimized: Whether to use the optimized version of the algorithm.
            k: The maximum number of changes to return. If None, all are returned.
            min_ate_ratio: The minimum relative change in ATE to return. If None, all are returned.

        Returns:
            A list of tuples, each containing the variable that was changed, the corresponding edge edit(s),
                and the resulting ATE.
        """
        base_ate = self.get_ate()

        ranking = []
        # Each element has (variable, edits, ate, ate_ratio)

        def maybe_update_ranking(v, edits, ate):
            nonlocal ranking

            # If the ate couldn't be computed, return
            if ate is None:
                return

            # If the achieved ate ratio is below cutoff, return
            ate_ratio = abs((ate - base_ate) / base_ate)
            if (min_ate_ratio is not None) and (ate_ratio < min_ate_ratio):
                return

            # If the ranking is full and the achieved ate ratio is below the
            # worst in the ranking, return
            if (
                (k is not None)
                and (len(ranking) == k)
                and (ate_ratio <= ranking[-1][3])
            ):
                return

            # Insert the new element in the right position in the ranking and trim
            # down to desired length if needed.
            inserted = False
            for i, (_, _, _, r) in enumerate(ranking):
                if ate_ratio > r:
                    ranking.insert(i, (v, edits, ate, ate_ratio))
                    inserted = True
                    break
            if not inserted:
                ranking.append((v, edits, ate, ate_ratio))

            if (k is not None) and (len(ranking) > k):
                ranking.pop()

        base_adj_set = ECCS._find_adjustment_set(
            self._graph, self.treatment, self.outcome
        )
        print("Found base adjustment set: ", base_adj_set)
        vars_not_in_adj_set = [
            v
            for v in self.vars
            if v not in base_adj_set and v != self.treatment and v != self.outcome
        ]

        mapper = MapAdjSetToGraph(
            self.graph,
            self.treatment,
            self.outcome,
            self._edge_states.fixed_list,
            self._edge_states.ban_list,
            base_adj_set,
        )

        # Try adding each of the addable
        for v in vars_not_in_adj_set:
            print(f"Trying to add {v} to the adjustment set")
            edits = mapper.map_addition(v, use_optimized)
            print("Got back edits for addition: ", edits)
            ate = self._edit_and_get_ate(edits)
            maybe_update_ranking(v, edits, ate)

        # Try removing each of the removable
        for v in base_adj_set:
            print(f"Trying to remove {v} from the adjustment set")
            edits = mapper.map_removal(v, use_optimized)
            print("Got back edit lists for removal: ", edits)
            ate = self._edit_and_get_ate(edits)
            maybe_update_ranking(v, edits, ate)

        return ranking

    def _check_if_edits_were_accepted(
        self, edits: list[EdgeEdit]
    ) -> Callable[[], bool]:
        """
        Given a list of edits, return a lambda that will evaluate True if those edits could be the most
        recent edits to have been applied to self._graph.

        Parameters:
            edits: The edits to evaluate.

        Returns:
            A lambda that will evaluate true if the edits could be the most
            recent edits to have been applied to self._graph.
        """
        lambdas = []
        for src, dst, edit_type in edits:
            if edit_type == EdgeEditType.ADD:
                lambdas.append(lambda: self.graph.has_edge(src, dst))
            elif edit_type == EdgeEditType.REMOVE:
                lambdas.append(lambda: (not self.graph.has_edge(src, dst)))
            elif edit_type == EdgeEditType.FLIP:
                lambdas.append(
                    lambda: (
                        self.graph.has_edge(dst, src)
                        and not self.graph.has_edge(src, dst)
                    )
                )

        return lambda: all([l() for l in lambdas])

    def suggest_random_single_edge_change(
        self,
    ) -> Tuple[list[EdgeEdit], float, int]:
        """
        Suggest a random_single_edge_change.

        Returns:
            A tuple containing a list of the suggested edge edit(s), the resulting ATE and,
            if the underlying algorithm was invoked anew, the total number of edits it produced.
        """

        # Derive the set of editable edges
        eligible_pairs = list(combinations(range(self._num_vars), 2))
        eligible_edges = [
            (self._data.columns[i], self._data.columns[j]) for i, j in eligible_pairs
        ]
        eligible_edges = [
            (src, dst)
            for src, dst in eligible_edges
            if src != dst  # Don't toggle self-edges
            and not self.is_edge_fixed(src, dst)  # Don't touch fixed edges
            and not self.is_edge_fixed(
                dst, src
            )  # No point toggling if reverse is fixed.
            and not self.is_edge_banned(src, dst)  # Don't touch banned edges
        ]

        while len(eligible_edges) > 0:
            e1, e2 = eligible_edges.pop(np.random.randint(len(eligible_edges)))

            edit = ()
            if self.get_edge_state(e1, e2) == EdgeState.ABSENT:
                edit = EdgeEdit(e1, e2, EdgeEditType.ADD)
            else:
                edit = EdgeEdit(e1, e2, EdgeEditType.REMOVE)

            ate = self._edit_and_get_ate([edit])

            if ate == None:
                continue

            return ([edit], ate, 1)
