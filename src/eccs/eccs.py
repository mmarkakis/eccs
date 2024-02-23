from typing import Optional, Tuple
import pandas as pd
import networkx as nx
from networkx.algorithms.d_separation import minimal_d_separator
from .edge_state_matrix import EdgeState, EdgeStateMatrix
from .graph_renderer import GraphRenderer
from .ate import ATECalculator
from itertools import combinations
from tqdm.auto import tqdm
from stqdm import stqdm


class EdgeChange:
    """
    Class to represent possible changes to a directed edge.
    """

    ADD: str = "Add"
    REMOVE: str = "Remove"
    FLIP: str = "Flip"


class ECCS:
    """
    A class for managing the Exposing Critical Causal Structures (ECCS) algorithm.
    """

    EDGE_SUGGESTION_METHODS = [
        "Best single edge change",
        "Best single adjustment set addition",
    ]

    def __init__(self, data_path: str, graph_path: Optional[str]):
        """
        Initialize the ECCS object.

        Parameters:
            data_path: The path to the data file.
            graph_path: Optionally, the path to the causal graph file in DOT format.
        """

        self._data_path = data_path
        self._graph_path = graph_path

        self._data = pd.read_csv(data_path)
        self._num_vars = self._data.shape[1]

        self._edge_decisions_matrix = EdgeStateMatrix(list(self._data.columns))

        self._graph = nx.DiGraph()
        if graph_path is not None:
            # Load the graph from a file in DOT format into a networkx DiGraph object
            graph = nx.DiGraph(nx.nx_pydot.read_dot(graph_path))
            for edge in graph.edges():
                self.add_edge(edge[0], edge[1])
        else:
            self._graph.add_nodes_from(range(self._num_vars))

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

    def _graph_is_acceptable(self) -> bool:
        """
        Check if self.graph is acceptable. A graph is acceptable if it satisfies the following conditions:
        - It is a directed acyclic graph.
        - It includes the treatment and outcome variables.
        - There is a directed path from the treatment to the outcome.
        - It includes no banned edges.
        - It includes all fixed edges.
        The conditions are checked in order of expense, so that the most expensive checks are only performed if the
        less expensive checks pass.

        Returns:
            True if the graph is acceptable, False otherwise.
        """
        is_acceptable = (
            self.treatment in self.graph.nodes  # It includes the treatment.
            and self.outcome in self.graph.nodes  # It includes the outcome.
            and all(  # It includes no banned edges.
                not self._edge_decisions_matrix.is_edge_banned(src, dst)
                for src, dst in self.graph.edges
            )
            and all(  # It includes all fixed edges.
                self.graph.has_edge(src, dst)
                for src, dst in self._edge_decisions_matrix.fixed_list
            )
            and nx.has_path(  # There is a directed path from the treatment to the outcome.
                self.graph, self.treatment, self.outcome
            )
            and nx.is_directed_acyclic_graph(
                self.graph
            )  # It is a directed acyclic graph.
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

    def add_edge(self, src: str, dst: str) -> None:
        """
        Add an edge to the graph and mark it as present.
        Can only add an edge if its current state is absent.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.ABSENT):
            self._graph.add_edge(src, dst)
            self._edge_decisions_matrix.mark_edge(src, dst, EdgeState.PRESENT)

    def remove_edge(self, src: str, dst: str, remove_isolates: bool = True) -> None:
        """
        Remove an edge from the graph and then optionally remove any nodes with degree zero.
        Can only remove an edge if its current state is present.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
            remove_isolates: Whether to remove any nodes with degree zero after removing the edge.
        """
        if self._edge_decisions_matrix.is_edge_in_state(src, dst, EdgeState.PRESENT):
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

    def suggest(self, method: str) -> tuple[float, str, pd.DataFrame]:
        """
        Suggest a modification to the graph that yields a maximally different ATE,
        compared to the current ATE. The modification should not edit edges that are
        currently fixed or banned. The method used for suggestion is specified by `method`.

        Parameters:
            method: The method to use for suggestion. Must be in ECCS.EDGE_SUGGESTION_METHODS.

        Returns:
            A tuple containing the suggested ATE, the suggested graph, and the suggested
                modification(s) as a dataframe.

        Raises:
            ValueError: If `method` is not in ECCS.EDGE_SUGGESTION_METHODS.
        """

        if method not in ECCS.EDGE_SUGGESTION_METHODS:
            raise ValueError(f"Invalid method: {method}")

        if method == "Best single edge change":
            return self._suggest_best_single_edge_change()
        elif method == "Best single adjustment set addition":
            return self._suggest_best_single_adjustment_set_addition()

    def _find_ate_diff_for_changes(
        self, changes: list[tuple[str, str, str]], base_ate: float, best_ate_diff: float
    ) -> Tuple[bool, float, Optional[float], Optional[str], Optional[pd.DataFrame]]:
        """
        Edit the graph according to the changes and evaluate the ATE difference.

        Parameters:
            changes: A list of tuples containing the changes to be made to the graph. Each tuple
                contains the source and destination of the edge and the change type.
            base_ate: The base ATE.
            best_ate_diff: The best ATE difference so far.

        Returns:
            A tuple containing:
                - A boolean indicating if the current ATE is the best so far.
                - The best ATE difference so far.
                - The best ATE so far, or None if the first element is False.
                - The best graph so far, or None if the first element is False.
                - The best modifications so far as a dataframe, or None if the first element is False.
        """

        # Edit graph
        for src, dst, change_type in changes:
            if change_type == EdgeChange.ADD:
                self.add_edge(src, dst)
            elif change_type == EdgeChange.REMOVE:
                self.remove_edge(src, dst, remove_isolates=True)
            elif change_type == EdgeChange.FLIP:
                self.remove_edge(src, dst, remove_isolates=False)
                self.add_edge(dst, src)

        # Check if the ATE is maximally changed
        is_current_best = False
        best_ate = None
        best_graph = None
        best_modifications = None
        if self._graph_is_acceptable():
            new_ate = self.get_ate()
            new_ate_diff = abs(new_ate - base_ate)
            if new_ate_diff > best_ate_diff:
                is_current_best = True
                best_ate_diff = new_ate_diff

                best_ate = new_ate
                best_graph = self.draw_graph()
                best_modifications = pd.DataFrame(
                    changes,
                    columns=["Source", "Destination", "Change"],
                )

        # Undo graph edits
        for src, dst, change_type in changes:
            if change_type == EdgeChange.ADD:
                self.remove_edge(src, dst)
            elif change_type == EdgeChange.REMOVE:
                self.add_edge(src, dst)
            elif change_type == EdgeChange.FLIP:
                self.remove_edge(dst, src, remove_isolates=False)
                self.add_edge(src, dst)

        return is_current_best, best_ate_diff, best_ate, best_graph, best_modifications

    def _suggest_best_single_edge_change(
        self,
    ) -> Tuple[float, str, pd.DataFrame]:
        """
        Suggest the best single edge change that maximally changes the ATE.

        Returns:
            A tuple containing the suggested ATE, the suggested graph, and the suggested
                modification(s) as a dataframe.
        """
        base_ate = self.get_ate()
        best_ate = self.get_ate()
        best_ate_diff = 0
        best_graph = self.draw_graph()
        best_modifications = pd.DataFrame(columns=["Source", "Destination", "Change"])

        # Check and update best
        def update_best(
            vals: Tuple[
                bool, float, Optional[float], Optional[str], Optional[pd.DataFrame]
            ]
        ) -> None:
            if vals[0]:
                best_ate_diff = vals[1]
                best_ate = vals[2]
                best_graph = vals[3]
                best_modifications = vals[4]

        # Iterate over all unordered pairs of variables using stqdm
        pairs = list(combinations(range(self._num_vars), 2))
        for i, j in stqdm(
            pairs,
            frontend=True,
            backend=True,
        ):

            # Extract edge endpoints and if none of the two are in the graph, skip
            e1 = self._data.columns[i]
            e2 = self._data.columns[j]
            if not self._graph.has_node(e1) and not self._graph.has_node(e2):
                continue
            
            # Extract edge states 
            f_state = self._edge_decisions_matrix.get_edge_state(e1, e2)
            r_state = self._edge_decisions_matrix.get_edge_state(e2, e1)

            # Apply the change and evaluate the ATE difference
            if f_state == EdgeState.ABSENT and r_state == EdgeState.ABSENT:
                # Try adding in the "forward" direction
                vals = self._find_ate_diff_for_changes(
                    [(e1, e2, EdgeChange.ADD)], base_ate, best_ate_diff
                )
                update_best(vals)
                # Try adding in the "reverse" direction
                vals = self._find_ate_diff_for_changes(
                    [(e2, e1, EdgeChange.ADD)], base_ate, best_ate_diff
                )
                update_best(vals)
            elif f_state == EdgeState.PRESENT:
                # Try removing the edge
                vals = self._find_ate_diff_for_changes(
                    [(e1, e2, EdgeChange.REMOVE)], base_ate, best_ate_diff
                )
                update_best(vals)
                if r_state == EdgeState.ABSENT:  # As opposed to banned
                    # Try flipping the edge
                    vals = self._find_ate_diff_for_changes(
                        [(e1, e2, EdgeChange.FLIP)], base_ate, best_ate_diff
                    )
                    update_best(vals)
            elif r_state == EdgeState.PRESENT:
                # Try removing the edge
                vals = self._find_ate_diff_for_changes(
                    [(e2, e1, EdgeChange.REMOVE)], base_ate, best_ate_diff
                )
                update_best(vals)
                if f_state == EdgeState.ABSENT:  # As opposed to banned
                    # Try flipping the edge
                    vals = self._find_ate_diff_for_changes(
                        [(e2, e1, EdgeChange.FLIP)], base_ate, best_ate_diff
                    )
                    update_best(vals)

        return best_ate, best_graph, best_modifications

    def _suggest_best_single_adjustment_set_addition(
        self,
    ) -> Tuple[float, nx.DiGraph, pd.DataFrame]:
        """
        Suggest the best single adjustment set addition that maximally changes the ATE.

        Returns:
            A tuple containing the suggested ATE, the suggested graph, and the suggested
                modification(s) as a dataframe.
        """
        base_ate = self.get_ate()
        best_ate = self.get_ate()
        best_ate_diff = 0
        best_graph = self._graph.copy()
        best_modifications = pd.DataFrame(columns=["Source", "Destination", "Change"])

        # Find a current minimal adjustment set
        # TODO: Can we guarantee that this set will be the same across calls?
        base_adj_set = nx.algorithms.minimal_d_separator(
            self._graph, self.treatment, self.outcome
        )

        vars_not_in_adj_set = [
            v
            for v in self.vars
            if v not in base_adj_set and v != self.treatment and v != self.outcome
        ]

        for v in stqdm(
            vars_not_in_adj_set,
            total=len(vars_not_in_adj_set),
            frontend=True,
            backend=True,
        ):
            (
                is_current_best,
                best_ate_diff,
                maybe_ate,
                maybe_graph,
                maybe_modifications,
            ) = self._find_ate_diff_for_changes(
                [(v, self.treatment, "Add"), (v, self.outcome, "Add")],
                base_ate,
                best_ate_diff,
            )

            if is_current_best:
                best_ate = maybe_ate
                best_graph = maybe_graph
                best_modifications = maybe_modifications

        return best_ate, best_graph, best_modifications
