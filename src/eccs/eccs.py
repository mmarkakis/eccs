from typing import Optional, Tuple
import pandas as pd
import networkx as nx
from .edge_state_matrix import EdgeState, EdgeStateMatrix
from .graph_renderer import GraphRenderer
from .ate import ATECalculator
from itertools import product
from tqdm.auto import tqdm


class ECCS:

    EDGE_SUGGESTION_METHODS = ["Best single edge change"]

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

    def num_vars(self) -> int:
        """
        Calculate the number of variables in the data.

        Returns:
            The number of variables in the data.
        """
        return self._data.shape[1]

    def _graph_is_acceptable(self, graph: nx.DiGraph) -> bool:
        """
        Check if the graph is acceptable. A graph is acceptable if it is a directed acyclic graph (DAG)
        and there is a directed path from the treatment to the outcome.

        Parameters:
            graph: The graph to be checked.

        Returns:
            True if the graph is acceptable, False otherwise.
        """
        return nx.is_directed_acyclic_graph(graph) and nx.has_path(
            graph, self._treatment, self._outcome
        )

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

    def suggest(self, method: str) -> tuple[float, nx.DiGraph, pd.DataFrame]:
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

    def _suggest_best_single_edge_change(
        self,
    ) -> Tuple[float, nx.DiGraph, pd.DataFrame]:
        """
        Suggest the best single edge change that maximally changes the ATE.

        Returns:
            A tuple containing the suggested ATE, the suggested graph, and the suggested
                modification(s) as a dataframe.
        """
        base_ate = self.get_ate()
        best_ate = self.get_ate()
        best_ate_diff = 0
        best_graph = self._graph.copy()
        best_modifications = pd.DataFrame(columns=["Source", "Destination", "Change"])

        # iterate over all pairs of variables using tqdm
        for i, j in tqdm(
            product(range(self._num_vars), range(self._num_vars)),
            total=self._num_vars**2,
        ):

            src = self._data.columns[i]
            dst = self._data.columns[j]

            forward_state = self._edge_decisions_matrix.get_edge_state(src, dst)
            reverse_state = self._edge_decisions_matrix.get_edge_state(dst, src)

            if forward_state == EdgeState.ABSENT and reverse_state == EdgeState.ABSENT:
                self.add_edge(src, dst)
                if self._graph_is_acceptable(self._graph):
                    new_ate = self.get_ate()
                    new_ate_diff = abs(new_ate - base_ate)
                    if new_ate_diff > best_ate_diff:
                        best_ate = new_ate
                        best_ate_diff = new_ate_diff
                        best_graph = self.draw_graph()

                        best_modifications = pd.DataFrame(
                            [[src, dst, "Add"]],
                            columns=["Source", "Destination", "Change"],
                        )
                self.remove_edge(src, dst)
            elif (
                forward_state == EdgeState.ABSENT and reverse_state == EdgeState.PRESENT
            ):
                self.remove_edge(dst, src, remove_isolates=False)
                self.add_edge(src, dst)
                if self._graph_is_acceptable(self._graph):
                    new_ate = self.get_ate()
                    new_ate_diff = abs(new_ate - base_ate)
                    if new_ate_diff > best_ate_diff:
                        best_ate = new_ate
                        best_ate_diff = new_ate_diff
                        best_graph = self.draw_graph()
                        best_modifications = pd.DataFrame(
                            [[dst, src, "Flip"]],
                            columns=["Source", "Destination", "Change"],
                        )
                self.remove_edge(src, dst, remove_isolates=False)
                self.add_edge(dst, src)
            elif forward_state == EdgeState.PRESENT:
                self.remove_edge(src, dst, remove_isolates=False)
                if self._graph_is_acceptable(self._graph):
                    new_ate = self.get_ate()
                    new_ate_diff = abs(new_ate - base_ate)
                    if new_ate_diff > best_ate_diff:
                        best_ate = new_ate
                        best_ate_diff = new_ate_diff
                        best_graph = self.draw_graph()
                        best_modifications = pd.DataFrame(
                            [[src, dst, "Remove"]],
                            columns=["Source", "Destination", "Change"],
                        )
                self.add_edge(src, dst)

        return best_ate, best_graph, best_modifications
