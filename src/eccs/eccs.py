from typing import Optional, Tuple
import pandas as pd
import networkx as nx
from .edge_state_matrix import EdgeStateMatrix
from .graph_renderer import GraphRenderer


class ECCS:

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

        if graph_path is not None:
            # Load the graph from a file in DOT format into a networkx DiGraph object
            self._graph = nx.DiGraph(nx.nx_pydot.read_dot(graph_path))
        else:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(range(self._num_vars))

        self._edge_decisions_matrix = EdgeStateMatrix(list(self._data.columns))

        # Initialize the banlist to include all self-edges
        for i in range(self._num_vars):
            self.ban_edge(self._data.columns[i], self._data.columns[i])


    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the data.
        """
        return self._data

    @property
    def banlist(self) -> pd.DataFrame:
        """
        Returns the banlist, which includes all edges marked as rejected,
        except for self-edges.
        """
        banlist = pd.DataFrame(columns=["Source", "Destination"])
        for i in range(self._num_vars):
            for j in range(self._num_vars):
                if i != j and self.is_edge_banned(
                    self._data.columns[i], self._data.columns[j]
                ):
                    banlist = pd.concat(
                        [
                            banlist,
                            pd.DataFrame(
                                {
                                    "Source": self._data.columns[i],
                                    "Destination": self._data.columns[j],
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )

        return banlist

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
            self._edge_states = EdgeStateMatrix(self.prepared_variable_names)

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
        print(type(self._graph))
        print(type(self._edge_decisions_matrix))
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
        Add an edge to the graph. Does nothing if we try to add an edge that is banned.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if not self._edge_decisions_matrix.edge_is_rejected(src, dst):   
            self._graph.add_edge(src, dst)

    def remove_edge(self, src: str, dst: str) -> None:
        """
        Remove an edge from the graph and then remove any nodes with degree zero.
        Does nothing if we try to remove an edge that is fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if not self._edge_decisions_matrix.edge_is_accepted(src, dst):     
            self._graph.remove_edge(src, dst)
            self._graph.remove_nodes_from(list(nx.isolates(self._graph)))

    def fix_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as fixed and mark its reverse as banned. If the
        edge is not in the graph, it is added.

        Does nothing if the edge is already fixed or banned.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.edge_is_undecided(src, dst):
            self._edge_decisions_matrix.mark_edge(src, dst, "Accepted")
            self._edge_decisions_matrix.mark_edge(dst, src, "Rejected")
            if not self._graph.has_edge(src, dst):
                self.add_edge(src, dst)

    def ban_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as banned. If the edge is in the graph, it is removed.

        Does nothing if the edge is already banned or fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if self._edge_decisions_matrix.edge_is_undecided(src, dst):
            self._edge_decisions_matrix.mark_edge(src, dst, "Rejected")
            if self._graph.has_edge(src, dst):
                self.remove_edge(src, dst)

    def mark_edge_undecided(self, src: str, dst: str) -> None:
        """
        Mark an edge as not banned or fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self._edge_decisions_matrix.mark_edge(src, dst, "Undecided")

    def unban_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as not banned. Does nothing if the edge is the
        reverse of a fixed edge.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        if not self.is_edge_fixed(dst, src):
            self.mark_edge_undecided(src, dst)

    def unfix_edge(self, src: str, dst: str) -> None:
        """
        Mark an edge as not fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self.mark_edge_undecided(src, dst)
        self.mark_edge_undecided(dst, src)

    def is_edge_fixed(self, src: str, dst: str) -> bool:
        """
        Check if an edge is fixed.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is fixed, False otherwise.
        """
        return self._edge_decisions_matrix.edge_is_accepted(src, dst)

    def is_edge_banned(self, src: str, dst: str) -> bool:
        """
        Check if an edge is banned.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            True if the edge is banned, False otherwise.
        """
        return self._edge_decisions_matrix.edge_is_rejected(src, dst)
