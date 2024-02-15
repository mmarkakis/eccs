from typing import Optional, Tuple
import pandas as pd
import networkx as nx
from .edge_state_matrix import EdgeStateMatrix
from .graph_renderer import GraphRenderer


class ECCS:

    def __init__(
        self, data_path: str, graph_path: Optional[str]
    ):
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
        self._edge_decisions_matrix.clear_and_set_from_graph(
            self._graph, mark_missing_as="Undecided"
        )

        # Initialize the banlist to include all self-edges
        for i in range(self._num_vars):
            self._edge_decisions_matrix.mark_edge(i, i, "Rejected")

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

    def save_graph(self, filename: str) -> None:
        """
        Save the current graph to a file.

        Parameters:
            filename: The name of the file to save to.
        """
        GraphRenderer.save_graph(self._graph, self._edge_decisions_matrix, filename)

    def add_edge(self, src: str, dst: str) -> None:
        """
        Add an edge to the graph.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self._graph.add_edge(src, dst)

    def remove_edge(self, src: str, dst: str) -> None:
        """
        Remove an edge from the graph.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self._graph.remove_edge(src, dst)

    def fix_edge_in_graph(self, src: str, dst: str) -> None:
        """
        Mark an edge as accepted and mark its reverse as rejected.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self._edge_decisions_matrix.mark_edge(src, dst, "Accepted")
        self._edge_decisions_matrix.mark_edge(dst, src, "Rejected")

    def fix_edge_outside_graph(self, src: str, dst: str) -> None:
        """
        Mark an edge as rejected.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.
        """
        self._edge_decisions_matrix.mark_edge(src, dst, "Rejected")

   

  
