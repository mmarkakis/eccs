from typing import Optional
import pandas as pd
import networkx as nx
from .mab import MAB
from .ate import ATECalculator


class ECCS:

    def __init__(
        self, treatment: str, outcome: str, data_path: str, graph_path: Optional[str]
    ):
        """
        Initialize the ECCS object.

        Parameters:
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            data_path: The path to the data file.
            graph_path: Optionally, the path to the causal graph file in DOT format.
        """

        self._data_path = data_path
        self._graph_path = graph_path

        self._data = pd.read_csv(data_path)
        self._num_vars = self._data.shape[1]

        self._treatment_idx = self._data.columns.get_loc(treatment)
        self._outcome_idx = self._data.columns.get_loc(outcome)

        if graph_path is not None:
            # Load the graph from a file in DOT format into a networkx DiGraph object
            self._graph = nx.DiGraph(nx.nx_pydot.read_dot(graph_path))

        # Initialize the banlist to include all self-edges
        self._banlist = [self._edge_to_edge_idx((i, i)) for i in range(self.num_vars)]

    def num_vars(self) -> int:
        """
        Calculate the number of variables in the data.

        Returns:
            The number of variables in the data.
        """
        return self._data.shape[1]

    def _num_possible_dir_edges(self) -> int:
        """
        Calculate the number of possible directed edges in the graph.

        Returns:
            The number of possible directed edges in the graph.
        """
        return self._data.shape[1] ** 2

    def _edge_idx_to_edge(self, idx: int) -> tuple[int, int]:
        """
        Convert an index of a possible directed edge to the corresponding edge.

        Parameters:
            idx: The index of the possible directed edge.

        Returns:
            A tuple containing the source and target nodes of the edge.
        """
        return (idx // self.num_vars, idx % self.num_vars)

    def _edge_to_edge_idx(self, edge: tuple[int, int]) -> int:
        """
        Convert a directed edge to the corresponding index of a possible directed edge.

        Parameters:
            edge: A tuple containing the source and target nodes of the edge.

        Returns:
            The index of the possible directed edge.
        """
        return edge[0] * self.num_vars + edge[1]

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

    def find_new_graph(self, rounds: int, epsilon: float) -> nx.DiGraph:
        """
        Find a new causal graph for the data, based on exploring the space of possible graphs
        using the multi-armed bandit algorithm.

        Returns:
            The new causal graph.
        """

        # Initialize the multi-armed bandit with one arm for each possible edge in the graph
        mab = MAB(self._num_possible_dir_edges, epsilon, self._banlist)

        current_graph = self._graph.copy()
        original_ate = ATECalculator.get_ate_and_confidence(
            self._data,
            self._treatment_idx,
            self._outcome_idx,
            current_graph,
        )["ATE"]

        for i in range(rounds):
            arm = mab.select_arm()

            # Convert the arm index to an edge
            edge = self._edge_idx_to_edge(arm)

            # Add or remove the edge from the graph
            if current_graph.has_edge(*edge):
                current_graph.remove_edge(*edge)
            else:
                current_graph.add_edge(*edge)

            # Check if the new graph is acceptable otherwise undo change and skip this iteration
            if not self._graph_is_acceptable(current_graph):
                if current_graph.has_edge(*edge):
                    current_graph.remove_edge(*edge)
                else:
                    current_graph.add_edge(*edge)
                continue

            # Calculate the ATE of the treatment on the outcome using the new graph
            new_ate = ATECalculator.get_ate_and_confidence(
                self._data,
                self._treatment_idx,
                self._outcome_idx,
                current_graph,
            )["ATE"]

            # Update the reward of the selected arm
            reward = abs(new_ate - original_ate)
            mab.update(arm, reward)

        return current_graph
