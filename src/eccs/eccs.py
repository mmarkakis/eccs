from typing import Optional
import pandas as pd
import networkx as nx


class ECCS:

    def __init__(self, data_path: str, graph_path: Optional[str]):
        """
        Initialize the ECCS object.

        Parameters:
            data_path: The path to the data file.
            graph_path: Optionally, the path to the causal graph file.
        """

        self._data_path = data_path
        self._graph_path = graph_path

        self._data = pd.read_csv(data_path)

        if graph_path is not None:
            # Load the graph from a file in GraphML format into a networkx DiGraph object
            self._graph = nx.read_graphml(graph_path)


        