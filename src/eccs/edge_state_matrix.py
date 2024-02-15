import numpy as np
import networkx as nx


class EdgeStateMatrix:
    """
    A class for managing an edge state matrix.

    An edge state matrix is square, with the entry (i,j) representing the state
    of the directed edge between nodes i and j. The state of an edge is one of:
         0: The edge is undecided.
        -1: The edge does not exist.
         1: The edge exists.

    Self-edges are not allowed. The presence of an edge implies the absence of
    its inverse.
    """

    def __init__(self, variables: list[str]) -> None:
        """
        Initialize the edge state matrix to the right dimensions and mark self-edges
        as rejected and all other edges as undecided.

        Parameters:
            variables: The variables to initialize the edge state matrix based on.
        """

        n = len(variables)
        self._variables = variables
        self._m = np.zeros((n, n))
        for i in range(n):
            self._m[i, i] = -1

    @property
    def m(self) -> np.ndarray:
        """
        Returns the edge state matrix.
        """
        return self._m

    @property
    def n(self) -> int:
        """
        Returns the number of nodes.
        """
        return self._m.shape[0]

    def clear_and_set_from_graph(self, graph: nx.DiGraph, mark_missing_as:str = "Undecided") -> None:
        """
        Clear the edge state matrix and then set it based on the provided graph.
        In particular, mark all edges in the graph as accepted and all others as either
        rejected or undecided, depending on the value of `mark_missing_as`.

        Parameters:
            graph: The graph to use to set the edge states.
            mark_missing_as: The value to use for edges that are not in the graph.

        Throws:
            ValueError: If `mark_missing_as` is not one of "Rejected" or "Undecided".
        """

        self._m = np.zeros((self.n, self.n))
        for edge in graph.edges:
            print("Marking edge as accepted: ", edge)
            self._m[self.idx(edge[0]), self.idx(edge[1])] = 1

        if mark_missing_as == "Undecided":
            self._m[self._m == 0] = 0
        elif mark_missing_as == "Rejected":
            self._m[self._m == 0] =  -1
        else:
            raise ValueError(f"Invalid value for mark_missing_as: {mark_missing_as}")

    def clear_and_set_from_matrix(self, m: np.ndarray) -> None:
        """
        Clear the edge state matrix and then set it based on the provided matrix.

        Parameters:
            m: The matrix to use to set the edge states.
        """

        self._m = m

    def idx(self, var: str) -> int:
        """
        Retrieve the index of a variable in the edge state matrix.

        Parameters:
            var: The name of the variable.

        Returns:
            The index of the variable in the edge state matrix.
        """
        return self._variables.index(var)
    
    def name(self, idx: int) -> str:
        """
        Retrieve the name of a variable in the edge state matrix.

        Parameters:
            idx: The index of the variable.

        Returns:
            The name of the variable in the edge state matrix.
        """
        return self._variables[idx]

    def get_edge_state(self, src: str, dst: str) -> str:
        """
        Get the state of a specific edge.

        Parameters:
            src: The name of the source variable.
            dst: The name of the destination variable.

        Returns:
            The state of the edge (Accepted, Rejected, or Undecided).
        """
        src_idx = self.idx(src)
        dst_idx = self.idx(dst)
        return self.edge_state_to_str(self._m[src_idx][dst_idx])

    def edge_state_to_str(self, state: int) -> str:
        """
        Translate between edge value and its interpretation.

        Parameters:
            state: The state of the edge represented as an integer.

        Returns:
            The state of the edge (Accepted, Rejected, or Undecided).
        """
        if state == 0:
            return "Undecided"
        elif state == -1:
            return "Rejected"
        elif state == 1:
            return "Accepted"
        else:
            raise ValueError(f"Invalid edge state {state}")

    def mark_edge(self, src: str | int, dst: str| int, state: str) -> None:
        """
        Mark an edge as being in a specified state.

        Parameters:
            src: The name or index of the source variable.
            dst: The name or index of the destination variable.
            state: The state to mark the edge with (Accepted, Rejected, or Undecided).


        Throws:
            ValueError: If `state` is not one of "Accepted", "Rejected", or "Undecided".
        """

        src_idx = self.idx(src) if type(src) == str else src
        dst_idx = self.idx(dst) if type(dst) == str else dst

        if state == "Accepted":
            self._m[src_idx][dst_idx] = 1
            self._m[dst_idx][src_idx] = -1
        elif state == "Rejected":
            self._m[src_idx][dst_idx] = -1
        elif state == "Undecided":
            self._m[src_idx][dst_idx] = 0
        else:
            raise ValueError(f"Invalid edge state {state}")
        

    @staticmethod
    def enumerate_with_max_edges(n: int, max_edges: int) -> list[np.ndarray]:
        """
        Enumerate all edge state matrices of dimension `n` with at most `max_edges` accepted edges.

        Parameters:
            n: The dimension of the edge state matrices.
            max_edges: The maximum number of edges to allow in the edge state matrices.

        Returns:
            A list of edge state matrices.
        """
        valid_matrices = {0: [np.full(shape=(n, n), fill_value=-1)]}

        # Enumerate all valid matrices with k edges
        for k in range(1, max_edges + 1):
            valid_matrices[k] = []

            # For each valid matrix with k-1 edges...
            for m in valid_matrices[k - 1]:
                # ...add a new edge in every possible way
                for i in range(n):
                    for j in range(i + 1, n):
                        if m[i, j] < 0 and m[j, i] < 0:
                            forward = m.copy()
                            forward[i, j] = 1
                            valid_matrices[k].append(forward)
                            backward = m.copy()
                            backward[j, i] = 1
                            valid_matrices[k].append(backward)

        # Flatten the collection of matrices into a single list
        returned_matrices = []
        for k in range(1, max_edges + 1):
            returned_matrices.extend(valid_matrices[k])

        return returned_matrices
    
