import networkx as nx

from .edits import EdgeEditType, EdgeEdit
from typing import TypeAlias

Edge: TypeAlias = tuple[str, str]
Path: TypeAlias = list[Edge]


class MapAdjSetToGraph:
    """
    A class for mapping adjustment set edits to causal graph edits.
    """

    def __init__(
        self, graph: nx.DiGraph, treatment: str, outcome: str, base_adj_set: list[str]
    ):
        """
        Initializes the class.

        Parameters:
            graph: The causal graph.
            treatment: The treatment variable.
            outcome: The outcome variable.
            base_adj_set: The starting adjustment set.
        """
        self.graph = graph
        self.skeleton = graph.to_undirected()
        self.treatment = treatment
        self.outcome = outcome
        self.base_adj_set = base_adj_set

    def map_addition(
        self,
        v: str,
        naive: bool,
    ) -> list[list[EdgeEdit]]:
        """
        Map an addition to the adjustment set to a list of lists.
        Each inner list contains causal graph edits that correspond to the
        addition of v to the adjustment set.

        Parameters:
            v: The variable to add to the adjustment set.
            naive: Whether to use the naive adjustment set addition strategy.

        Returns:
            A list of lists, where each inner list contains causal graph edits that correspond
            to the addition of v to the adjustment set.
        """

        if naive:
            return [
                [
                    EdgeEdit(v, self.treatment, EdgeEditType.ADD),
                    EdgeEdit(v, self.outcome, EdgeEditType.ADD),
                ]
            ]
        else:
            l = []
            # Each set of edges in l will be a Python tuple inside this function,
            # to facilitate deduplication before returning.
            if v in nx.ancestors(self.graph, self.treatment):
                l.append((EdgeEdit(v, self.outcome, EdgeEditType.ADD),))
                l.append((EdgeEdit(self.outcome, v, EdgeEditType.ADD),))
            elif v in nx.descendants(self.graph, self.treatment):
                paths = nx.all_simple_edge_paths(self.graph, self.treatment, v)
                for path in paths:
                    for edge in path:
                        l.append(
                            (
                                EdgeEdit(
                                    edge[0], edge[1], EdgeEditType.FLIP
                                ),  # v no longer a descendant
                                EdgeEdit(
                                    v, self.treatment, EdgeEditType.ADD
                                ),  # v blocks path
                                EdgeEdit(v, self.outcome, EdgeEditType.ADD),
                            )
                        )
                        l.append(
                            (
                                EdgeEdit(
                                    edge[0], edge[1], EdgeEditType.FLIP
                                ),  # v no longer a descendant
                                EdgeEdit(
                                    v, self.treatment, EdgeEditType.ADD
                                ),  # v blocks path
                                EdgeEdit(self.outcome, v, EdgeEditType.ADD),
                            )
                        )
            elif nx.has_path(self.skeleton, self.treatment, v):
                must_connect_to_outcome = not nx.has_path(
                    self.skeleton, v, self.outcome
                )

                for path in nx.all_simple_edge_paths(self.skeleton, self.treatment, v):
                    must_flip_first = path[0] in self.graph.edges()
                    must_flip_last = path[-1] in self.graph.edges()

                    local_l = tuple()
                    if must_flip_first:
                        local_l += (
                            EdgeEdit(path[0][0], path[0][1], EdgeEditType.FLIP),
                        )
                    if must_flip_last:
                        local_l += (
                            EdgeEdit(path[-1][0], path[-1][1], EdgeEditType.FLIP),
                        )
                    if must_connect_to_outcome:
                        l.append(
                            local_l + (EdgeEdit(v, self.outcome, EdgeEditType.ADD),)
                        )
                        l.append(
                            local_l + (EdgeEdit(self.outcome, v, EdgeEditType.ADD),)
                        )
                    else:
                        if len(local_l) == 0:
                            local_l = (EdgeEdit(v, self.treatment, EdgeEditType.ADD),)

                        l.append(local_l)

            else:  # v unconnected from treatment in original graph
                if nx.has_path(self.skeleton, v, self.outcome):
                    l.append((EdgeEdit(v, self.treatment, EdgeEditType.ADD)))
                else:
                    l.append(
                        (
                            EdgeEdit(v, self.treatment, EdgeEditType.ADD),
                            EdgeEdit(v, self.outcome, EdgeEditType.ADD),
                        )
                    )
                    l.append(
                        (
                            EdgeEdit(v, self.treatment, EdgeEditType.ADD),
                            EdgeEdit(self.outcome, v, EdgeEditType.ADD),
                        )
                    )

            print("The list of lists of edits is: ", l)
            # Deduplicate edits within each set of edits
            for i in range(len(l)):
                l[i] = tuple(set(l[i]))
            # Deduplciate strategies
            deduplicated_l = [list(e) for e in list(set(l))]
            print("The deduplicated list of lists of edits is: ", deduplicated_l)

            return deduplicated_l

    def map_removal(
        self,
        v: str,
        naive: bool,
    ) -> list[list[EdgeEdit]]:
        """
        Map a removal from the adjustment set to a list of lists.
        Each inner list contains causal graph edits that correspond to the
        removal of v from the adjustment set.

        Parameters:
            graph: The causal graph.
            treatment: The treatment variable.
            outcome: The outcome variable.
            base_adj_set: The starting adjustment set.
            v: The variable to remove from the adjustment set.
            naive: Whether to use the naive adjustment set removal strategy.

        Returns:
            A list of lists, where each inner list contains causal graph edits that correspond
            to the removal of v from the adjustment set.
        """

        if naive:
            return [
                [
                    EdgeEdit(self.treatment, v, EdgeEditType.ADD),
                ]
            ]
        else:
            l = [tuple()]
            reduced_adj_set = self.base_adj_set.copy()
            reduced_adj_set.remove(v)
            print("The reduced adj set is: ", reduced_adj_set)
            for path in self._backdoor_paths_through(v):
                print("The path is: ", path)
                print("It has length: ", len(path))
                if not self._is_path_blocked(path, self.graph, reduced_adj_set):
                    print("The path would not be blocked if we do the removal")
                    # We found a path that will no longer be blocked after removing v.
                    # Break the path by removing the edges that go out of v.
                    l[0] += tuple(
                        [
                            EdgeEdit(edge[0], edge[1], EdgeEditType.REMOVE)
                            for edge in path
                            if edge[0] == v
                        ]
                    )

            print("The list of lists of edits is: ", l)

            return [list(set(l[0]))] if len(l[0]) > 0 else []

    def _treatment_parents(self) -> list[str]:
        """
        Get the parents of the treatment variable.

        Returns:
            The parents of the treatment variable.
        """

        if hasattr(self, "treatment_parents"):
            return self.treatment_parents

        self.treatment_parents = list(self.graph.predecessors(self.treatment))
        return self.treatment_parents

    def _backdoor_paths(self) -> list[Path]:
        """
        Get backdoor paths between the treatment and the outcome.
        Backdoor paths are paths that start with an incoming edge to the treatment,
        and end at the outcome.

        Returns:
            A list of all backdoor paths.
        """

        if hasattr(self, "backdoor_paths"):
            return self.backdoor_paths

        treatment_parents = self._treatment_parents()

        self.backdoor_paths = []
        for parent in treatment_parents:
            for path in nx.all_simple_paths(self.skeleton, parent, self.outcome):
                if self.treatment not in path:
                    dir_path = [(parent, self.treatment)]
                    for i in range(len(path) - 1):
                        if (path[i], path[i + 1]) in self.graph.edges():
                            dir_path.append((path[i], path[i + 1]))
                        else:
                            dir_path.append((path[i + 1], path[i]))

                    self.backdoor_paths.append(dir_path)

        return self.backdoor_paths

    def _backdoor_paths_through(self, v: str) -> list[Path]:
        """
        Get backdoor paths between the treatment and the outcome that go through v.
        Backdoor paths are paths that start with an incoming edge to the treatment,
        and end at the outcome.

        Parameters:
            v: The variable that the backdoor paths should go through.

        Returns:
            A list of all backdoor paths that go through v.
        """

        return [
            path
            for path in self._backdoor_paths()
            if any([(edge[1] == v or edge[0] == v) for edge in path])
        ]

    def _is_path_blocked(
        self, path: Path, graph: nx.DiGraph, adj_set: list[str]
    ) -> bool:
        """
        Check if a path is blocked (in the causal sense) by the specified adjustment set.

        Parameters:
            path: The path to check.
            graph: The causal graph.
            adj_set: The adjustment set.

        Returns:
            True if the path is blocked, False otherwise.
        """

        # Check if path is blocked by inclusion
        print("Checking if path is blocked by inclusion")
        for i in range(len(path) - 1):
            # Forward chain
            if path[i][1] == path[i + 1][0] and path[i][1] in adj_set:
                print("The path is blocked by forward chain")
                return True

            # Backward chain
            if path[i][0] == path[i + 1][1] and path[i][0] in adj_set:
                print("The path is blocked by backward chain")
                return True

            # Fork
            if path[i][0] == path[i + 1][0] and path[i][0] in adj_set:
                print("The path is blocked by fork")
                return True

        # Check if path is blocked by exclusion
        print("Checking if path is blocked by exclusion")
        for i in range(len(path) - 1):
            if (
                path[i][1] == path[i + 1][1]  # Collider
                and path[i][1] not in adj_set
                and all(nx.descendants(self.graph, path[i][1])) not in adj_set
            ):
                print(f"The path is blocked by collider {path[i][1]}")
                return True
