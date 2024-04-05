import networkx as nx

from .edges import EdgeEditType, EdgeEdit, Path, Edge
from typing import Optional
from collections import deque


class MapAdjSetToGraph:
    """
    A class for mapping adjustment set edits to causal graph edits.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        fix_list: list[Edge],
        ban_list: list[Edge],
        base_adj_set: Optional[list[str]] = None,
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
        self.fix_list = fix_list
        self.ban_list = ban_list
        self.base_adj_set = base_adj_set

    def _find_directed_paths(self, src: str, dst: str) -> list[Path]:
        """
        Find all directed paths between two nodes in the graph.

        Parameters:
            src: The source node.
            dst: The destination node.

        Returns:
            A list of directed paths.
        """
        return list(nx.all_simple_edge_paths(self.graph, src, dst))

    def _break_path_near(self, path: Path, var: str) -> Optional[EdgeEdit]:
        """
        Find an edge to remove that will break the path near a variable. The variable
        has to be the first or last node in the path.

        Parameters:
            path: The path to break.
            var: The variable to break the path near.

        Returns:
            An edge edit that will break the path near the variable.
        """
        step = 1

        if var == path[-1][1]:
            step = -1
        elif var != path[0][0]:
            return None

        for edge in path[::step]:
            if edge not in self.fix_list:
                return EdgeEdit(edge[0], edge[1], EdgeEditType.REMOVE)
        return None

    def _find_ordered_descendants(self, v: str) -> list[str]:
        """
        Find all descendants of a variable, including itself, and order them by their distance from the variable.

        Parameters:
            v: The variable.

        Returns:
            A list of descendants ordered by their distance from the variable.
        """
        d = []
        q = deque([v])
        while len(q) > 0:
            current = q.popleft()
            d.append(current)
            q.extend([n for n in nx.descendants(self.graph, current) if n not in d])
        return d

    def map_addition(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map an addition to the adjustment set to a list of causal graph edits.

        Parameters:
            v: The variable to add to the adjustment set.

        Returns:
            A list of causal graph edits that correspond to the addition of v to the adjustment set.
        """

        e = []
        if v in nx.descendants(self.graph, self.treatment) or v in nx.descendants(
            self.graph, self.outcome
        ):
            paths = self._find_directed_paths(
                self.treatment, v
            ) + self._find_directed_paths(v, self.outcome)
            for p in paths:
                edit = self._break_path_near(p, v)
                if edit is None:
                    return []
                e.append(edit)
        e.append(EdgeEdit(v, self.treatment, EdgeEditType.ADD))
        e.append(EdgeEdit(v, self.outcome, EdgeEditType.ADD))
        return list(set(e))

    def map_removal(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map a removal from the adjustment set to a list of causal graph edits.

        Parameters:
            v: The variable to remove from the adjustment set.

        Returns:
            A list of causal graph edits that correspond to the removal of v from the adjustment set.
        """

        e = []
        if v in nx.ancestors(self.graph, self.treatment):
            paths = self._find_directed_paths(v, self.treatment)
            for p in paths:
                edit = self._break_path_near(p, v)
                if edit is None:
                    return []
                e.append(edit)

        ord_desc = self._find_ordered_descendants(self.treatment)
        for w in ord_desc:
            if not (w, v) in self.ban_list:
                if (v, w) in self.graph.edges:
                    e.append(EdgeEdit(v, w, EdgeEditType.FLIP))
                else:
                    e.append(EdgeEdit(w, v, EdgeEditType.ADD))
                return list(set(e))
        return []

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
