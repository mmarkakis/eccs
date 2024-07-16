import networkx as nx

from .edges import EdgeEditType, EdgeEdit, Path, Edge
from typing import Optional, Iterator
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
        self.treatment = treatment
        self.outcome = outcome
        self.fix_list = fix_list
        self.ban_list = ban_list
        self.base_adj_set = base_adj_set

    def update_fixlist(self, fix_list: list[Edge]) -> None:
        """
        Update the fix list.

        Parameters:
            fix_list: The new fix list.
        """
        self.fix_list = fix_list

    def update_banlist(self, ban_list: list[Edge]) -> None:
        """
        Update the ban list.

        Parameters:
            ban_list: The new ban list.
        """
        self.ban_list = ban_list

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
        use_optimized: bool = True,
    ) -> list[EdgeEdit]:
        """
        Map an addition to the adjustment set to a list of causal graph edits.

        Parameters:
            v: The variable to add to the adjustment set.
            use_optimized: Whether to use the optimized version of the function, developed after the ECCS paper.

        Returns:
            A list of causal graph edits that correspond to the addition of v to the adjustment set.
        """
        if use_optimized:
            return self._optimized_map_addition(v)
        return self._unoptimized_map_addition(v)

    def _optimized_map_addition(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map an addition to the adjustment set to a list of causal graph edits. This is the version of
        this function developed after the ECCS paper.

        Parameters:
            v: The variable to add to the adjustment set.

        Returns:
            A list of causal graph edits that correspond to the addition of v to the adjustment set.
        """

        S = []
        if v in nx.descendants(self.graph, self.treatment):
            B, success = self._break_paths(self.treatment, v)
            if not success:
                return []
            S.extend(B)
        ignore = [e.edge for e in S]
        if self._is_reachable_with_ignored_edges(self.outcome, v, ignore):
            B, success = self._break_paths(self.outcome, v, ignore)
            if not success:
                return []
            S.extend(B)

        if not self.graph.has_edge(v, self.treatment):
            S.append(EdgeEdit(v, self.treatment, EdgeEditType.ADD))
        if not self.graph.has_edge(v, self.outcome):
            S.append(EdgeEdit(v, self.outcome, EdgeEditType.ADD))
        return S

    def _unoptimized_map_addition(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map an addition to the adjustment set to a list of causal graph edits. This is the version of
        this function presented in the ECCS paper.

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
        use_optimized: bool = True,
    ) -> list[EdgeEdit]:
        """
        Map a removal from the adjustment set to a list of causal graph edits.

        Parameters:
            v: The variable to remove from the adjustment set.
            use_optimized: Whether to use the optimized version of the function, developed after the ECCS paper.

        Returns:
            A list of causal graph edits that correspond to the removal of v from the adjustment set. If the
            returned list is empty, the removal was unsuccessful.
        """

        if use_optimized:
            return self._optimized_map_removal(v)
        return self._unoptimized_map_removal(v)

    def _optimized_map_removal(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map a removal from the adjustment set to a list of causal graph edits. This is the version of
        this function developed after the ECCS paper.

        Parameters:
            v: The variable to remove from the adjustment set.

        Returns:
            A list of causal graph edits that correspond to the removal of v from the adjustment set. If
            the returned list is empty, the removal was unsuccessful.
        """

        v_descendants = nx.descendants(self.graph, v)
        t_bfs_descendants = self._BFS_descendants(self.treatment)
        t_bfs_descendants_and_v_descendants = []

        # First search among descendants that don't need path breaking.
        for w in t_bfs_descendants:
            if (w, v) in self.ban_list:
                continue

            if w in v_descendants:
                t_bfs_descendants_and_v_descendants.append(w)
                continue

            return [EdgeEdit(w, v, EdgeEditType.ADD)]

        # If unsuccessful, next search among descendants that need path breaking.
        best_S = []
        for w in t_bfs_descendants_and_v_descendants:
            S, success = self._break_paths(v, w)
            if not success:
                continue
            if not self._is_reachable_with_ignored_edges(
                self.treatment, w, [e.edge for e in S]
            ):
                continue
            S.append(EdgeEdit(w, v, EdgeEditType.ADD))
            if len(S) < len(best_S) or len(best_S) == 0:
                best_S = S

        return best_S

    def _unoptimized_map_removal(
        self,
        v: str,
    ) -> list[EdgeEdit]:
        """
        Map a removal from the adjustment set to a list of causal graph edits. This is the version of
        this function presented in the ECCS paper.

        Parameters:
            v: The variable to remove from the adjustment set.

        Returns:
            A list of causal graph edits that correspond to the removal of v from the adjustment set. If
            the returned list is empty, the removal was unsuccessful.
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

    def _all_reachable_with_ignored_edges(
        self, source: str, ignore: Optional[set[Edge]] = None, reverse: bool = False
    ) -> set[str]:
        """
        Perform a breadth-first search starting from a source node, ignoring certain edges,
        and return a list of reachable nodes.

        Parameters:
            source: The source node.
            ignore: The edges to ignore.
            reverse: Whether to perform the search in the reverse directed graph.

        Returns:
            A set of reachable nodes.
        """

        queue = deque([source])
        visited = set([source])

        if reverse:
            next_nodes = self.graph.predecessors
            edge_is_ignored = (
                (lambda x, y: (y, x) in ignore) if ignore else (lambda x, y: False)
            )
        else:
            next_nodes = self.graph.successors
            edge_is_ignored = (
                (lambda x, y: (x, y) in ignore) if ignore else (lambda x, y: False)
            )

        while queue:
            node = queue.popleft()
            for neighbor in next_nodes(node):
                if edge_is_ignored(node, neighbor):
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def _is_reachable_with_ignored_edges(
        self, source: str, sink: str, ignore: set[Edge]
    ) -> bool:
        """
        Check if a sink node is reachable from a source node, ignoring certain edges.

        Parameters:
            source: The source node.
            sink: The sink node.
            ignore: The edges to ignore.

        Returns:
            A boolean indicating whether the sink node is reachable from the source node.
        """

        return sink in self._all_reachable_with_ignored_edges(source, ignore)

    def _break_paths(
        self, source: str, sink: str, ignore: Optional[set[Edge]] = None
    ) -> tuple[list[EdgeEdit], bool]:
        """
        Break all directed paths between two nodes in a directed graph, ignoring any edges specified in `ignore`.

        Parameters:
            source: The source node.
            sink: The sink node.
            ignore: The edges to ignore.

        Returns:
            A list of causal graph edits that break all paths between the two nodes, and a boolean indicating success.
        """

        reachable_from_source = self._all_reachable_with_ignored_edges(source, ignore)
        sink_reachable_from = self._all_reachable_with_ignored_edges(
            sink, ignore, reverse=True
        )

        nodes_to_keep = reachable_from_source.intersection(sink_reachable_from)
        if len(nodes_to_keep) == 0:
            return [], True

        B = []
        queue = deque([source])
        visited = set([source])

        while queue:
            V = queue.popleft()

            if V == sink:
                return [], False

            for W in self.graph.successors(V):
                if W not in nodes_to_keep:
                    continue
                if (V, W) not in self.fix_list:
                    B.append(EdgeEdit(V, W, EdgeEditType.REMOVE))
                elif W not in visited:
                    visited.add(W)
                    queue.append(W)

        return B, True

    def _BFS_descendants(self, v: str) -> list[str]:
        """
        Return the descendants of a variable, starting with the variable itself, in breadth-first search order.

        Parameters:
            v: The variable.

        Returns:
            A list of descendants in BFS order.
        """
        queue = deque([v])
        visited = [v]

        while queue:
            node = queue.popleft()
            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)

        return visited
