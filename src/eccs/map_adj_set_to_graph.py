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
        if v in nx.descendants(self.graph, self.outcome):
            B, success = self._break_paths(self.outcome, v, S)
            if not success:
                return []
            S.extend(B)

        S.append(EdgeEdit(v, self.treatment, EdgeEditType.ADD))
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
            A list of causal graph edits that correspond to the removal of v from the adjustment set.
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
            A list of causal graph edits that correspond to the removal of v from the adjustment set.
        """
        S = []
        if v in nx.ancestors(self.graph, self.treatment):
            S, success = self._break_paths(v, self._treatment)
            if not success:
                return []
        
        for w in self._yield_BFS_descendants(self.treatment):
            if not (w, v) in self.ban_list:
                if (v, w) in self.graph.edges:
                    S.append(EdgeEdit(v, w, EdgeEditType.FLIP))
                else:
                    S.append(EdgeEdit(w, v, EdgeEditType.ADD))
                return S
        return []

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


    def _break_paths(self, src: str, dst: str, preremovals: Optional[list[EdgeEdit]]) -> tuple[list[EdgeEdit], bool]:
        """
        Break all directed paths between two nodes in the graph, taking into account some edges that may have
        already been removed.

        Parameters:
            src: The source node.
            dst: The destination node.
            preremovals: A list of edge edits indicating edge removals that have already been made.

        Returns:
            A list of causal graph edits that break all paths between the two nodes, and a boolean indicating success.
        """

        raise NotImplementedError
    
    def _yield_BFS_descendants(self, v: str) -> list[str]:
        """
        Yield the descendants of a variable, starting with the variable itself, in breadth-first search order.

        Parameters:
            v: The variable.

        Returns:
            A list of descendants in BFS order.
        """
        raise NotImplementedError
        