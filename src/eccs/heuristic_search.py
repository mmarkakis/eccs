import dihash
import heapq
import networkx as nx
import pandas as pd
from typing import Any, List, Optional, Tuple

from .ate import ATECalculator
from .edge_state_matrix import EdgeStateMatrix
from .edges import EdgeEditType, EdgeEdit

DEFAULT_BUDGET = 1000


class AStarSearch:
    def __init__(
        self,
        init_graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        edge_states: Optional[EdgeStateMatrix] = None,
        gamma_1: float = 2,
        gamma_2: float = 0.5,
        p_value_threshold: float = 0.5,
        std_err_threshold: float = -0.01,
        computational_budget: int = DEFAULT_BUDGET,
    ):
        if computational_budget is None or computational_budget < 0:
            computational_budget = DEFAULT_BUDGET

        # n is the number of causal variables
        # m is the number of edges in the initial graph
        print("Initializing A star")
        self.m = len(init_graph.edges())
        self.n = len(init_graph.nodes())
        self.init_graph = init_graph
        for (
            n
        ) in (
            self.init_graph.nodes
        ):  # This is some hacky code to make it work with graph hashing
            self.init_graph.nodes[n]["label"] = self.init_graph.nodes[n]["var_name"]
        self.treatment = treatment
        self.outcome = outcome
        self.data = data
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.p_value_threshold = p_value_threshold
        self.std_err_threshold = std_err_threshold
        self.ate_calculator = ATECalculator()
        if edge_states is None:  # backwards compatibility
            self.edge_states = EdgeStateMatrix(list(self.data.columns))
        else:
            self.edge_states = edge_states

        # graph id -> (pred graph id, (start causal variable, end causal variable, boolean addition or deletion))
        # TODO: make this code look nicer with edge types (low priority)
        # True is addition False is deletion
        self._predecessors = {}
        self._ATE_cache = {}  # causal graph id -> ATE info
        self._visited = set()
        self._hashtag_to_id = {}  # This notes that something is hashed
        self._id_to_graph = {}
        self._cur_next_id = 0
        self._computational_budget = computational_budget
        self._f_score = {}
        self._g_score = {}
        self._lookahead_threshold = 3

        init_ATE_info = self._get_ATE_info(0, init_graph)
        assert 0 in self._ATE_cache
        self.ATE_init = init_ATE_info["ATE"]
        self._init_potential = self._get_potential(0, init_graph)
        print("Initialization finished")

    def _get_ATE_info(self, id: int, graph: nx.DiGraph):
        try:
            return self._ATE_cache[id]
        except KeyError:
            try:
                ate = self.ate_calculator.get_ate_and_confidence(
                    data=self.data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    graph=graph,
                    calculate_p_value=True,
                    calculate_std_error=False, ## FIXME: Turned off for now
                    get_estimand=False,
                    print_timing_info=False,
                )
                self._ATE_cache[id] = ate
            except ValueError:  # Returned by handler of nx.exception.NodeNotFound
                # Special case of treatment and outcome being not connected
                ate = {"ATE": 0, "P-value": 0, "Standard Error": 0}
                self._ATE_cache[id] = ate
            return ate

    def _get_potential(self, id: int, graph: nx.DiGraph):
        """
        v is a (id: int, corresponding causal graph: nx.DiGraph)
        Phi(v) = |ATE_v - ATE_init| - gamma_1 * regression error + gamma_2 * other heuristic
        For now, use |number of edges in v - n| as the "other heuristic"
        Could plug in other forms of assessment in the literatures here
        """
        # {"ATE": float, "P-value": float, "Standard Error": float, "Estimand": ?}
        # TODO: what is the type of Estimand
        # TODO: implement caching for ATE information
        ATE_info = self._get_ATE_info(id, graph)
        # The math.inf stops graphs where treatment and outcome are not direct connected

        """
        stderr = 0
        if ATE_info["Standard Error"] == 0:
            stderr = 1
        else:
            try:
                stderr = ATE_info["Standard Error"][0]
            except IndexError:
                stderr = ATE_info["Standard Error"]
        """

        return (
            ATE_info["ATE"]
            #- self.gamma_1 * stderr
            + ATE_info["P-value"]
            + 0.01 * abs(len(graph.edges()) - self.m * 2)
        )  # TODO: 2 is a guess of average degree

    def _explore_neighbor(
        self,
        current_node_id: int,
        graph: nx.DiGraph,
        n1: int,
        n2: int,
        frontier: List[Any],
        n_lookahead: int,
        is_add: bool,
    ) -> Optional[int]:
        # the type of nx node is int unless DiGraph.nodes() was called with data options
        # returns the id of the new neighbor or None if we don't explore

        if (
            not is_add and len(graph.edges()) <= self.n
        ):  # Only explore completely connected graphs
            return (None, None)
        if (
            is_add and len(graph.edges()) >= self.n * self.n // 2
        ):  # Skip overly connected graphs too
            return (None, None)

        if is_add:
            graph.add_edge(n1, n2)
        else:
            graph.remove_edge(n1, n2)
        hashtag = dihash.hash_graph(graph, hash_nodes=False, apply_quotient=False)
        if (
            hashtag not in self._hashtag_to_id
        ):  # not seen yet, mark it as seen by storing hash
            self._hashtag_to_id[hashtag] = self._cur_next_id
            self._cur_next_id += 1

        id = self._hashtag_to_id[hashtag]
        ATE_info = self._get_ATE_info(id, graph)

        # if ATE_info["P-value"] < self.p_value_threshold:
        #if ATE_info["Standard Error"] > self.std_err_threshold:
        if ATE_info["ATE"] <= 0.01:
            # self._visited.add(id)  # discard
            if id not in self._id_to_graph:
                new_graph = graph.copy()
                self._id_to_graph[id] = new_graph
            if n_lookahead < self._lookahead_threshold:
                f_score = self._f_score.get(
                    id, self._g_score[current_node_id] - self._get_potential(id, graph)
                )
                if id not in self._f_score:
                    self._f_score[id] = f_score
                if id not in self._g_score:
                    self._g_score[id] = self._g_score[current_node_id]
                heapq.heappush(frontier, (f_score, id, n_lookahead + 1))

        result = None
        if id not in self._visited:
            # explore (but not commit to the path, so we don't set the precedessor here)
            if id not in self._id_to_graph:  # store the result of this exploration
                new_graph = graph.copy()
                self._id_to_graph[id] = new_graph
            result = id

        # Revert the change on this graph before finish exploring
        if is_add:
            graph.remove_edge(n1, n2)
        else:
            graph.add_edge(n1, n2)
        return (result, (n1, n2, is_add))

    def _get_neighbors(
        self, current_node_id: int, frontier: List[Any], n_lookahead: int
    ) -> List[Tuple[int, Tuple[int, int, bool]]]:
        # Returns list of neighbors
        graph: nx.DiGraph = self._id_to_graph[current_node_id]
        results = []
        for n1 in list(graph.nodes):
            for n2 in list(graph.nodes):
                if n2 == n1:
                    continue  # skip self-loops
                neighbor_res = None
                if graph.has_edge(n1, n2) and not self.edge_states.is_edge_fixed(
                    n1, n2
                ):
                    neighbor_res = self._explore_neighbor(
                        current_node_id,
                        graph,
                        n1,
                        n2,
                        frontier,
                        n_lookahead,
                        is_add=False,
                    )
                elif n2 in nx.ancestors(graph, n1):
                    continue  # Adding this edge his creates a cycle
                elif not graph.has_edge(n1, n2) and not self.edge_states.is_edge_banned(n1, n2):
                    neighbor_res = self._explore_neighbor(
                        current_node_id,
                        graph,
                        n1,
                        n2,
                        frontier,
                        n_lookahead,
                        is_add=True,
                    )
                if neighbor_res is not None and len(neighbor_res) > 0 and neighbor_res[0] is not None:
                    results.append(neighbor_res)

        return results

    def heuristic(self, predecessor, node):
        # predecessor was taken from the frontier pq, node is the candidate for expansion
        # the weight of an edge (v1, v2) in the A* graph is Phi(v1) - Phi(v2) since we want to
        # go towards the maximal Phi causal graph.
        # A* heuristic f(n) = g(n) + h(n). g(n) is the cost of this explored path, h(n) is a heuristical
        # estimate of the cost from here to the destination. To be admissible, h(n) cannot overestimate the true
        # value. Therefore, we use -Phi(v2) as an estimate for now.
        # g(n) = Phi(init) - Phi(current frontier of the path)
        return (
            self._init_potential
            - self._get_potential(predecessor)
            - self._get_potential(node)
        )

    def astar(self, k: int = 100):
        # Side effect: prints the top 10 result
        # Returns the most frequently seen edge flips in sorted order
        frontier = [
            (0, self._cur_next_id, 0)
        ]  # this is the pq (f(v), v), only store the ID
        self._cur_next_id += 1

        start_hash = dihash.hash_graph(
            self.init_graph, hash_nodes=False, apply_quotient=False
        )
        # self._visited.add(0)  # store actual graph here
        self._hashtag_to_id[start_hash] = 0
        self._id_to_graph[0] = self.init_graph
        # starting node always has id 0
        self._f_score[0] = -self._get_potential(0, self.init_graph)
        # g_score[0] = self._init_potential + f_score[0]
        self._g_score[0] = self._f_score[0]

        top_k_candidates = []

        while frontier:
            _, current_node_id, n_lookahead = heapq.heappop(frontier)
            if current_node_id in self._visited:
                continue
            heapq.heappush(
                top_k_candidates,
                (
                    self._get_potential(
                        current_node_id, self._id_to_graph[current_node_id]
                    ),
                    current_node_id,
                ),
            )
            if len(top_k_candidates) > k:
                heapq.heappop(top_k_candidates)
            neighbors = self._get_neighbors(current_node_id, frontier, n_lookahead)
            if self._cur_next_id > self._computational_budget:
                print(
                    "Out of computational budget: ",
                    self._cur_next_id,
                    self._computational_budget,
                )
                break
            for neighbor_id, edge_type in neighbors:
                # when expanding neighbors, just discard ones that are too low p-value
                tentative_g_score = self._g_score[current_node_id]
                neighbor_g_score = self._g_score.get(
                    neighbor_id,
                    self._init_potential
                    - self._get_potential(neighbor_id, self._id_to_graph[neighbor_id]),
                )
                if tentative_g_score <= neighbor_g_score:
                    self._predecessors[neighbor_id] = (current_node_id, edge_type)
                    self._g_score[neighbor_id] = tentative_g_score
                    # f_score[neighbor_id] = tentative_g_score + self.heuristic(neighbor)
                    self._f_score[neighbor_id] = (
                        tentative_g_score
                        - self._get_potential(
                            neighbor_id, self._id_to_graph[neighbor_id]
                        )
                    )
                    heapq.heappush(
                        frontier, (self._f_score[neighbor_id], neighbor_id, 0)
                    )

                if self._cur_next_id > self._computational_budget:
                    print(
                        "Out of computational budget: ",
                        self._cur_next_id,
                        self._computational_budget,
                    )
                    break
            self._visited.add(current_node_id)

        edge_tally = {}

        for p in top_k_candidates:
            c_id = p[1]
            initial_c_id = p[1]
            while c_id in self._predecessors:
                c_id, (n1, n2, _) = self._predecessors[c_id]
                if (n1, n2) not in edge_tally:
                    edge_tally[(n1, n2)] = (
                        1,
                        abs(self._ATE_cache[initial_c_id]["ATE"] - self.ATE_init),
                    )
                else:
                    cnt = edge_tally[(n1, n2)][0] + 1
                    total_diff = edge_tally[(n1, n2)][1] + abs(
                        self._ATE_cache[initial_c_id]["ATE"] - self.ATE_init
                    )
                    edge_tally[(n1, n2)] = (cnt, total_diff)
        
        num_return = 10
        res = []
        sorted_edges = sorted(edge_tally.items(), key=lambda x: x[1][0], reverse=True)                
        print("The top 10 edges are")
        print(sorted_edges[:10])
        for (edge, _) in sorted_edges:
            src = edge[0]
            dst = edge[1]
            if self.edge_states.is_edge_fixed(src, dst):
                continue
            
            edit_type = EdgeEditType.REMOVE if self.init_graph.has_edge(src, dst) else EdgeEditType.ADD
            res.append(EdgeEdit(src, dst, edit_type))
            if len(res) > num_return:
                break
        
        # Convert top edge to EdgeEdit
        if len(sorted_edges) == 0:
            return []
        return res
    
