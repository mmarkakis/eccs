import dihash
import heapq
import math
import networkx as nx
from typing import List, Optional

from .ate import ATECalculator

class AStarSearch:
    def __init__(self, init_graph, treatment, outcome, data, gamma_1=0.5, gamma_2=0.5, p_value_threshold=0.5, computational_budget=100000):
        # n is the number of causal variables
        # m is the number of edges in the initial graph
        print("Initializing A star")
        self.m = len(init_graph.edges())
        self.n = len(init_graph.nodes())
        self.init_graph = init_graph
        self.treatment = treatment
        self.outcome = outcome
        self.data = data
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.p_value_threshold = p_value_threshold

        # graph id -> (pred graph id, (start causal variable, end causal variable, boolean addition or deletion))
        # TODO: make this code look nicer with edge types (low priority)
        # True is addition False is deletion
        self._predecessors = {}
        self._ATE_cache = {} # causal graph id -> ATE info
        self._visited = set()
        self._hashtag_to_id = {} # This notes that something is hashed
        self._id_to_graph = {}
        self._cur_next_id = 0
        self._computational_budget = computational_budget

        init_ATE_info = self._get_ATE_info(0, init_graph)
        assert 0 in self._ATE_cache
        self.ATE_init = init_ATE_info["ATE"]
        self._init_potential = self._get_potential(0, init_graph)
        print("Initialization finished")
    
    def _get_ATE_info(self, id: int, graph: nx.DiGraph):
        try:
            return self._ATE_cache[id]
        except KeyError:
            ate = ATECalculator.get_ate_and_confidence(
                data=self.data,
                treatment=str(self.treatment),
                outcome=str(self.outcome),
                graph=graph,
                calculate_p_value=True,
                calculate_std_error=True,
                get_estimand=False,
            )
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
        return ATE_info["ATE"] - self.gamma_1 * ATE_info["Standard Error"] + ATE_info["P-value"] + math.abs(len(v.edges() - self.n))

    def _explore_neighbor(self, graph: nx.DiGraph, n1: int, n2: int, is_add: bool) -> Optional[int]:
        # the type of nx node is int unless DiGraph.nodes() was called with data options
        # returns the id of the new neighbor or None if we don't explore
        
        if is_add:
            graph.add_edge(n1, n2)
        else:
            graph.remove_edge(n1, n2)
        hashtag = dihash.hash_graph(
            graph,
            hash_nodes=False,
            apply_quotient=False
        )
        if hashtag not in self._hashtag_to_id: # not seen yet, mark it as seen by storing hash
            self._hashtag_to_id[hashtag] = self._cur_next_id
            self._cur_next_id += 1

        id = self._hashtag_to_id[hashtag]
        ATE_info = self._get_ATE_info(id, graph)
        if ATE_info["P-value"] < self.p_value_threshold:
            self._visited.add(id) # discard

        result = None
        if id not in self._visited:
            # explore (but not commit to the path, so we don't set the precedessor here)
            if id not in self._id_to_graph: # store the result of this exploration
                new_graph = graph.copy()
                self._id_to_graph[id] = new_graph
            result = id
        
        # Revert the change on this graph before finish exploring
        if is_add:
            graph.remove_edge(n1, n2)
        else:
            graph.add_edge(n1, n2)
        return result
    
    def _get_neighbors(self, current_node_id: int) -> List[int]:
        # Returns list of neighbors
        graph: nx.DiGraph = self._id_to_graph[current_node_id]
        results = []
        for n1 in list(graph.nodes):
            for n2 in list(graph.nodes):
                neighbor_res = None
                if graph.has_edge(n1, n2):
                    neighbor_res = self._explore_neighbor(graph, n1, n2, is_add=False)
                elif n2 in nx.ancestors(graph, n1):
                    continue # Adding this edge his creates a cycle
                else:
                    neighbor_res = self._explore_neighbor(graph, n1, n2, is_add=True)
                if neighbor_res is not None:
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
        return self._init_potential - self._get_potential(predecessor) - self._get_potential(node)

    def astar(self, k=100):
        # Side effect: prints the top 10 result
        # Returns the most frequently seen edge flips in sorted order
        frontier = [(0, self._cur_next_id)] # this is the pq (f(v), v), only store the ID
        self._cur_next_id += 1

        start_hash = dihash.hash_graph(
            self.init_graph,
            hash_nodes=False,
            apply_quotient=False
        )
        self._visited[0] = self.init_graph # store actual graph here
        self._hashtag_to_id[start_hash] = 0
        g_score = {}
        # starting node always has id 0
        g_score[0] = 0
        f_score = {}
        f_score[0] =  - self._get_potential(0, self.init_graph)

        top_k_candidates = []

        while frontier:
            _, current_node_id = heapq.heappop(frontier)
            self._visited.add(current_node_id)
            heapq.heappush(top_k_candidates, (self._get_potential(current_node_id, self._id_to_graph[current_node_id], current_node_id)))
            if len(top_k_candidates) > k:
                heapq.heappop(top_k_candidates)

            for neighbor_id, edge_type in self._get_neighbors(current_node_id):
                # when expanding neighbors, just discard ones that are too low p-value
                tentative_g_score = g_score[current_node_id] + self._init_potential - self._get_potential(current_node_id, self._id_to_graph[current_node_id])
                neighbor_g_score = g_score.get(neighbor_id, self._init_potential - self._get_potential(current_node_id, self._id_to_graph[current_node_id]))
                # g_score[neighbor_id] = neighbor_g_score # g(n) in f(n) = g(n) + h(n)
                if tentative_g_score < neighbor_g_score:
                    self._predecessors[neighbor_id] = (current_node_id, edge_type)
                    g_score[neighbor_id] = tentative_g_score
                    # f_score[neighbor_id] = tentative_g_score + self.heuristic(neighbor)
                    f_score[neighbor_id] = tentative_g_score - self._get_potential(neighbor_id, self._id_to_graph[neighbor_id])
                    heapq.heappush(frontier, (f_score[neighbor_id], neighbor_id))

                if self._cur_next_id > self._computational_budget:
                    break
        
        edge_tally = {}
        for p in top_k_candidates:
            c_id = p[0]
            while c_id in self._predecessors:
                n1, n2, _ = self._predecessors[c_id]
                if (n1, n2) not in edge_tally:
                    edge_tally[(n1, n2)] = (1, math.abs(self._ATE_cache[c_id]["ATE"] - self.ATE_init))
                else:
                    cnt = edge_tally[(n1, n2)][0] + 1
                    total_diff = edge_tally[(n1, n2)][1] + math.abs(self._ATE_cache[c_id]["ATE"] - self.ATE_init)
                    edge_tally[(n1, n2)] = (cnt, total_diff)
                current_node = self._predecessors[current_node]

        sorted_edges = sorted(edge_tally.items(), key=lambda x:x[1][0])
        print(sorted_edges[:10])

        return sorted_edges
