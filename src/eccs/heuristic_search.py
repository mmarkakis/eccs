import dihash
import heapq
import math
import networkx as nx
from typing import List, Optional, Tuple

from .ate import ATECalculator

class Graph:
    # def __init__(self, start_graph, edge_decision_matrix):
        # self._edge_decision_matrix = edge_decision_matrix # EdgeStateMatrix
        # self._start_graph = start_graph # nx.DiGraph
    def __init__(self) -> None:    
        self.adj_list = {'D': ['D']}

    def add_edge(self, u, v, weight):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append((v, weight))

    def get_neighbors(self, node):
        if node in self.adj_list:
            return self.adj_list[node]
        else:
            return []

class AStarSearch:
    def __init__(self, graph):
        self.graph = graph
        # n is the number of causal variables
        # m is the number of edges in the initial graph
        self.m = len(graph.edges())
        self.n = 0
        self.ATE_init = 0
        self.init_graph = None
        self._init_potential = 0 # TODO: initialize this
        self.treatment = None
        self.outcome = None
        self.data = None
        self.gamma_1 = 0
        self.gamma_2 = 0

        # graph id -> (pred graph id, (start causal variable, end causal variable, boolean addition or deletion))
        # TODO: make this code look nicer with edge types (low priority)
        # True is addition False is deletion
        self._predecessors = {}
        self._ATE_cache = {} # causal graph id -> ATE info
        self._visited = set()
        self._hashtag_to_id = {} # This notes that something is hashed
        self._id_to_graph = {}
        self._cur_next_id = 0
        self._computational_budget = 1000000
    
    def _get_ATE_info(self, id: int, graph: nx.DiGraph):
        try:
            return self._ATE_cache[id]
        except KeyError:
            ate = ATECalculator.get_ate_and_confidence(
                data=self.data,
                treatment=self.treatment,
                outcome=self.outcome,
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
        if ATE_info["P-value"] < 0.5: # TODO: look at this p-value threshold
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

    def astar(self, start, goal, k=100):
        # TODO: initialize ATE_init
        frontier = [(0, self._cur_next_id)] # this is the pq (f(v), v), only store the ID
        self._cur_next_id += 1

        start_hash = dihash.hash_graph(
            self.init_graph,
            hash_nodes=False,
            apply_quotient=False
        )
        self._visited[0] = self.init_graph # store actual graph here
        self._hashtag_to_id[start_hash] = 0
        g_score = {node: float('inf') for node in self.graph.adj_list}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph.adj_list}
        f_score[start] = self.heuristic(start)

        top_k_candidates = []

        while frontier:
            _, current_node_id = heapq.heappop(frontier)

            """ This is the path finding code
            if current_node == goal:
                path = []
                while current_node in self._predecessors:
                    path.append(current_node)
                    current_node = self._predecessors[current_node]
                path.append(start)
                return path[::-1]
            """
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
                    edge_tally[(n1, n2)] = 1
                else:
                    edge_tally[(n1, n2)] += 1
                current_node = self._predecessors[current_node]

        sorted_edges = sorted(edge_tally.items(), key=lambda x:x[1])
        print(sorted_edges[:10])

        return None

# Example usage:
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge('A', 'B', 5)
    graph.add_edge('A', 'C', 3)
    graph.add_edge('B', 'D', 9)
    graph.add_edge('C', 'D', 4)

    astar_search = AStarSearch(graph)
    path = astar_search.astar('A', 'D')
    print("Path found:", path)
