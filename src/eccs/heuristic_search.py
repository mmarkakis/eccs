import heapq
import math
import networkx as nx
from typing import Any, Optional, Tuple

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
        self._predecessors = {}
        self._ATE_cache = {} # causal graph id -> ATE info
        self._visited = {}
    
    def _get_potential(self, v: Tuple[int, nx.DiGraph]):
        """
        v is a (id: int, corresponding causal graph: nx.DiGraph)
        Phi(v) = |ATE_v - ATE_init| - gamma_1 * regression error + gamma_2 * other heuristic
        For now, use |number of edges in v - n| as the "other heuristic"
        Could plug in other forms of assessment in the literatures here
        """
        # {"ATE": float, "P-value": float, "Standard Error": float, "Estimand": ?}
        # TODO: what is the type of Estimand
        # TODO: implement caching for ATE information
        ATE_info = self._ATE_cache.get(
            v[0],
            ATECalculator.get_ate_and_confidence(
                data=self.data,
                treatment=self.treatment,
                outcome=self.outcome,
                graph=v[1],
                calculate_p_value=True,
                calculate_std_error=True,
                get_estimand=False,
            )
        )
        return ATE_info["ATE"] - self.gamma_1 * ATE_info["Standard Error"] + ATE_info["P-value"] + math.abs(len(v.edges() - self.n))

    def heuristic(self, predecessor, node):
        # predecessor was taken from the frontier pq, node is the candidate for expansion
        # the weight of an edge (v1, v2) in the A* graph is Phi(v1) - Phi(v2) since we want to
        # go towards the maximal Phi causal graph.
        # A* heuristic f(n) = g(n) + h(n). g(n) is the cost of this explored path, h(n) is a heuristical
        # estimate of the cost from here to the destination. To be admissible, h(n) cannot overestimate the true
        # value. Therefore, we use -Phi(v2) as an estimate for now.
        # g(n) = Phi(init) - Phi(current frontier of the path)
        return self._init_potential - self._get_potential(predecessor) - self._get_potential(node)

    def astar(self, start, goal):
        # TODO: initialize ATE_init
        open_list = [(0, 0)] # this is the pq (f(v), v), only store the ID
        self._visited[0] = self.init_graph # store actual graph here
        g_score = {node: float('inf') for node in self.graph.adj_list}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph.adj_list}
        f_score[start] = self.heuristic(start)

        while open_list:
            current_f, current_node = heapq.heappop(open_list)

            if current_node == goal:
                path = []
                while current_node in self._predecessors:
                    path.append(current_node)
                    current_node = self._predecessors[current_node]
                path.append(start)
                return path[::-1]

            for neighbor, cost in self.graph.get_neighbors(current_node):
                # when expanding neighbors, just discard ones that are too low p-value
                tentative_g_score = g_score[current_node] + cost
                if tentative_g_score < g_score[neighbor]:
                    self._predecessors[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

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
