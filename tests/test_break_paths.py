import networkx as nx
import random

from itertools import combinations

from src.eccs.edges import EdgeEditType
from src.eccs.map_adj_set_to_graph import MapAdjSetToGraph
from src.generators.random_dag_generator import RandomDAGGenerator


class TestBreakPaths:
    def test_one_graph(self):
        """
        Test the algorithm on a graph of fixed size and edge presence probability.
        """
        graph = RandomDAGGenerator.generate(10, 0.5)["graph"]
        topo_order = list(nx.topological_sort(graph))

        for source, sink in combinations(topo_order, 2):
            mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
            edits, success = mapper._break_paths(source, sink)
            if success:
                new_graph = mapper.graph.copy()
                for edit in edits:
                    assert graph.has_edge(edit.src, edit.dst)
                    assert edit.edit_type == EdgeEditType.REMOVE
                    new_graph.remove_edge(edit.src, edit.dst)
                assert sink not in nx.descendants(new_graph, source)

    def test_graph_size_sweep(self):
        """
        Test the algorithm on graphs of varying sizes.
        """

        for num_nodes in [10, 20, 50, 100]:
            graph = RandomDAGGenerator.generate(num_nodes, 0.5)["graph"]
            topo_order = list(nx.topological_sort(graph))

            for source, sink in combinations(topo_order, 2):
                mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
                edits, success = mapper._break_paths(source, sink)
                if success:
                    new_graph = mapper.graph.copy()
                    for edit in edits:
                        assert graph.has_edge(edit.src, edit.dst)
                        assert edit.edit_type == EdgeEditType.REMOVE
                        new_graph.remove_edge(edit.src, edit.dst)
                    assert sink not in nx.descendants(new_graph, source)

    def test_edge_prob_sweep(self):
        """
        Test the algorithm on graphs of varying edge presence probabilities.
        """

        for edge_prob in [0.1, 0.25, 0.5, 0.75, 0.9]:
            graph = RandomDAGGenerator.generate(20, edge_prob)["graph"]
            topo_order = list(nx.topological_sort(graph))

            for source, sink in combinations(topo_order, 2):
                mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
                edits, success = mapper._break_paths(source, sink)
                if success:
                    new_graph = mapper.graph.copy()
                    for edit in edits:
                        assert graph.has_edge(edit.src, edit.dst)
                        assert edit.edit_type == EdgeEditType.REMOVE
                        new_graph.remove_edge(edit.src, edit.dst)
                    assert sink not in nx.descendants(new_graph, source)

    def test_fixlist_sweep(self):
        """
        Fix a certain proportion of the edges that the algorithm suggested removing,
        and then run the algorithm again. Ensure that the algorithm still successfully
        breaks the path between source and sink, and that none of the edges in the fixlist
        are removed.
        """

        for fix_prop in [0.1, 0.25, 0.5, 0.75, 0.9]:
            graph = RandomDAGGenerator.generate(20, 0.5)["graph"]
            topo_order = list(nx.topological_sort(graph))

            for source, sink in combinations(topo_order, 2):
                mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
                edits, success = mapper._break_paths(source, sink)

                if not success or len(edits) == 0:
                    continue

                # Select some random edges to add to the fixlist
                random.shuffle(edits)
                fixlist = [(edit.src, edit.dst) for edit in edits[: int(fix_prop * len(edits))]]
                mapper.update_fixlist(fixlist)
                edits, success = mapper._break_paths(source, sink)

                if success:
                    new_graph = mapper.graph.copy()
                    for edit in edits:
                        assert graph.has_edge(edit.src, edit.dst)
                        assert (edit.src, edit.dst) not in fixlist
                        assert edit.edit_type == EdgeEditType.REMOVE
                        new_graph.remove_edge(edit.src, edit.dst)
                    assert sink not in nx.descendants(new_graph, source)
