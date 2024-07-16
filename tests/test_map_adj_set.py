import networkx as nx
import random
import sys

from itertools import combinations

from src.eccs.eccs import ECCS
from src.eccs.edges import EdgeEditType
from src.eccs.map_adj_set_to_graph import MapAdjSetToGraph
from src.generators.random_dag_generator import RandomDAGGenerator


class TestMapAdjSet:
    @staticmethod
    def _do_edits(graph, edits):
        """
        Apply the given `edits` to `graph` and return the new graph.
        """
        for edit in edits:
            if edit.edit_type == EdgeEditType.ADD:
                assert not graph.has_edge(edit.src, edit.dst)
                graph.add_edge(edit.src, edit.dst)
            elif edit.edit_type == EdgeEditType.REMOVE:
                assert graph.has_edge(edit.src, edit.dst)
                graph.remove_edge(edit.src, edit.dst)
            elif edit.edit_type == EdgeEditType.FLIP:
                assert graph.has_edge(edit.src, edit.dst)
                graph.remove_edge(edit.src, edit.dst)
                graph.add_edge(edit.dst, edit.src)
        return graph

    @staticmethod
    def _undo_edits(graph, edits):
        """
        Apply the opposite of `edits` to `graph` and return the new graph.
        """
        for edit in edits:
            if edit.edit_type == EdgeEditType.ADD:
                assert graph.has_edge(edit.src, edit.dst)
                graph.remove_edge(edit.src, edit.dst)
            elif edit.edit_type == EdgeEditType.REMOVE:
                assert not graph.has_edge(edit.src, edit.dst)
                graph.add_edge(edit.src, edit.dst)
            elif edit.edit_type == EdgeEditType.FLIP:
                assert graph.has_edge(edit.dst, edit.src)
                graph.remove_edge(edit.dst, edit.src)
                graph.add_edge(edit.src, edit.dst)
        return graph

    @staticmethod
    def _test_opt_map_removal_core(
        num_nodes=20, edge_prob=0.5, fix_fraction=0, ban_fraction=0
    ):
        """
        Core test function for the optimized map_removal function.

        Parameters:
            num_nodes: Number of nodes in the graph.
            edge_prob: Edge presence probability.
            fix_fraction: Fraction of edges to fix after first attempt.
            ban_fraction: Fraction of edges to ban after first attempt.
        """

        graph = RandomDAGGenerator.generate(num_nodes, edge_prob)["graph"]
        topo_order = list(nx.topological_sort(graph))

        for treatment, outcome in combinations(topo_order, 2):
            base_adj_set = ECCS._find_adjustment_set(graph, treatment, outcome)
            if base_adj_set is None or len(base_adj_set) == 0:
                continue
            mapper = MapAdjSetToGraph(graph, treatment, outcome, [], [], base_adj_set)

            for v in base_adj_set:
                edits = mapper.map_removal(v, use_optimized=True)
                if len(edits) == 0:
                    continue

                if fix_fraction > 0:
                    removals = [
                        (e.src, e.dst)
                        for e in edits
                        if e.edit_type == EdgeEditType.REMOVE
                    ]
                    num_fixes = int(fix_fraction * len(removals))
                    fixlist = random.sample(removals, num_fixes)
                    mapper.update_fixlist(fixlist)
                    edits = mapper.map_removal(v, use_optimized=True)
                    if len(edits) == 0:
                        continue

                if ban_fraction > 0:
                    additions = [
                        (e.src, e.dst) for e in edits if e.edit_type == EdgeEditType.ADD
                    ]
                    num_bans = int(ban_fraction * len(additions))
                    banned = random.sample(additions, num_bans)
                    mapper.update_banlist(banned)
                    edits = mapper.map_removal(v, use_optimized=True)
                    if len(edits) == 0:
                        continue

                graph = TestMapAdjSet._do_edits(graph, edits)
                new_adj_set = ECCS._find_adjustment_set(graph, treatment, outcome)
                graph = TestMapAdjSet._undo_edits(graph, edits)
                assert v not in new_adj_set

    def test_opt_map_removal_one_graph(self):
        """
        Test the algorithm on a graph of fixed size and edge presence probability.
        """
        TestMapAdjSet._test_opt_map_removal_core()

    def test_opt_map_removal_graph_size_sweep(self):
        """
        Test the algorithm on graphs of varying sizes.
        """

        for num_nodes in [10, 20, 50]:
            print(f"Testing graph size {num_nodes}")
            sys.stdout.flush()
            TestMapAdjSet._test_opt_map_removal_core(num_nodes=num_nodes)

    def test_opt_map_removal_edge_prob_sweep(self):
        """
        Test the algorithm on graphs of varying edge presence probabilities.
        """

        for edge_prob in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_removal_core(edge_prob=edge_prob)

    def test_opt_map_removal_fixlist_sweep(self):
        """
        Fix a certain proportion of the edges that the algorithm suggested removing,
        and then run the algorithm again. Ensure that the algorithm still produces
        an adjustment set with the desired properties.
        """

        for fix_fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_removal_core(fix_fraction=fix_fraction)

    def test_opt_map_removal_banlist_sweep(self):
        """
        Ban a certain proportion of the edges that the algorithm suggested adding,
        and then run the algorithm again. Ensure that the algorithm still produces
        an adjustment set with the desired properties.
        """

        for ban_fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_removal_core(ban_fraction=ban_fraction)


    @staticmethod
    def _test_opt_map_addition_core(
        num_nodes=20, edge_prob=0.5, fix_fraction=0, ban_fraction=0
    ):
        """
        Core test function for the optimized map_addition function.

        Parameters:
            num_nodes: Number of nodes in the graph.
            edge_prob: Edge presence probability.
            fix_fraction: Fraction of edges to fix after first attempt.
            ban_fraction: Fraction of edges to ban after first attempt.
        """

        graph = RandomDAGGenerator.generate(num_nodes, edge_prob)["graph"]
        topo_order = list(nx.topological_sort(graph))

        for treatment, outcome in combinations(topo_order, 2):
            base_adj_set = ECCS._find_adjustment_set(graph, treatment, outcome)
            if base_adj_set is None or len(base_adj_set) == 0:
                continue
            mapper = MapAdjSetToGraph(graph, treatment, outcome, [], [], base_adj_set)

            for v in set(graph.nodes) - set(base_adj_set) - set([treatment, outcome]):
                edits = mapper.map_addition(v, use_optimized=True)
                if len(edits) == 0:
                    continue

                if fix_fraction > 0:
                    removals = [
                        (e.src, e.dst)
                        for e in edits
                        if e.edit_type == EdgeEditType.REMOVE
                    ]
                    num_fixes = int(fix_fraction * len(removals))
                    fixlist = random.sample(removals, num_fixes)
                    mapper.update_fixlist(fixlist)
                    edits = mapper.map_addition(v, use_optimized=True)
                    if len(edits) == 0:
                        continue

                if ban_fraction > 0:
                    additions = [
                        (e.src, e.dst) for e in edits if e.edit_type == EdgeEditType.ADD
                    ]
                    num_bans = int(ban_fraction * len(additions))
                    banned = random.sample(additions, num_bans)
                    mapper.update_banlist(banned)
                    edits = mapper.map_addition(v, use_optimized=True)
                    if len(edits) == 0:
                        continue

                graph = TestMapAdjSet._do_edits(graph, edits)
                new_adj_set = ECCS._find_adjustment_set(graph, treatment, outcome)
                graph = TestMapAdjSet._undo_edits(graph, edits)
                assert v in new_adj_set
    
    def test_opt_map_addition_one_graph(self):
        """
        Test the algorithm on a graph of fixed size and edge presence probability.
        """
        TestMapAdjSet._test_opt_map_addition_core()

    def test_opt_map_addition_graph_size_sweep(self):
        """
        Test the algorithm on graphs of varying sizes.
        """

        for num_nodes in [10, 20, 50]:
            print(f"Testing graph size {num_nodes}")
            sys.stdout.flush()
            TestMapAdjSet._test_opt_map_addition_core(num_nodes=num_nodes)

    def test_opt_map_addition_edge_prob_sweep(self):
        """
        Test the algorithm on graphs of varying edge presence probabilities.
        """

        for edge_prob in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_addition_core(edge_prob=edge_prob)

    def test_opt_map_addition_fixlist_sweep(self):
        """
        Fix a certain proportion of the edges that the algorithm suggested removing,
        and then run the algorithm again. Ensure that the algorithm still produces
        an adjustment set with the desired properties.
        """

        for fix_fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_addition_core(fix_fraction=fix_fraction)

    def test_opt_map_addition_banlist_sweep(self):
        """
        Ban a certain proportion of the edges that the algorithm suggested adding,
        and then run the algorithm again. Ensure that the algorithm still produces
        an adjustment set with the desired properties.
        """

        for ban_fraction in [0.1, 0.25, 0.5, 0.75, 0.9]:
            TestMapAdjSet._test_opt_map_addition_core(ban_fraction=ban_fraction)
            