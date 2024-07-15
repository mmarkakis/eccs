import os
from src.generators.random_dag_generator import RandomDAGGenerator
from src.eccs.map_adj_set_to_graph import MapAdjSetToGraph
import networkx as nx
from src.eccs.graph_renderer import GraphRenderer
from src.eccs.edge_state_matrix import EdgeStateMatrix, EdgeState
from src.eccs.edges import EdgeEditType
from itertools import combinations


class TestBreakPaths:
    @staticmethod
    def set_test_directory():
        os.chdir(os.path.dirname(__file__))

    def test_one_graph(self):
        TestBreakPaths.set_test_directory()

        # Generate graph and create appropriate subdirectory for output files.
        dag = RandomDAGGenerator.generate(10, 0.5)
        name = dag["name"]
        os.makedirs(os.path.join("test_results", name), exist_ok=True)

        # Save the initial graph.
        graph = dag["graph"]
        esm = EdgeStateMatrix(list(graph.nodes()))
        esm.clear_and_set_from_graph(graph)
        GraphRenderer.save_graph(
            graph, esm, os.path.join("test_results", name, "000_starting.png")
        )

        # Break all paths in the graph.
        topo_order = list(nx.topological_sort(graph))
        print(topo_order)

        for i, source in enumerate(topo_order):
            for j, sink in enumerate(topo_order):
                if i >= j:
                    continue

                mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
                edits, success = mapper._break_paths(source, sink)
                with open(
                    os.path.join("test_results", name, f"{source}_{sink}.txt"), "w"
                ) as f:
                    f.write(f"{graph.edges()}\n")
                    if success:
                        f.write(
                            f"Successfully broke path between {source} and {sink}\n"
                        )
                        esm.clear_and_set_from_graph(graph)
                        for edit in edits:
                            assert graph.has_edge(edit.src, edit.dst)
                            assert edit.edit_type == EdgeEditType.REMOVE
                            f.write(
                                f"\tRemoving edge between {edit.src} and {edit.dst}\n"
                            )
                            esm.mark_edge(edit.src, edit.dst, EdgeState.SUGGESTED)
                        GraphRenderer.save_graph(
                            graph,
                            esm,
                            os.path.join("test_results", name, f"{source}_{sink}.png"),
                        )
                        new_graph = mapper.graph.copy()
                        for edit in edits:
                            new_graph.remove_edge(edit.src, edit.dst)
                        assert sink not in nx.descendants(new_graph, source)
                    else:
                        f.write(f"Failed to break path between {source} and {sink}\n")

    def test_one_graph_with_fixlist(self):
        TestBreakPaths.set_test_directory()

        # Generate graph and create appropriate subdirectory for output files.
        dag = RandomDAGGenerator.generate(10, 0.5)
        name = dag["name"]
        os.makedirs(os.path.join("test_results", name), exist_ok=True)

        # Save the initial graph.
        graph = dag["graph"]
        esm = EdgeStateMatrix(list(graph.nodes()))
        esm.clear_and_set_from_graph(graph)
        GraphRenderer.save_graph(
            graph, esm, os.path.join("test_results", name, "000_starting.png")
        )

        # Break all paths in the graph.
        topo_order = list(nx.topological_sort(graph))
        print(topo_order)

        for i, source in enumerate(topo_order):
            for j, sink in enumerate(topo_order):
                if i >= j:
                    continue

                mapper = MapAdjSetToGraph(graph, source, sink, [], [], None)
                edits, success = mapper._break_paths(source, sink)
                if success and len(edits) > 0:
                    mapper.update_fixlist([(edits[0].src, edits[0].dst)])
                edits, success = mapper._break_paths(source, sink)

                with open(
                    os.path.join("test_results", name, f"{source}_{sink}.txt"), "w"
                ) as f:
                    f.write(f"Edges: {graph.edges()}\n")
                    f.write(f"Fixlist: {mapper.fix_list}\n")
                    if success:
                        f.write(
                            f"Successfully broke path between {source} and {sink}\n"
                        )
                        esm.clear_and_set_from_graph(graph)
                        for edit in edits:
                            assert graph.has_edge(edit.src, edit.dst)
                            assert edit.edit_type == EdgeEditType.REMOVE
                            f.write(
                                f"\tRemoving edge between {edit.src} and {edit.dst}\n"
                            )
                            esm.mark_edge(edit.src, edit.dst, EdgeState.SUGGESTED)
                        GraphRenderer.save_graph(
                            graph,
                            esm,
                            os.path.join("test_results", name, f"{source}_{sink}.png"),
                        )
                        new_graph = mapper.graph.copy()
                        for edit in edits:
                            new_graph.remove_edge(edit.src, edit.dst)
                        assert sink not in nx.descendants(new_graph, source)
                    else:
                        f.write(f"Failed to break path between {source} and {sink}\n")

    def test_graph_size_sweep(self):
        TestBreakPaths.set_test_directory()

        # Generate graph and create appropriate subdirectory for output files.
        for num_nodes in [10,20,50,100,200,500,1000]:
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
    
