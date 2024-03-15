import pandas as pd
import sys

sys.path.append("../..")
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.HiddenCausal.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.graph.Endpoint import Endpoint
from datetime import datetime
import networkx as nx
from typing import Callable, Optional
from ..eccs.graph_renderer import GraphRenderer
from ..eccs.edge_state_matrix import EdgeStateMatrix, EdgeState
import asyncio
import os


DEFAULT_TIMEOUT_SECONDS = 30 * 60  # 30 mintues

METHODS_OPTIONS = {
    "pc": [
        "fisherz",
        "mv_fisherz",
        "mc_fisherz",
        "kci",
        "chisq",
        "gsq",
        "d_separation",
    ],
    "fci": ["fisherz", "kci", "chisq", "gsq"],
    "lingam": ["default"],
    "gin": ["kci", "hsic"],
    "grasp": [
        "local_score_CV_general",
        "local_score_marginal_general",
        "local_score_CV_multi",
        "local_score_marginal_multi",
        "local_score_BIC",
        "local_score_BDeu",
    ],
    "ges": [
        "local_score_CV_general",
        "local_score_marginal_general",
        "local_score_CV_multi",
        "local_score_marginal_multi",
        "local_score_BIC",
        "local_score_BIC_from_cov",
        "local_score_BDeu",
    ],
    "exact_search": ["dp", "astar"],
}


class CausalDiscovery:
    """
    A class that provides several causal discovery algorithms to use for generating evaluation graphs.

    """

    def __init__(self, dataset_name: str, dataset: str | pd.DataFrame) -> None:
        """
        Initializes the CausalDiscovery object.

        Parameters:
            dataset_name: The name of the dataset.
            dataset: The dataset or a path to it.

        """
        self._dataset_name = dataset_name

        if isinstance(dataset, str):
            data_df = pd.read_csv(self.dataset)
            self._var_names = data_df.columns
            self._data = data_df.to_numpy().astype(float)
        else:
            self._var_names = list(dataset.columns)
            self._data = dataset.to_numpy().astype(float)

    async def run_with_timer(
        self, method: str, option: str, out_path: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> tuple[nx.DiGraph, float]:
        """
        Runs the causal discovery `method` with the specified `option`,
        with the given timeout (in seconds) and returns the resulting causal graph and timing.

        Parameters:
            method: The causal discovery method to run.
            option: The option to use for the causal discovery method.
            out_path: The path to save the results.
            timeout_seconds: The maximum time to allow the method to run.

        Returns:
            The resulting causal graph and timing.
        """

        if (
            method not in METHODS_OPTIONS.keys()
            or option not in METHODS_OPTIONS[method]
        ):
            raise ValueError(f"Invalid method or option: {method}, {option}")

        # Open logging files
        fout = open(f"discovery.log", "a")
        sys.stdout = fout
        sys.stderr = fout
        fres = open(f"discovery-results.csv", "a")
        fres.write(f"dataset_name,method_name,option,result,time\n")
        fres.flush()

        

        ##############################

        # Define causal discovery methods

        def general_graph_to_nx_digraph(gg_cg):
            # Convert the graph to a NetworkX DiGraph
            nx_cg = nx.DiGraph()
            for edge in gg_cg.get_graph_edges():
                node1 = edge.get_node1().get_name()
                node2 = edge.get_node2().get_name()
                points_left = edge.get_endpoint1() == Endpoint.ARROW and (
                    edge.get_endpoint2() == Endpoint.TAIL
                    or edge.get_endpoint2() == Endpoint.CIRCLE
                )
                points_right = edge.get_endpoint2() == Endpoint.ARROW and (
                    edge.get_endpoint1() == Endpoint.TAIL
                    or edge.get_endpoint1() == Endpoint.CIRCLE
                )
                if not points_left:  # points right or undirected
                    nx_cg.add_edge(node1, node2)
                if not points_right:  # points left or undirected
                    nx_cg.add_edge(node2, node1)

            return nx_cg

        def run_pc(option):
            cg = pc(self._data, indep_test=option, show_progress=False)
            cg.to_nx_graph()
            return cg.nx_graph

        def run_fci(option):
            cg, _ = fci(self._data, independence_test_method=option, show_progress=False)
            nx_cg = general_graph_to_nx_digraph(cg)
            return nx_cg

        def run_lingam(option):
            model = lingam.ICALiNGAM()
            model.fit(self._data)
            gv_cg = make_dot(model.adjacency_matrix_, labels=self._var_names)

            # Convert the Graphviz Digraph to NetworkX DiGraph
            nx_cg = nx.DiGraph()
            for line in gv_cg.body:
                tokens = line.strip(";").split("->")
                if len(tokens) == 2:
                    nx_cg.add_edge(tokens[0].strip(), tokens[1].strip())
                tokens = tokens[0].split("--")
                if len(tokens) == 2:
                    nx_cg.add_edge(tokens[0].strip(), tokens[1].strip())
                    nx_cg.add_edge(tokens[1].strip(), tokens[0].strip())
            return nx_cg

        def run_gin(option):
            cg, _ = GIN.GIN(self._data, indep_test_method=option)
            nx_cg = general_graph_to_nx_digraph(cg)
            return nx_cg

        def run_grasp(option):
            cg = grasp(self._data, score_func=option)
            nx_cg = general_graph_to_nx_digraph(cg)
            return nx_cg

        def run_ges(option):
            record = ges(self._data, score_func=option)
            nx_cg = general_graph_to_nx_digraph(record["G"])
            return nx_cg

        def run_exact_search(option):
            matrix = bic_exact_search(self._data, search_method=option)
            # Convert the matrix to a NetworkX DiGraph
            # If matrix[i,j] == 1, then there is an edge from i to j
            nx_cg = nx.DiGraph()
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] == 1:
                        nx_cg.add_edge(i, j)
            return nx_cg

        # Run a function within a try except block
        async def run_safe(function: Callable) -> Optional[nx.DiGraph]:
            """
            Run the specified causal discovery method with the speficied option within a
            try-except block to catch any exceptions.

            Parameters:
                function: The function to run.

            Returns:
                The resulting causal graph, if successful, or None otherwise.
            """

            try:
                starttime = datetime.now()
                nx_cg = function(option=option)

                # Check if the graph is empty
                if nx_cg.number_of_edges() == 0:
                    fres.write(
                        f"{self._dataset_name},{method},{option},empty,{(datetime.now() - starttime).seconds}\n"
                    )
                    fres.flush()
                    return

                # Remove isolated nodes
                nx_cg.remove_nodes_from(list(nx.isolates(nx_cg)))

                print(f"{datetime.now()} Writing out {method} with {option}")
                sys.stdout.flush()
                fres.write(
                    f"{self._dataset_name},{method},{option},success,{(datetime.now() - starttime).seconds}\n"
                )
                fres.flush()

                # Save the graph as png
                esm = EdgeStateMatrix(self._var_names)
                for src, dst in nx_cg.edges():
                    esm.mark_edge(src, dst, EdgeState.PRESENT)
                GraphRenderer.save_graph(
                    nx_cg, esm, os.path.join(out_path, f"{self._dataset_name}_{method}_{option}.png")
                )

                # Write out graph in dot format
                nx.nx_pydot.write_dot(
                    nx_cg, os.path.join(out_path, f"{self._dataset_name}_{method}_{option}.dot")
                )

                return nx_cg
            except Exception as e:
                print(f"{datetime.now()} Error running {function}: {e}")
                sys.stdout.flush()
                fres.write(f"{self._dataset_name},{method},{option},exception: {e},\n")
                fres.flush()
                return None

        ##############################

        # Run the discovery
        print(f"{datetime.now()} Running {method} with {option}")
        sys.stdout.flush()
        nx_cg = None
        start_time = datetime.now()
        try:
            nx_cg = await asyncio.wait_for(
                run_safe(locals()[f"run_{method}"]), timeout_seconds
            )
        except TimeoutError:
            print(f"{datetime.now()} Function timed out")
            fres.write(
                f"{self._dataset_name},{method},{option},timeout,{timeout_seconds}\n"
            )
            fres.flush()
        end_time = datetime.now()
        duration = min((end_time - start_time).seconds, timeout_seconds)

        print(
            f"{datetime.now()} Done running {method} with {option}. Took {duration:.3f} seconds"
        )
        sys.stdout.flush()

        fout.close()
        fres.close()

        return nx_cg, duration
