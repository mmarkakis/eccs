import networkx as nx
import base64
from io import BytesIO
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from .edge_state_matrix import EdgeStateMatrix


class GraphRenderer:
    """
    Render a digraph with appropriate margins and node tags.
    """

    @staticmethod
    def draw_graph(graph: nx.DiGraph, esm: EdgeStateMatrix) -> str:
        """
        Draw a graph with appropriate margins and node tags.

        Parameters:
            graph: The graph to be drawn.
            esm: The edge state matrix to be used to determine
                the color of the edges.

        Returns:
            A base64-encoded string representation of the graph.
        """
        if graph.number_of_nodes() == 0:
            return ""

        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            edgelist=graph.edges(),
            with_labels=False,
            width=2.0,
            node_color="#d3d3d3",
        )
        text = nx.draw_networkx_labels(graph, pos, font_size=12)
        for _, t in text.items():
            t.set_rotation(30)

        # Color the edges based on the edge state matrix
        # Edges are green if they are accepted and orange if they are undecided
        for i in range(esm.n):
            for j in range(esm.n):
                if esm.m[i, j] == 1:
                    graph[esm.name(i)][esm.name(j)]["color"] = "#00FF25"
                elif esm.m[i, j] == 0:
                    graph[esm.name(i)][esm.name(j)]["color"] = "#FFA500"

        # Fix margins
        x_values, y_values = zip(*pos.values())
        x_max, x_min = max(x_values), min(x_values)
        y_max, y_min = max(y_values), min(y_values)
        if x_max != x_min:
            x_margin = (x_max - x_min) * 0.3
            plt.xlim(x_min - x_margin, x_max + x_margin)
        if y_max != y_min:
            y_margin = (y_max - y_min) * 0.3
            plt.ylim(y_min - y_margin, y_max + y_margin)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.clf()
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return img_str

    @staticmethod
    def save_graph(graph: nx.DiGraph, esm: EdgeStateMatrix, filename: str) -> None:
        """
        Save the graph to a file as a png image.

        Parameters:
            graph: The graph to be saved.
            esm: The edge state matrix to be used to determine
                the color of the edges.
            filename: The name of the file to which the graph should be saved.
        """
        img_str = GraphRenderer.draw_graph(graph, esm)
        with open(filename, "wb") as f:
            f.write(base64.b64decode(img_str))

    @staticmethod
    def graph_string_to_html(graph: str) -> HTML:
        """
        Convert the string representation of the rgaph to an HTML object

        Parameters:
            graph: The graph to be displayed.
        """
        return HTML(
            '<img src="data:image/png;base64,{}" style="max-width: 100%; height: auto;">'.format(
                graph
            )
        )

    @staticmethod
    def display_graph(graph: nx.DiGraph, esm: EdgeStateMatrix) -> None:
        """
        Display the graph.

        Parameters:
            graph: The graph to be displayed.
            esm: The edge state matrix to be used to determine
                the color of the edges.
        """
        display(
            GraphRenderer.graph_string_to_html(GraphRenderer.draw_graph(graph, esm))
        )
