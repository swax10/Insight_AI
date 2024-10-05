import requests
import json
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import random
import matplotlib.pyplot as plt
import community as community_louvain
import plotly.graph_objects as go
import pickle
import os

print(os.getcwd())
from rd import (
    remove_duplicate_edges_with_ollama,
    create_prompt_for_duplicates,
    query_ollama,
)


class GraphVisualization:
    def __init__(self, df1, df2):
        self.dfg1 = pd.merge(
            df1, df2[["chunk_id", "source"]], on="chunk_id", how="left"
        )
        self.G = nx.Graph()
        self.graph_output_directory = "index.html"

    def clean_data(self):
        self.dfg1.replace("", np.nan, inplace=True)
        self.dfg1.dropna(subset=["node_1", "node_2"], inplace=True)

    def process_graph(self):
        dfg = self.dfg1.copy()
        dfg["chunk_id"] = dfg["chunk_id"].astype(str)
        dfg["edge"] = dfg["edge"].astype(str)

        dfg = (
            dfg.groupby(["node_1", "node_2"])
            .agg({"chunk_id": ",".join, "edge": ",".join})
            .reset_index()
        )

        nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()

        for node in nodes:
            self.G.add_node(str(node))

        for index, row in dfg.iterrows():
            self.G.add_edge(
                str(row["node_1"]), str(row["node_2"]), title=row["edge"], weight=1
            )

        community_partition = self.detect_communities()
        for community in community_partition:
            most_connected_node = max(community, key=lambda node: self.G.degree[node])
            source_rows = self.dfg1[self.dfg1["node_1"] == most_connected_node]
            if not source_rows.empty:
                source_node = str(source_rows["source"].values[0])
                self.G.add_node(source_node)
                self.G.add_edge(
                    most_connected_node, source_node, title="sourced from", weight=1
                )

    def detect_communities(self):
        partition = community_louvain.best_partition(self.G)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        return list(communities.values())

    def assign_colors_to_communities(self, communities):
        p = sns.color_palette("hls", len(communities)).as_hex()
        random.shuffle(p)

        rows = []
        group = 0
        for community in communities:
            color = p.pop()
            group += 1
            for node in community:
                rows += [{"node": node, "color": color, "group": group}]

        df_colors = pd.DataFrame(rows)

        for index, row in df_colors.iterrows():
            self.G.nodes[row["node"]]["group"] = row["group"]
            self.G.nodes[row["node"]]["color"] = row["color"]
            self.G.nodes[row["node"]]["size"] = self.G.degree[row["node"]]

    def save_graph(self, filename="graph.pkl"):
        """Save the graph to a file using Pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self.G, f)
        print(f"Graph saved to {filename}")

    def load_graph(self, filename="graph.pkl"):
        """Load the graph from a file using Pickle."""
        try:
            with open(filename, "rb") as f:
                self.G = pickle.load(f)
            print(f"Graph loaded from {filename}")
        except FileNotFoundError:
            print(f"No file found at {filename}. Please check the path or file name.")

    def visualize_graph(self):
        pos = nx.spring_layout(self.G, k=0.1)

        node_x = [pos[node][0] for node in self.G.nodes()]
        node_y = [pos[node][1] for node in self.G.nodes()]
        colors = [self.G.nodes[node]["color"] for node in self.G.nodes()]

        degrees = np.array([self.G.degree[node] for node in self.G.nodes()])

        min_size, max_size = 10, 50
        if len(degrees) > 1:
            normalized_sizes = min_size + (degrees - degrees.min()) * (
                max_size - min_size
            ) / (degrees.max() - degrees.min())
        else:
            normalized_sizes = np.full_like(degrees, min_size)

        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color="black"),
                hoverinfo="none",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(size=normalized_sizes, color=colors, line=dict(width=1)),
                hoverinfo="none",
            )
        )

        fig.update_layout(
            title="Interactive Graph Visualization",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        fig.write_html("interactive_graph.html")
        fig.show()

    def get_connected_nodes_with_relationships(self, node):
        """Retrieve all nodes connected to a specific node and their relationships."""
        if node not in self.G:
            print(f"Node {node} not found in the graph.")
            return None

        connected_nodes = list(self.G[node].items())
        result = [
            {"connected_node": n, "relationship": data.get("title", "unknown")}
            for n, data in connected_nodes
        ]
        return result

    def remove_duplicates_in_batches(self):
        edges = [
            {
                "node1": edge[0],
                "node2": edge[1],
                "relation": self.G.edges[edge]["title"],
            }
            for edge in self.G.edges()
        ]
        unique_edges = []

        # Send edges in batches of 25
        for i in range(0, len(edges), 25):
            batch = edges[i : i + 25]
            edges_json = json.dumps({"edges": batch})
            unique_edges_json = remove_duplicate_edges_with_ollama(edges_json)
            unique_edges_data = json.loads(unique_edges_json)
            unique_edges.extend(unique_edges_data["edges"])

        # Clear the existing graph and add unique edges
        self.G.clear()
        for edge in unique_edges:
            self.G.add_edge(
                edge["node1"], edge["node2"], title=edge["relation"], weight=1
            )

    def run(self):
        self.clean_data()
        self.process_graph()
        self.remove_duplicates_in_batches()  # Remove duplicates in batches
        communities = self.detect_communities()
        self.assign_colors_to_communities(communities)
        self.visualize_graph()
        self.save_graph()  # Save the graph after visualization
        return self.G


# Usage example
if __name__ == "__main__":
    df1 = pd.read_csv(
        "data_output/research_papers/graph.csv", sep="|", low_memory=False
    )  # Adjust this line according to your data source
    df2 = pd.read_csv(
        "data_output/research_papers/chunks.csv", sep="|", low_memory=False
    )  # Adjust this line according to your data source
    gv = GraphVisualization(df1, df2)
    # gv.run()  # Uncomment to run the entire process
    gv.load_graph("pain_graph.pkl")
    gv.remove_duplicates_in_batches()
    connected_nodes = gv.get_connected_nodes_with_relationships("pain")
    print(connected_nodes)
    # number of nodes
    print(f"Number of nodes: {gv.G.number_of_nodes()}")
