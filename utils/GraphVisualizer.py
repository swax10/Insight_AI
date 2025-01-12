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
import time
from tqdm import tqdm
from pyvis.network import Network
import glob


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

        # Group data to merge edges and chunk_ids for the same node pairs
        dfg = (
            dfg.groupby(["node_1", "node_2"])
            .agg({"chunk_id": ",".join, "edge": ",".join})
            .reset_index()
        )

        nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()

        # Add nodes to the graph
        for node in nodes:
            self.G.add_node(str(node))

        # Add edges to the graph
        for index, row in dfg.iterrows():
            self.G.add_edge(
                str(row["node_1"]), str(row["node_2"]), title=row["edge"], weight=1
            )

        #remove duplicates
        #self.remove_duplicates_in_batches()
        # Detect communities in the graph
        community_partition = self.detect_communities()

        # Add source nodes and edges to the graph
        for community in community_partition:
            most_connected_node = max(community, key=lambda node: self.G.degree[node])

            # Filter to get the rows where node_1 is the most connected node
            source_rows = self.dfg1[self.dfg1["node_1"] == most_connected_node]

            if not source_rows.empty:
                source_node = str(source_rows["source"].values[0])

                if source_node != 'nan':  # Check if source node exists and is valid
                    # Add the source node if it doesn't already exist
                    if not self.G.has_node(source_node):
                        self.G.add_node(source_node)

                    # Add an edge between the most connected node and the source node
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
        net = Network(notebook=False, height='750px', width='100%', bgcolor='#ffffff', font_color='black')

        # Set options for better visualization (you can customize this further)
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 16,
            "font": {
              "size": 14,
              "face": "Tahoma"
            }
          },
          "edges": {
            "width": 0.15,
            "color": {
              "inherit": true
            },
            "smooth": {
              "type": "continuous"
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)

        # Add nodes with hover text and color
        for node, node_data in self.G.nodes(data=True):
            # Get connected nodes and their relationships
            connected_info = self.get_connected_nodes_with_relationships(node)
            relationships = ", ".join(
                [f"{info['connected_node']} ({info['relationship']})" for info in connected_info]
            ) if connected_info else "No connections"

            title = f"Node {node}: {node_data.get('label', '')}\nConnections: {relationships}"
            color = node_data.get('color', 'blue')  # Default color if not provided
            net.add_node(node, label=str(node), title=title, color=color)

        # Add edges with hover text showing relationship and weight
        for node1, node2, edge_data in self.G.edges(data=True):
            title = f"Edge {node1} - {node2}: {edge_data.get('title', 'unknown')} (Weight: {edge_data.get('weight', 1)})"
            net.add_edge(node1, node2, title=title, value=edge_data.get('weight', 1))  # Default weight 1 if none exists

        net.write_html("interactive_graph.html")
        # Open HTML file in browser
        os.system("start interactive_graph.html")


    def get_connected_nodes_with_relationships(self, node, depth=1):
        """Retrieve all nodes connected to a specific node, their relationships, 
        and sub-connected nodes up to a specified depth."""

        if node not in self.G:
            print(f"Node {node} not found in the graph.")
            return None

        def recursive_retrieval(current_node, current_depth):
            if current_depth > depth:
                return []

            connected_nodes = list(self.G[current_node].items())
            result = []

            for n, data in connected_nodes:
                result.append({
                    "connected_node": n,
                    "relationship": data.get("title", "unknown"),
                    "sub_connected": recursive_retrieval(n, current_depth + 1)
                })

            return result

        return recursive_retrieval(node, 1)

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
        for i in tqdm(range(0, len(edges), 50)):
            batch = edges[i : i + 50]
            edges_json = json.dumps({"edges": batch})

            # Retry logic directly within the loop
            for attempt in range(5):  # Retry up to 5 times
                try:
                    unique_edges_json = remove_duplicate_edges_with_ollama(edges_json)
                    break  # If successful, break out of retry loop
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for batch {i}: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to process batch {i} after 5 attempts, skipping")
                continue

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
    gv.load_graph("graph.pkl")
    x = gv.get_connected_nodes_with_relationships("llms")
    print(x)
    #gv.run()
    #list_of_files = []
    #for file in glob.glob("research_papers/*.pdf"):
    #    #replace \\ with \ in the path 
    #    file = str(file)
    #    #add the path to a list in for on strings
    #    list_of_files.append(file)
    #all_text = []
    #for item in list_of_files:
    #    x = gv.get_connected_nodes_with_relationships(item, depth=3)
    #    #one entery from x = [{'connected_node': 'rag', 'relationship': 'sourced from', 'sub_connected': []}, {'connected_node': 'rag', 'relationship': 'sourced from', 'sub_connected': []}]
    #    for i in x:
    #        text = f"Node {item} is connect to {i['connected_node']} because of ({i['relationship']})"
    #        all_text.append(text)
    #        for j in i['sub_connected']:
    #            text += f"Node {i['connected_node']} is connect to {j['connected_node']} because of ({j['relationship']})"
    #            all_text.append(text)
    ##save the text to a file
    #with open("data_output/research_papers/connected_nodes.txt", "w", encoding="utf-8") as f:
    #    for item in all_text:
    #        f.write("%s\n" % item)
    
    
   
