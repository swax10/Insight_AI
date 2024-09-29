import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import random
import matplotlib.pyplot as plt
import community as community_louvain
import plotly.graph_objects as go
import pickle  # Import pickle for serialization

class GraphVisualization:
    def __init__(self, df1, df2):
        self.dfg1 = pd.merge(df1, df2[['chunk_id', 'source']], on='chunk_id', how='left')
        self.G = nx.Graph()
        self.graph_output_directory = "index.html"

    def clean_data(self):
        self.dfg1.replace("", np.nan, inplace=True)
        self.dfg1.dropna(subset=["node_1", "node_2"], inplace=True)

    def contextual_proximity(self, df):
        dfg_long = pd.melt(df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node")
        dfg_long.drop(columns=["variable"], inplace=True)
        
        dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
        
        dfg2 = dfg_wide[dfg_wide["node_1"] != dfg_wide["node_2"]].reset_index(drop=True)
        
        dfg2 = (
            dfg2.groupby(["node_1", "node_2"])
            .agg({"chunk_id": [",".join, "count"]})
            .reset_index()
        )
        dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
        dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
        
        dfg2 = dfg2[dfg2["count"] != 1]
        dfg2["edge"] = "contextual proximity"
        
        return dfg2
    
    def process_graph(self):
        dfg2 = self.contextual_proximity(self.dfg1)
        dfg = pd.concat([self.dfg1, dfg2], axis=0)

        dfg['chunk_id'] = dfg['chunk_id'].astype(str)
        dfg['edge'] = dfg['edge'].astype(str)

        dfg = (
            dfg.groupby(["node_1", "node_2"])
            .agg({"chunk_id": ",".join, "edge": ','.join})
            .reset_index()
        )

        nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()

        for node in nodes:
            self.G.add_node(str(node))

        for index, row in dfg.iterrows():
            self.G.add_edge(
                str(row["node_1"]),
                str(row["node_2"]),
                title=row["edge"],
                weight=1
            )

        community_partition = self.detect_communities()
        for community in community_partition:
            most_connected_node = max(community, key=lambda node: self.G.degree[node])
            source_rows = self.dfg1[self.dfg1['node_1'] == most_connected_node]
            if not source_rows.empty:
                source_node = str(source_rows['source'].values[0])
                self.G.add_node(source_node)
                self.G.add_edge(most_connected_node, source_node, title="sourced from", weight=1)

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
            self.G.nodes[row['node']]['group'] = row['group']
            self.G.nodes[row['node']]['color'] = row['color']
            self.G.nodes[row['node']]['size'] = self.G.degree[row['node']]
    
    def save_graph(self, filename='graph.pkl'):
        """Save the graph to a file using Pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"Graph saved to {filename}")

    def load_graph(self, filename='graph.pkl'):
        """Load the graph from a file using Pickle."""
        with open(filename, 'rb') as f:
            self.G = pickle.load(f)
        print(f"Graph loaded from {filename}")

    def visualize_graph(self):
        pos = nx.spring_layout(self.G, k=0.1)

        node_x = [pos[node][0] for node in self.G.nodes()]
        node_y = [pos[node][1] for node in self.G.nodes()]
        colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
        sizes = [max(10, self.G.degree[node] * 10) for node in self.G.nodes()]

        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='black'),
            hoverinfo='none',
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=sizes, color=colors, line=dict(width=1)),
            text=[str(node) for node in self.G.nodes()],
            textposition="top center",
            hoverinfo='text'
        ))

        fig.update_layout(
            title='Interactive Graph Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        fig.write_html("interactive_graph.html")
        fig.show()
        
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
    df1 = pd.read_csv("data_output/research_papers/graph.csv", sep="|")  # Adjust this line according to your data source
    df2 = pd.read_csv("data_output/research_papers/chunks.csv", sep="|")  # Adjust this line according to your data source
    gv = GraphVisualization(df1, df2)
    gv.run()
