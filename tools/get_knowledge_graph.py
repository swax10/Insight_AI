import sys
import os

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from utils.GraphVisualizer import GraphVisualizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from utils and tools
from utils.df_helpers import documents2Dataframe, df2Graph, graph2Df
from tools.download_research_papers import download_research_papers

class KnowledgeGraphGenerator:
    def __init__(self):
        pass

    def generate_knowledge_graph(self, search_topic):
        try:
            # Create research_papers directory if it doesn't exist
            research_papers_dir = "./research_papers"
            os.makedirs(research_papers_dir, exist_ok=True)

            # Download research papers
            data_dir = download_research_papers(search_topic, research_papers_dir)
            inputdirectory = Path(data_dir)
            out_dir = os.path.basename(data_dir)
            outputdirectory = Path(f"./data_output/{out_dir}")

            # Create output directory if it doesn't exist
            os.makedirs(outputdirectory, exist_ok=True)

            # Load documents
            loader = DirectoryLoader(inputdirectory, glob="**/*.pdf", show_progress=True)
            documents = loader.load()
            if not documents:
                logging.error("No documents were loaded.")
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                length_function=len,
                is_separator_regex=False,
            )

            pages = splitter.split_documents(documents)
            logging.info(f"Number of chunks = {len(pages)}")

            # Create dataframe of chunks
            df = documents2Dataframe(pages)
            if df.empty:
                logging.error("No data available in the dataframe.")
                return None
            logging.info(f"Dataframe shape: {df.shape}")

            # Extract concepts
            concepts_list = df2Graph(df, model='zephyr:latest')
            if not concepts_list:
                logging.error("No concepts extracted from the documents.")
                return None
            dfg1 = graph2Df(concepts_list)

            dfg1.to_csv(outputdirectory/"graph.csv", sep="|", index=False)
            df.to_csv(outputdirectory/"chunks.csv", sep="|", index=False)
            gv = GraphVisualizer(dfg1, df)
            G = gv.run()
            

            return G

        except Exception as e:
            logging.exception(f"An error occurred while generating the knowledge graph: {e}")
            return None

if __name__ == "__main__":
    graph_generator = KnowledgeGraphGenerator()
    topic = input("Enter the research topic: ")
    try:
        knowledge_graph = graph_generator.generate_knowledge_graph(topic)
        if knowledge_graph:
            print(f"Knowledge graph generated successfully for topic: {topic}")
            print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
            print(f"Number of edges: {knowledge_graph.number_of_edges()}")
        else:
            print("Failed to generate knowledge graph.")
    except Exception as e:
        print(f"An error occurred: {e}")