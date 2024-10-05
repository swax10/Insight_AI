import requests
import json  # Import json module for JSON handling


def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"  # Adjust to match your Ollama endpoint
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama3.2",  # Assuming you are using Ollama Llama3
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()  # This should be a dictionary if the response is correct


def create_prompt_for_duplicates(edges):
    prompt = (
        "You are an intelligent assistant responsible for processing a list of edges in a graph. Each edge represents a connection between two nodes and describes a specific relationship between them. Your task is to identify and remove duplicates or similar edges based on their contextual meaning.\n\n"
        "When evaluating edges for similarity, consider the following criteria:\n"
        "- **Node Similarity**: Two nodes may be similar if they represent the same entity or concept, even if their names are slightly different (e.g., 'Santa Clara' and 'Santa Clara, California').\n"
        "- **Relation Similarity**: Edges can be considered duplicates if their relations convey similar meanings, such as 'Location of the company' and 'Company Headquarters'.\n"
        "- **Case Sensitivity**: Treat variations in capitalization as equivalent (e.g., 'nvidia' and 'NVIDIA' are the same).\n"
        "- **Synonyms**: Be aware of synonyms or commonly used phrases that convey the same idea (e.g., 'Location of Organization' could be synonymous with 'Company Location').\n\n"
        "Your goal is to ensure that each edge in your final output is unique based on these criteria.\n\n"
        "Here is the list of edges you need to process:\n"
        "try to remove as many duplicates as possible if there relationes are not Unique Edges\n\n"
    )

    for edge in edges:
        prompt += f"- Node1: {edge['node1']}, Node2: {edge['node2']}, Relation: {edge['relation']}\n"

    prompt += (
        "\nPlease return a JSON object with the unique edges in the following format:\n"
    )
    prompt += '{"edges": [{"node1": "unique_node_name", "node2": "unique_node_name", "relation": "unique_relation_name"}, ...]}'

    return prompt


def remove_duplicate_edges_with_ollama(edges_json):
    edges_data = json.loads(edges_json)  # Load JSON input to Python object
    edges = edges_data["edges"]  # Extract edges from input data

    prompt = create_prompt_for_duplicates(edges)
    ollama_response = query_ollama(prompt)

    # Check if ollama_response is a dictionary
    if isinstance(ollama_response, dict):
        # Get the 'response' field and parse it as JSON
        response_str = ollama_response.get("response", "{}")
        try:
            unique_edges = json.loads(response_str).get("edges", [])
        except json.JSONDecodeError:
            print("Failed to decode response string:", response_str)
            return json.dumps({"edges": []})  # Return empty if decoding fails
    else:
        print("Unexpected response format:", ollama_response)
        return json.dumps({"edges": []})  # Return empty if format is unexpected

    result_edges = []
    for edge in unique_edges:
        node1 = edge.get("node1")  # Extract node1
        node2 = edge.get("node2")  # Extract node2
        relation = edge.get("relation")  # Get the relation
        if node1 and node2 and relation:
            result_edges.append({"node1": node1, "node2": node2, "relation": relation})

    return json.dumps({"edges": result_edges})  # Return the result as a JSON string


if __name__ == "__main__":
    # Sample data in JSON format
    edges_json = json.dumps(
        {
            "nodes": [
                "nvidia",
                "santa clara",
                "santa clara, california",
                "apple",
                "cupertino",
                "cupertino, california",
                "microsoft",
                "redmond",
                "redmond, washington",
                "google" "elon musk",
                "tesla",
                "mars",
                "spacex",
                "jeff bezos",
                "blue origin",
                "amazon",
                "andy jassy",
                "harvard university",
                "bill gates",
                "microsoft",
                "philanthropy",
            ],
            "edges": [
                {
                    "node1": "nvidia",
                    "node2": "santa clara",
                    "relation": "Location of the company",
                },
                {
                    "node1": "santa clara",
                    "node2": "santa clara, california",
                    "relation": "City of headquarters",
                },
                {
                    "node1": "santa clara, california",
                    "node2": "nvidia",
                    "relation": "Location of Organization",
                },
                {
                    "node1": "apple",
                    "node2": "cupertino",
                    "relation": "Location of the company",
                },
                {
                    "node1": "cupertino",
                    "node2": "cupertino, california",
                    "relation": "City of headquarters",
                },
                {
                    "node1": "cupertino, california",
                    "node2": "apple",
                    "relation": "Location of Organization",
                },
                {
                    "node1": "microsoft",
                    "node2": "redmond",
                    "relation": "Location of the company",
                },
                {
                    "node1": "redmond",
                    "node2": "redmond, washington",
                    "relation": "City of headquarters",
                },
                {
                    "node1": "redmond, washington",
                    "node2": "microsoft",
                    "relation": "Location of Organization",
                },
                {
                    "node1": "google",
                    "node2": "mountain view",
                    "relation": "Location of the company",
                },
                {"node1": "elon musk", "node2": "tesla", "relation": "Founder of"},
                {"node1": "elon musk", "node2": "spacex", "relation": "CEO of"},
                {
                    "node1": "elon musk",
                    "node2": "mars",
                    "relation": "Ambition to colonize",
                },
                {
                    "node1": "jeff bezos",
                    "node2": "blue origin",
                    "relation": "Founder of",
                },
                {"node1": "jeff bezos", "node2": "amazon", "relation": "Former CEO of"},
                {
                    "node1": "andy jassy",
                    "node2": "amazon",
                    "relation": "Current CEO of",
                },
                {
                    "node1": "bill gates",
                    "node2": "microsoft",
                    "relation": "Co-founder of",
                },
                {
                    "node1": "bill gates",
                    "node2": "philanthropy",
                    "relation": "Engaged in",
                },
                {
                    "node1": "andy jassy",
                    "node2": "harvard university",
                    "relation": "Alumni of",
                },
                {
                    "node1": "jeff bezos",
                    "node2": "philanthropy",
                    "relation": "Engaged in",
                },
            ],
        }
    )

    # Call the function
    unique_edges_json = remove_duplicate_edges_with_ollama(edges_json)

    # Display the results
    print(unique_edges_json)  # Print the JSON output
    # number of nodes in the graph
    print(
        len(json.loads(unique_edges_json)["edges"])
    )  # Print the number of unique edges
