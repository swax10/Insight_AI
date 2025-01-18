import json
import openai
import re
import time

with open("entity_with_description_responses_with_ec.json") as f:
    data = json.load(f)

MAX_RETRIES = 3

client = openai.OpenAI(
    api_key="d56c239c-5f3e-48a9-9534-5d16d3dcb538",
    base_url="https://api.sambanova.ai/v1",
)
instruction = (
    "Please read all of the discreptiosn and summarize them into one sentence."
)
output_template = """
{
discreption: "string",
}
"""
template = """
{}

# List of descriptions:
{}

# Output format:
{}

# Response:
```json"""

result = {}

for chunk_id, entities in data.items():
    for entity in entities:
        name = entity["name"]
        description = entity["description"]

        if name not in result:
            result[name] = {"description": "", "chunk_ids": set()}

        if len(result[name]["description"]) < len(description):
            result[name]["description"] = description

        if chunk_id not in result[name]["chunk_ids"]:
            result[name]["chunk_ids"].add(chunk_id)

with open("chunkes.json", "r") as f:
    chunks = json.load(f)

print(chunks["d76189f2-3c59-488a-bf2a-d8f2105cd65b"]["metadata"])

for entity, data in result.items():
    references = []
    for chunk_id in data.get("chunk_ids", []):
        chunk_metadata = chunks.get(chunk_id, {}).get("metadata")
        if chunk_metadata and chunk_metadata not in references:
            references.append(chunk_metadata)
    result[entity]["references"] = list(references)
    result[entity].pop("chunk_ids", None)

# number of entities
num_entities = len(result)
print(f"Number of entities: {num_entities}")
# select the all entities starting with 'A'
entities = {k: v for k, v in result.items() if k.startswith("l")}

# number of entities starting with 'A'
num_entities = len(entities)
print(f"Number of entities starting with 'l': {num_entities}")
for entity, data in entities.items():
    print(entity)

with open("entities.json", "w") as f:
    json.dump(result, f)
