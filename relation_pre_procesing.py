import json
from collections import defaultdict

with open("entity_relation_extraction_responses_with_ec.json") as f:
    data = json.load(f)

# Process data
relation_map = defaultdict(
    lambda: {"description": "", "ids": [], "entity1": "", "entity2": "", "relation": ""}
)

for doc_id, relationships in data.items():
    for relation in relationships:
        key = (relation["entity1"], relation["entity2"], relation["relation"])
        current_length = len(relation_map[key]["description"])
        new_length = len(relation["description"])

        if new_length > current_length:
            relation_map[key].update(
                {
                    "entity1": relation["entity1"],
                    "entity2": relation["entity2"],
                    "relation": relation["relation"],
                    "description": relation["description"],
                }
            )
        relation_map[key]["ids"].append(doc_id)

# Prepare output
output = list(relation_map.values())

# print len of output
print(f"Number of relations: {len(output)}")

with open("relations.json", "w") as f:
    json.dump(output, f, indent=4)
