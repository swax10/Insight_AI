import json


def make_lowercase(obj):
    if isinstance(obj, dict):
        return {k: make_lowercase(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_lowercase(elem) for elem in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj


with open("entity_relation_extraction_responses_with_ec.json.json", "r") as f:
    data = json.load(f)

data = make_lowercase(data)

with open("entity_extraction_responses_with_ec.json", "w") as f:
    json.dump(data, f)
