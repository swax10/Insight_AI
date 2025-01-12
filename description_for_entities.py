import json
import re
import time
from tqdm import tqdm
import openai

examples = """[
    {"name": "Apple Inc.", "description": "A multinational technology company."},
    {"name": "Albert Einstein", "description": "A physicist known for the theory of relativity."},
    {"name": "Python (programming language)", "description": "A versatile and readable programming language."},
    {"name": "E=mc^2", "description": "The equation that describes the relationship between mass and energy."}
    {"mame": "llama 3.1", "description": "a llm model"}
]"""
entity_decription_extraction_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Examples
{}

### Chunck:
{}


### Entities
{}
Note: You need to write a short description for all entities mentioned above.

### Response:
```json"""
instruction = """Return List of description all below Entite's you can get inforamtion about Entite from the Chunck below.
- make the discription short and clear
### output format:
[
    {name: "string", description: "string"},
    .,
    .,
]
"""
with open("entity_extraction_responses_with_ec.json", "r") as f:
    entity_list = json.load(f)
with open("chunkes.json", "r") as f:
    chunkes = json.load(f)
client = openai.OpenAI(
    api_key="d56c239c-5f3e-48a9-9534-5d16d3dcb538",
    base_url="https://api.sambanova.ai/v1",
)
responses = {}
chunkes_with_errors = []
requests = 0
last_time = time.time()

# Get all unique entities and count entities by type
for key, value in tqdm(entity_list.items()):
    if key in chunkes:
        try:
            _entities = [entity["name"] for entity in value["entities"]]
            _entities = list(set(_entities))
            entities = ", ".join(_entities)
            response = client.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                prompt=entity_decription_extraction_prompt.format(
                    instruction, examples, chunkes[key]["text"], entities
                ),
                temperature=0.0,
            )
            requests += 1
            if requests % 20 == 0:
                current_time = time.time()
                if current_time - last_time < 60:
                    time.sleep(60 - (current_time - last_time))
                    time.sleep(1)
                last_time = time.time()
            response = "```json" + response.choices[0].text + "```"
            json_regex = re.search(
                r"```json\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL
            )
            if json_regex:
                _json = json.loads(json_regex.group(1))
                if (
                    isinstance(_json, list)
                    and len(_json) > (len(_entities) - 10)
                    and "name" in _json[0]
                    and "description" in _json[0]
                ):
                    responses[key] = _json
                    # for testing
                    # if requests > 10:
                    #    break
                else:
                    print(f"Error in chunk (1): {key}")
                    chunkes_with_errors.append(key)
            else:
                print(f"Error in chunk (2): {key}")
                chunkes_with_errors.append(key)
        except Exception as e:
            print(e)
            chunkes_with_errors.append(key)
            print(f"Error in chunk(3): {key}")
    else:
        print(f"Chunk {key} not found in chunkes")
        chunkes_with_errors.append(key)

with open("entity_with_description_responses_with_ec.json", "w") as f:
    json.dump(responses, f)
with open("chunkes_with_errors_entity_with_description.json", "w") as f:
    json.dump(chunkes_with_errors, f)
