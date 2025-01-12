import json
import openai
import re
import time
from tqdm import tqdm

with open("entity_extraction_responses_with_ec.json", "r") as f:
    entity_list = json.load(f)
with open("chunkes.json", "r") as f:
    chunkes = json.load(f)

client = openai.OpenAI(
    api_key="d56c239c-5f3e-48a9-9534-5d16d3dcb538",
    base_url="https://api.sambanova.ai/v1",
)
entity_relation_extraction_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Entities
{}

### Chunck:
{}

### Response:
```json"""

instruction = """Here’s the improved version of the prompt:

---

Extract all relationships between the following entities from the provided text:

### Guidelines for extraction:  
1. Identify each pair of related entities in the text.  
2. Clearly specify the type of relationship using the provided list.  
3. Provide a concise and accurate description of the relationship.  
4. Ensure the output is formatted as valid JSON.
5. Make description short and clear.

examples Relationships to identify:
- Co-author, Affiliation, Citation, Result, Methodology, Funding, Research Area, Tool/Resource, Publication Venue, 
Conclusion, Hypothesis, Experiment-Outcome, Author-Expertise, Institution-Collaboration, Peer Review, Objective, Theoretical Framework,
Dataset-Origin, Author-Role, Research Funding-Grant, Innovation, Challenge, Comparison, Ethics, Data Collection, Impact, Collaboration-Type,
Publication Date, mathematical model, Application, Limitation, Future Work, Experiment-Design, Experiment-Outcome, etc.

Output format:  
```json
[
  {
    "entity1": "string",
    "entity2": "string",
    "relation": "string",
    "description": "string"
  },
  ...
]
```
"""
with open("chunkes_with_errors_entity_relation_extraction.json", "r") as f:
    chunke_key_with_errors = json.load(f)
# responses = {}
# for error correction:
with open("entity_relation_extraction_responses_with_ec.json", "r") as f:
    responses = json.load(f)
chunkes_with_errors = []
requests = 0
last_time = time.time()
# for key, value in tqdm(entity_list.items()):
for key in tqdm(chunke_key_with_errors):
    value = entity_list[key]
    if key in chunkes:
        try:
            entities = [entity["name"] for entity in value["entities"]]
            entities = ", ".join(set(entities))
            response = client.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                prompt=entity_relation_extraction_prompt.format(
                    instruction, entities, chunkes[key]["text"]
                ),
                temperature=0.0,
                # frequency_penalty=0.1,
                # presence_penalty=0.2,
                # top_p=0.1,
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
                    and len(_json) > 0
                    and "entity1" in _json[0]
                    and "entity2" in _json[0]
                    and "relation" in _json[0]
                    and "description" in _json[0]
                ):
                    responses[key] = _json
                else:
                    print(f"Error in chunk: {key} - {response}")
                    chunkes_with_errors.append(key)
            else:
                print(f"Error in chunk: {key} - {response}")
                chunkes_with_errors.append(key)
        except Exception as e:
            print(e)
            chunkes_with_errors.append(key)
            print(f"Error in chunk: {key} - {response}")
    else:
        print(f"Chunk {key} not found in chunkes")
        chunkes_with_errors.append(key)

with open("entity_relation_extraction_responses_with_ec.json", "w") as f:
    json.dump(responses, f)
with open("chunkes_with_errors_entity_relation_extraction.json", "w") as f:
    json.dump(chunkes_with_errors, f)
