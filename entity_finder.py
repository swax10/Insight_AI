import json
import openai
import re
import time
from tqdm import tqdm

client = openai.OpenAI(
    api_key="d56c239c-5f3e-48a9-9534-5d16d3dcb538",
    base_url="https://api.sambanova.ai/v1",
)

with open("chunkes.json", "r") as json_file:
    chunkes = json.load(json_file)

entity_extraction_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Data:
{}

### Response:
```json
"""
instruction = """Return list Entities from this following input the entities can be of types like
name, place, animal, thing, concept, topic, person, paper, model, method, figure, equation, Variable, etc,
while creating an entity list, please make sure to include the type of entity as well.
in case of equations, include the complete equation into one entity.

json output format:
[
  {
    "name": "string",
    "type": "string"
  },
  .,
  .,
]
"""
with open("entity_extraction_responses_without_ec.json", "r") as json_file:
    responses = json.load(json_file)

with open("entity_extraction_responses_chunkes_with_errors.json", "r") as json_file:
    chunkes_with_errors = json.load(json_file)
    chunkes_with_errors = list(chunkes_with_errors)
# rate limit of 10 per minute
requests = 0
last_time = time.time()
for chunk in tqdm(chunkes):
    # to retry failed chunks
    if chunk["id"] in chunkes_with_errors:
        try:
            response = client.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                prompt=entity_extraction_prompt.format(instruction, chunk["text"]),
                temperature=0.0,
                top_p=0.1,
            )
            requests += 1
            if requests % 15 == 0:
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
                    and "name" in _json[0]
                    and "type" in _json[0]
                ):
                    responses.append({"id": chunk["id"], "entities": _json})
                else:
                    print(f"Error in chunk: {chunk['id']} - {response}")
                    chunkes_with_errors.append(chunk["id"])
            else:
                print(f"Error in chunk: {chunk['id']} - {response}")
                chunkes_with_errors.append(chunk["id"])
        except Exception as e:
            print(e)
            chunkes_with_errors.append(chunk["id"])
            print(f"Error in chunk: {chunk['id']} - {response}")

# save to json file
with open("entity_extraction_responses_with_ec.json", "w") as json_file:
    json.dump(responses, json_file)

# with open("chunkes_with_errors.json", "w") as json_file:
#    json.dump(chunkes_with_errors, json_file)

print(len(responses[0]["entities"]))
