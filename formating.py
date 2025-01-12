import json
from tqdm import tqdm

with open("chunkes.json") as f:
    data = json.load(f)

new_data = {}
for i in tqdm(data):
    id = i["id"]
    # add all the data to the new_data except the "id" key
    new_data[id] = {k: v for k, v in i.items() if k != "id"}

with open("chunkes.json", "w") as f:
    json.dump(new_data, f)
