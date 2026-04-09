import json
import os

ipynb_path = r"c:\Users\HP\Desktop\bk work\DeepL\HW4\IDL-HW4\IDL-HW4\HW4P2.ipynb"

with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "ACKNOWLEDGED = False" in source:
            cell['source'] = [s.replace("ACKNOWLEDGED = False", "ACKNOWLEDGED = True") for s in cell['source']]
        if 'root                 : "hw4_data/hw4p2_data"' in source:
            cell['source'] = [s.replace('root                 : "hw4_data/hw4p2_data"', 'root                 : "/local/dataset/hw4p2_data"') for s in cell['source']]
        if 'MODEL = None # TODO: Initialize to your tained model' in source:
            cell['source'] = [s.replace('MODEL = None # TODO: Initialize to your tained model', 'MODEL = model # TODO: Initialize to your tained model') for s in cell['source']]

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
