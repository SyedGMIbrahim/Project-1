import json

with open("scratch/audit_synonyms.py", "r") as f:
    content = f.read()

content = content.replace("for col, syns in synonyms_data.items():", "syn_dict = synonyms_data.get('synonyms', synonyms_data)\nfor col, syns in syn_dict.items():")

with open("scratch/audit_synonyms.py", "w") as f:
    f.write(content)
