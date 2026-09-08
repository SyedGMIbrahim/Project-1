import re

with open("scratch/audit_results_backend.txt", "r") as f:
    lines = f.readlines()

output = "| Synonym | Assigned Feature (Score) | Closer To Feature (Score) | Delta |\n"
output += "|---|---|---|---|\n"

synonym = ""
assigned = ""
closer = ""
delta = ""

for line in lines:
    line = line.strip()
    if line.startswith("Synonym:"):
        synonym = re.search(r"Synonym: '(.*)'", line).group(1)
    elif line.startswith("Assigned:"):
        match = re.search(r"Assigned: \[(.*)\] \(Score: (.*)\)", line)
        assigned = f"{match.group(1)} ({match.group(2)})"
    elif line.startswith("Closer to:"):
        match = re.search(r"Closer to: \[(.*)\] \(Score: (.*)\)", line)
        closer = f"{match.group(1)} ({match.group(2)})"
    elif line.startswith("Delta:"):
        delta = line.replace("Delta: ", "")
        output += f"| {synonym} | {assigned} | {closer} | {delta} |\n"

with open("scratch/synonym_audit_table.md", "w") as f:
    f.write(output)
