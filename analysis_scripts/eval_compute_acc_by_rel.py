
import sys
import json

"""
sys.argv[1]: test.jsonl path
sys.argsv[2]: preds.txt path
"""

# read test.jsonl
data_json = []
with open(sys.argv[1], "r") as f:
    lines = f.readlines()
    for line in lines:
        j_line = json.loads(line)
        data_json.append(j_line)

preds = []
with open(sys.argv[2], "r") as f:
    lines = f.readlines()
    for line in lines:
        preds.append(int(line.strip()))

rel_dict = {}
rel_acc = {}
for i, instance in enumerate(data_json):
    correct = 1 if preds[i] == instance["label"] else 0
    #if correct == 0:
    #    print (f"error: {i}")
    if instance["relation"] in rel_dict.keys():
        rel_dict[instance["relation"]].append(correct)
    else:
        rel_dict[instance["relation"]] = [correct]

for k,v in rel_dict.items():
    rel_acc[k] = (sum(v)/len(v), len(v))
rel_acc_sorted = {k: v for k, v in sorted(rel_acc.items(), key=lambda item: item[1][1], reverse=True)}

for k,v in rel_acc_sorted.items():
    print (f"{k}\t{v[0]:.4f}\t{v[1]}")
