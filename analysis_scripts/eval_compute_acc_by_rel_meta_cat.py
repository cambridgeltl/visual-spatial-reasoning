
import sys
import json

"""
sys.argv[1]: test.jsonl path
sys.argsv[2]: preds.txt path
sys.argsv[3]: meta category dict path
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

rel2cat = {}
cats = []
rel_count = 0
with open(sys.argv[3], "r") as f:
    lines = f.readlines()
    for line in lines:
        cat, rels = line.strip().split(": ")
        cat = cat.strip()
        rel_list = rels.split(",")
        rel_count+=len(rel_list)
        rel_list = [rel.strip() for rel in rel_list]
        for rel in rel_list:
            rel2cat[rel] = cat
        cats.append(cat)
print (f"# rel: {rel_count}")
cats = list(set(cats))

cat_dict, cat_acc = {}, {}
for cat in cats:
    cat_dict[cat] = []
for i, instance in enumerate(data_json):
    correct = 1 if preds[i] == instance["label"] else 0
    if instance["relation"] in rel2cat.keys():
        cat_dict[rel2cat[instance["relation"]]].append(correct)
    else:
        print ("no availible meta cat:")
        print (instance)

for k,v in cat_dict.items():
    cat_acc[k] = (sum(v)/len(v), len(v))

cat_acc_sorted = {k: v for k, v in sorted(cat_acc.items(), key=lambda item: item[1][0], reverse=True)}

#for k,v in cat_acc.items():
for k,v in cat_acc_sorted.items():
    print (f"{k}\t{v[0]:.4f}\t{v[1]}")
