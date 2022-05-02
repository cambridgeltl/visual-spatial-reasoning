
import sys
import json
import random
import pandas as pd

with open(sys.argv[1], "r") as f:
    lines = f.readlines()

all_instances = []
for line in lines:
    instance = json.loads(line)
    all_instances.append(instance)
random.shuffle(all_instances)

df = pd.read_json(sys.argv[1], lines=True)

all_concepts = list(set(df["subj"].tolist()+df["obj"].tolist()))
random.shuffle(all_concepts)


train_set_concepts = all_concepts[:int(0.5 * len(all_concepts))]
dev_set_concepts = all_concepts[int(0.5 * len(all_concepts)):int(0.7 * len(all_concepts))]
test_set_concepts = all_concepts[int(0.7 * len(all_concepts)):]
print (len(train_set_concepts), len(dev_set_concepts), len(test_set_concepts))

train_set, dev_set, test_set = [], [], []
for instance in all_instances:
    if instance["subj"] in train_set_concepts and instance["obj"] in train_set_concepts:
        train_set.append(instance)
    if instance["subj"] in dev_set_concepts and instance["obj"] in dev_set_concepts:
        dev_set.append(instance)
    if instance["subj"] in test_set_concepts and instance["obj"] in test_set_concepts:
        test_set.append(instance)

print (len(train_set), len(dev_set), len(test_set), len(train_set+dev_set+test_set))

with open("train.jsonl", "w") as f:
    for i in train_set:
        row = json.dumps(i)
        f.write(row+"\n")

with open("dev.jsonl", "w") as f:
    for i in dev_set:
        row = json.dumps(i)
        f.write(row+"\n")

with open("test.jsonl", "w") as f:
    for i in test_set:
        row = json.dumps(i)
        f.write(row+"\n")

