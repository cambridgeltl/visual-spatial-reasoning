
import sys
import json
import random

with open(sys.argv[1], "r") as f:
    lines = f.readlines()

all_instances = []
for line in lines:
    instance = json.loads(line)
    all_instances.append(instance)
random.shuffle(all_instances)

# randomly select 70% as train
train_set = all_instances[:int(0.7 * len(all_instances))]
dev_set = all_instances[int(0.7 * len(all_instances)):int(0.8 * len(all_instances))]
test_set = all_instances[int(0.8 * len(all_instances)):]
print (len(train_set), len(dev_set), len(test_set))

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

