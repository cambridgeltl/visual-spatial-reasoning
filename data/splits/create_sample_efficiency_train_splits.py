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

ticks = [0.1, 0.25, 0.5, 0.75]


for tick in ticks:
    train_set = all_instances[:int(tick*len(all_instances))]
    print (len(train_set))
    with open(f"train_{tick}_{len(train_set)}.jsonl", "w") as f:
        for i in train_set:
            row = json.dumps(i)
            f.write(row+"\n")



