import os
import json
import sys

data_json = []
with open(sys.argv[1], "r") as f:
    lines = f.readlines()
    for line in lines:
        j_line = json.loads(line)
        data_json.append(j_line)

imgs = []
for instance in data_json:
    imgs.append(instance["image"])
print (f"total data points: {len(imgs)}")
imgs = list(set(imgs))
print (f"total images: {len(imgs)}")

for img in imgs:
    os.system(f"cp {os.path.join(sys.argv[2], img)} {sys.argv[3]}")

print (f"{len(os.listdir(sys.argv[3]))} images moved to {sys.argv[3]}")

