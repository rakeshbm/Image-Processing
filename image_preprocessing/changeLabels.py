#########################################################
# This script converts label that are paired tuples in the JSON file to a single entry that can be used for trainning
# and generates a JSON file that summarizes the labellings in each file.
# The script modifies the labels in the original JSON files by performing the following changes:
# a.	Removes the negative labels
# b.	Retains the labels only if the label is one of sign, door, window, traffic light, pole
# c.    Extract labeling information and save them in a file named “extraction.json” under the same directory
# Authors: Boyang Lu
# Version: Python 3.6
#########################################################

import json
import ntpath
import sys
import glob

validLabels = {"sign", "door", "window", "traffic light", "pole"}

input_path = sys.argv[1]
files = glob.glob(input_path + "/*.json")
extract_dict = dict()

# get file name
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

for file in files:
    with open(file) as f:
        data = json.load(f)

    shapes = data["shapes"]
    updatedShapes = []
    positiveObjects = {}

    for shape in shapes:
        label = shape["label"].replace(" ", "").split(",")

        if label[0] in validLabels and label[1] == "positive":
            shape["label"] = label[0]
            updatedShapes.append(shape)
            if label[0] in positiveObjects.keys():
                positiveObjects[label[0]] += 1
            else:
                positiveObjects[label[0]] = 1

    data["shapes"] = updatedShapes
    extract_dict[path_leaf(file)] = positiveObjects

    # write back to file
    output = json.dumps(data)
    with open(file, "w") as f:
        f.write(output)

# write extraction file
extraction = json.dumps(extract_dict)

with open(input_path + "/extraction.json", "w") as f:
    f.write(extraction)


