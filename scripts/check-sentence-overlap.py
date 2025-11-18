import json
import sys

with open(sys.argv[1], "r") as fin:
    data1 = [json.loads(line) for line in fin]

with open(sys.argv[2], "r") as fin2:
    data2 = [json.loads(line) for line in fin2]

examples1 = [(item["paraphrase"]["sentence1"], item["paraphrase"]["sentence2"]) for item in data1]
examples2 = [(item["paraphrase"]["sentence1"], item["paraphrase"]["sentence2"]) for item in data2]

common = set(examples1).intersection(set(examples2))
common_out = [{"paraphrase": {"sentence1": pair[0], "sentence2": pair[1]}} for pair in common]

with open("common_pairs_in_train_and_negation_test_set.json", "w") as fout:
    for pair in common_out:
        fout.write(json.dumps(pair) + "\n")

print("Common sentence pairs written to 'common_pairs_in_train_and_negation_test_set.txt'.")
