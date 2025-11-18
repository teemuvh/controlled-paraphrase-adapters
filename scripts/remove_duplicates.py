import json
import sys

with open(sys.argv[1], "r") as file1, open(sys.argv[2], "r") as file2:
    data1 = [json.loads(line) for line in file1]
    data2 = [json.loads(line) for line in file2]

ex_pairs1 = {(item["paraphrase"]["sentence1"], item["paraphrase"]["sentence2"]) for item in data1}
ex_pairs2 = {(item["paraphrase"]["sentence1"], item["paraphrase"]["sentence2"]) for item in data2}

unique_pairs = ex_pairs1 - ex_pairs2

unique_data = [item for item in data1 if (item["paraphrase"]["sentence1"], item["paraphrase"]["sentence2"]) in unique_pairs]

with open("negation_test_data_uniques.json", "w") as fout:
    for item in unique_data:
        fout.write(json.dumps(item) + "\n")

print("Unique negation test data written to 'negation_test_data_uniques.json'.")
