import pandas as pd
import os

print("Implementing Find-S algorithm...")
def find_s_algorithm(csv_file):
    dataset = pd.read_csv(csv_file)
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, -1].values
    hypothesis = None
    for i in range(len(labels)):
        if labels[i].lower() == 'yes':
            if hypothesis is None:
                hypothesis = list(attributes[i])
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != attributes[i][j]:
                        hypothesis[j] = '?'
    return hypothesis
csv_file = os.path.expanduser("~/Downloads/data.csv")
final_hypothesis = find_s_algorithm(csv_file)
print("Final Hypothesis:", final_hypothesis)