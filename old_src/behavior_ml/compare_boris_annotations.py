## Compare BORIS annotations
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

rate = 1/30
max_time = 360
n_bins = int(max_time//rate)

#Load in BORIS results
boris_brett = ['./data/boris/DLC1.csv',
                './data/boris/DLC2.csv',
                './data/boris/DLC3.csv',
                './data/boris/DLC4.csv',
                './data/boris/DLC5.csv']

boris_donnie = ['./data/boris/DLC1_donnie.csv',
                './data/boris/DLC2_donnie.csv',
                './data/boris/DLC3_donnie.csv',
                './data/boris/DLC4_donnie.csv',
                './data/boris/DLC5_donnie.csv']

#mabe_labels = np.load(behavior_results_in, allow_pickle=True).item()

def format_labels(boris_in):
    boris_labels = pd.read_csv(boris_in, skiprows = 15)
    boris_labels['index'] = (boris_labels.index//2)
    boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
    boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
    boris_labels = [list(i) for i in boris_labels]
    ground_truth = np.zeros(n_bins)
    for start, end in boris_labels:
        ground_truth[int(start/rate):int(end/rate)] = 1
    return ground_truth

accuracies = []

N = len(boris_brett)

for fn_brett, fn_donnie in zip(boris_brett, boris_donnie):

    brett_labels = format_labels(fn_brett)
    donnie_labels = format_labels(fn_donnie)
    acc = accuracy_score(brett_labels, donnie_labels)
    print('File:', fn_brett)
    print('Precision:', precision_score(brett_labels, donnie_labels))
    print('Recall:', recall_score(brett_labels, donnie_labels))
    print('F1 score:', f1_score(brett_labels, donnie_labels))
    print('Accuracy score:', acc)
    accuracies.append(acc)

accuracies = np.array(accuracies)
print('Mean accuracy:', np.mean(accuracies))
print('SE accuracy:', np.std(accuracies)/np.sqrt(N))