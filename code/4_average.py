import numpy as np
import csv
import math
from numpy import genfromtxt



num_of_trial = 10



results_of_our_method = np.zeros((num_of_trial + 1, 46, 4)) # Format: result[trial_num][record_label][num_data, time_total, time_quantum, cost]
our_total_time = np.zeros((46, num_of_trial))
our_quantum_time = np.zeros((46, num_of_trial))

for trial_num in range(1, num_of_trial + 1):
    results_of_our_method[trial_num] = genfromtxt('data/trial_' + str(trial_num) +'/results_of_our_method.csv', delimiter=',', skip_header = 0)
    for j in range (1, 46):
        our_total_time[j][trial_num - 1] = results_of_our_method[trial_num][j][1]
        our_quantum_time[j][trial_num - 1] = results_of_our_method[trial_num][j][2]

with open('data/average_of_our_method.csv','w', newline='') as resultsave:
    writer=csv.writer(resultsave)
    writer.writerow(("Number of computed data", "Average time spent in total", "Relative standard deviation (total)", "Average time spent on the quantum circuit", "Relative standard deviation (quantum)"))

for j in range (1, 46):
    mean_total = np.mean(our_total_time[j])
    rsd_total = np.std(our_total_time[j]) / mean_total
    mean_quantum = np.mean(our_quantum_time[j])
    rsd_quantum = np.std(our_quantum_time[j]) / mean_quantum
    with open('data/average_of_our_method.csv','a', newline='') as resultsave:
        writer=csv.writer(resultsave)
        writer.writerow((results_of_our_method[1][j][0], mean_total, rsd_total, mean_quantum, rsd_quantum))



results_of_old_method = np.zeros((num_of_trial + 1, 46, 4)) # Format: result[trial_num][record_label][num_data, time_total, time_quantum, cost]
old_total_time = np.zeros((46, num_of_trial))
old_quantum_time = np.zeros((46, num_of_trial))

for trial_num in range(1, num_of_trial + 1):
    results_of_old_method[trial_num] = genfromtxt('data/trial_' + str(trial_num) +'/results_of_old_method.csv', delimiter=',', skip_header = 0)
    for j in range (1, 46):
        old_total_time[j][trial_num - 1] = results_of_old_method[trial_num][j][1]
        old_quantum_time[j][trial_num - 1] = results_of_old_method[trial_num][j][2]

with open('data/average_of_old_method.csv','w', newline='') as resultsave:
    writer=csv.writer(resultsave)
    writer.writerow(("Number of computed data", "Average time spent in total", "Relative standard deviation (total)", "Average time spent on the quantum circuit", "Relative standard deviation (quantum)"))

for j in range (1, 46):
    mean_total = np.mean(old_total_time[j])
    rsd_total = np.std(old_total_time[j]) / mean_total
    mean_quantum = np.mean(old_quantum_time[j])
    rsd_quantum = np.std(old_quantum_time[j]) / mean_quantum
    with open('data/average_of_old_method.csv','a', newline='') as resultsave:
        writer=csv.writer(resultsave)
        writer.writerow((results_of_old_method[1][j][0], mean_total, rsd_total, mean_quantum, rsd_quantum))



print("The average of time was saved to the file average_of_our_method.csv and average_of_old_method.csv.\n")

print("Comparing the cost functions:")
for trial_num in range(1, num_of_trial + 1):
    mismatch = 0
    for j in range (1, 46):
        if abs((results_of_our_method[trial_num][j][3] - results_of_old_method[trial_num][j][3]) / results_of_old_method[trial_num][j][3]) > 1e-6:
            mismatch += 1
    print("Number of mismatched results in trial", trial_num, ":", mismatch)
