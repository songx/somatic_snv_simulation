import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def simulate_pileup(D, e, Fv, AF, DT):
    #simulate total read depth from a Poisson distribution
    total_read_depth = np.random.poisson(D)
    
    #simulate sequencing noise from a Binomial distribution
    noise_depth = np.random.binomial(total_read_depth, e)
    
    #simulate the presence of a somatic mutation
    has_somatic_mutation = np.random.rand() < Fv
    
    #if there is a somatic mutation, simulate ALT reads from a Binomial distribution
    if has_somatic_mutation:
        somatic_depth = np.random.binomial(total_read_depth, AF)
    else:
        somatic_depth = 0
    
    #calculate the total ALT reads
    alt_reads = noise_depth + somatic_depth
    
    #determine if the pileup is a Positive or Negative based on DT
    is_positive = alt_reads >= DT
    
    return is_positive, has_somatic_mutation

def simulate_and_evaluate(k, D, e, Fv, AF, DT):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(k):
        is_positive, has_somatic_mutation = simulate_pileup(D, e, Fv, AF, DT)
        
        if is_positive:
            if has_somatic_mutation:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if has_somatic_mutation:
                false_negatives += 1
            else:
                true_negatives += 1
    
    #calculate PPA (Positive Predictive Accuracy)
    ppa = true_positives / (true_positives + false_negatives)
    
    #calculate PPV (Positive Predictive Value)
    ppv = true_positives / (true_positives + false_positives)
    
    #calculate Specificity
    specificity = true_negatives / (true_negatives + false_positives)

    #create confusion matrix
    confusion_matrix = np.array([[true_negatives, false_positives],
                                 [false_negatives, true_positives]])

    return ppa, ppv, specificity, confusion_matrix

fig, axs = plt.subplots(nrows=2)
#1)
params1 = [10000, 100, 0.005, 0.01, 0.05, 2]
ppa1, ppv1, specificity1, confusion_matrix1 = simulate_and_evaluate(*params1)
print(f"1) PPA: {ppa1}, PPV: {ppv1}, Specificity: {specificity1}")

sn.heatmap(confusion_matrix1,annot=True, ax=axs[0], cmap='Blues')
axs[0].set_title('Confusion Matrix 1')

#2)
params2 = [10000, 300, 0.005, 0.01, 0.05, 2]
ppa2, ppv2, specificity2, confusion_matrix2 = simulate_and_evaluate(*params2)
print(f"2) PPA: {ppa2}, PPV: {ppv2}, Specificity: {specificity2}")

sn.heatmap(confusion_matrix2,annot=True, ax=axs[1], cmap='Blues')
axs[1].set_title('Confusion Matrix 2')
plt.show()

#3)
print("3)")
AF_values = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
ppa_values = []
ppv_values = []

for AF in AF_values:
    ppa, ppv, specificity, confusion_matrix = simulate_and_evaluate(10000, 100, 0.005, 0.01, AF, 2)
    ppa_values.append(ppa)
    ppv_values.append(ppv)
    print(f"AF: {AF}, PPA: {ppa}, PPV: {ppv}")

plt.figure(figsize=(10, 5))
plt.plot(AF_values, ppa_values, label='PPA')
plt.plot(AF_values, ppv_values, label='PPV')
plt.xlabel('Somatic AF')
plt.title('PPA and PPV as a function of somatic AF')
plt.legend()
plt.show()



