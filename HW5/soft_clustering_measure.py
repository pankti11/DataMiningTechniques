from __future__ import division
import numpy as np
import itertools, operator
import scipy.stats

# Homogeneity: A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
# Completeness: A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.
# V-measure: The V-measure is the harmonic mean between homogeneity and completeness.
# This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.

# To run this script, take the gmm_outputs(membership) as preds, and the ground truth label(category) of the dataset as labels.
# num_clusters is the number of clusters(K), and num_labels is the number of true categories in the dataset.
# For example:
# preds = np.array([[0.8,0.2], [0.9,0.1], [0.1,0.9], [0.2,0.9]])
# labels = np.array([0, 0, 1, 1])
# print(v_measure(preds, labels, num_clusters, num_labels))


def group_by_label(l):
    it = itertools.groupby(l, operator.itemgetter(1))
    counts = []
    for key, subiter in it:
        counts.append(sum(item[0] for item in subiter))
    return counts

def compute_homogeneity(preds, labels):
    cluster_label_counts = []
    for pred in preds.transpose():
        cluster_label_counts.append(group_by_label([(p,label) for p,label in zip(pred,labels)]))
    
    entropys = []
    for cluster_label_count in cluster_label_counts:
        entropys.append(scipy.stats.entropy(cluster_label_count))
         
    return np.mean(entropys)

def compute_completeness(preds, labels, num_clusters, num_labels):
    label_cluster_counts = {label:np.zeros(num_clusters) for label in range(num_labels)}
    
    for pred, label in zip(preds, labels):
        label_cluster_counts[label] = np.sum([label_cluster_counts[label], pred], axis=0)
    
    entropys = []
    for label_cluster_count in label_cluster_counts.values():
        entropys.append(scipy.stats.entropy(label_cluster_count))
          
    return np.mean(entropys)


def v_measure(preds, labels, num_clusters, num_labels):
    if len(labels) == 0:
        return 1.0, 1.0, 1.0
      
    homogeneity = compute_homogeneity(preds, labels)
    completeness = compute_completeness(preds, labels, num_clusters, num_labels)
    
    if homogeneity==0.0 and completeness==0.0:
        return 0.0, 0.0, 0.0
    v_measure_score = (2.0 * homogeneity * completeness /
                   (homogeneity + completeness))
      
    return homogeneity, completeness, v_measure_score




