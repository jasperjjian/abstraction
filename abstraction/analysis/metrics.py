import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json


"""
All metrics should take in two dataset and a checkpoint number, and some optional parameters
They should all return a dataframe with the checkpoint number as the column name and the metric as the value
"""


def get_layer(dataset, layer=7):
    # if array is 1D, expand
    if len(dataset[0].shape) == 2:
        return [np.expand_dims(x[layer], 0) for x in dataset]
    return [x[layer] for x in dataset]

def pca_classifier(dataset1, dataset2, checkpoint_n=None, layers=[7], labels=[], verbose=False, clf="ridge", pca_rank=4):
    results_ridge = []
    for layer in layers:
        list1 = get_layer(dataset1, layer)
        list2 = get_layer(dataset2, layer)
        try:
            combined_data = np.concatenate([np.concatenate(list1), np.concatenate(list2)])
        except ValueError:
            print(checkpoint_n)
            break

        # Perform PCA
        pca = PCA(n_components=pca_rank)
        pca.fit(combined_data)
        # get rid of top 3 components
        transformed_data = pca.transform(combined_data)
        #transformed_data = transformed_data[:, 1:]
        
        # Separate transformed data based on their original lists
        transformed_list1 = transformed_data[:len(list1)*len(list1[0])]
        transformed_list2 = transformed_data[len(list1)*len(list1[0]):len(list1)*len(list1[0])+len(list2)*len(list2[0])]
        distance_between_means = distance.euclidean(np.mean(transformed_list1, axis=0), np.mean(transformed_list2, axis=0))

        
        X = np.concatenate([transformed_list1, transformed_list2])
        y = [0]*len(transformed_list1) + [1]*len(transformed_list2)
        clf = RidgeClassifier().fit(X, y)
        results_ridge.append(clf.score(X, y))
        
        
        # Plot transformed data with separate colors for each list
        if verbose:

            print(f"Layer {layer}")
            plt.scatter(transformed_list1[:, 0], transformed_list1[:, 1], color='blue', label=labels[0], marker='^')
            plt.scatter(transformed_list2[:, 0], transformed_list2[:, 1], color='red', label=labels[1], marker='s')
            # combined datasets and create labels
            plt.scatter(np.mean(transformed_list1, axis=0)[0], np.mean(transformed_list1, axis=0)[1], color='green')
            plt.scatter(np.mean(transformed_list2, axis=0)[0], np.mean(transformed_list2, axis=0)[1], color='orange')

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('Classified PCA')
            plt.legend()

            # Plot decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                np.arange(y_min, y_max, 0.01))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
            plt.colorbar(label='Predicted Value')
            plt.show()

            # Print other results
            print("Distance between mean of PCAed datasets")
            print(distance_between_means)
            print("RidgeClassifier Score")
            print(results_ridge)
        
    results_df = pd.DataFrame({checkpoint_n : results_ridge})
    return results_df

def pca_classifier_train_test(train_datasets, test_datasets, checkpoint_n=None, layers=[7], labels=[], verbose=False, clf="ridge", pca_rank=4):
    results_ridge = []
    for layer in layers:
        train_a = get_layer(train_datasets[0], layer)
        train_b = get_layer(train_datasets[1], layer)
        test_a = get_layer(test_datasets[0], layer)
        test_b = get_layer(test_datasets[1], layer)
    
        try:
            combined_data = np.concatenate([np.concatenate(train_a), np.concatenate(train_b)])
        except ValueError:
            print(checkpoint_n)
            break

        # Perform PCA
        pca = PCA(n_components=pca_rank)
        pca.fit(combined_data)
        transformed_train_data = pca.transform(combined_data)
        
        # Separate transformed data based on their original lists
        transformed_train_data_a = transformed_train_data[:len(train_a)*len(train_a[0])]
        transformed_train_data_b = transformed_train_data[len(train_a)*len(train_a[0]):len(train_b)*len(train_b[0])+len(train_b)*len(train_b[0])]

        
        X = np.concatenate([transformed_train_data_a, transformed_train_data_b])
        y = [0]*len(transformed_train_data_a) + [1]*len(transformed_train_data_b)
        clf = RidgeClassifier().fit(X, y)

        transformed_test_data = pca.transform(np.concatenate([np.concatenate(test_a), np.concatenate(test_b)]))
        transformed_test_data_a = transformed_test_data[:len(test_a)*len(test_a[0])]
        transformed_test_data_b = transformed_test_data[len(test_a)*len(test_a[0]):len(test_b)*len(test_b[0])+len(test_b)*len(test_b[0])]
        X_test = np.concatenate([transformed_test_data_a, transformed_test_data_b])
        y_test = [0]*len(transformed_test_data_a) + [1]*len(transformed_test_data_b)
        results_ridge.append(clf.score(X_test, y_test))
        
    results_df = pd.DataFrame({checkpoint_n : results_ridge})
    return results_df

@ignore_warnings(category=Warning)
def logistic_regression(dataset1, dataset2, checkpoint_n=None, layers=[7], labels=[], verbose=False, clf="ridge", pca_rank=4):
    results_ridge = []
    for layer in layers:
        list1 = get_layer(dataset1, layer)
        list2 = get_layer(dataset2, layer)
        try:
            combined_data = np.concatenate([np.concatenate(list1), np.concatenate(list2)])
        except ValueError:
            print(checkpoint_n)
            break

        
        X = combined_data
        y = [0]*len(list1) + [1]*len(list2)
        fit_intercept=False 
        penalty='l2'
        solver="lbfgs" 
        C=1.0
        lr = LogisticRegression(random_state=0, max_iter=2000, 
                                fit_intercept=fit_intercept, C=C,
                                penalty=penalty, solver=solver)
        pipe = make_pipeline(lr)
        pipe.fit(X, y)
        results_ridge.append(pipe.score(X, y))
        
    results_df = pd.DataFrame({checkpoint_n : results_ridge})
    return results_df

def pca_classifier_per_lemma(dataset1, dataset2, mapping=None, checkpoint_n=None, layers=[7], labels=[], verbose=False, clf="ridge", pca_rank=4):
    results_ridge = {}
    for layer in layers:
        list1 = get_layer(dataset1, layer)
        list2 = get_layer(dataset2, layer)
        try:
            combined_data = np.concatenate([np.concatenate(list1), np.concatenate(list2)])
        except ValueError:
            print(checkpoint_n)
            break

        # Perform PCA
        pca = PCA(n_components=pca_rank)
        pca.fit(combined_data)
        transformed_data = pca.transform(combined_data)
        
        # Separate transformed data based on their original lists
        transformed_list1 = transformed_data[:len(list1)*len(list1[0])]
        transformed_list2 = transformed_data[len(list1)*len(list1[0]):len(list1)*len(list1[0])+len(list2)*len(list2[0])]
        
        X = np.concatenate([transformed_list1, transformed_list2])
        y = [0]*len(transformed_list1) + [1]*len(transformed_list2)
        y = np.array(y)
        clf = RidgeClassifier().fit(X, y)
        predictions = clf.predict(X)
        
        for verb, indices in mapping.items():
            predicted_labels = predictions[indices]
            true_labels = y[indices]
            assert true_labels.all() == 0 or true_labels.all() == 1
            # get the accuracy
            accuracy = np.mean(predicted_labels == true_labels)
            if verb not in results_ridge:
                results_ridge[verb] = []
            results_ridge[verb].append(accuracy)
    results_ridge["checkpoint"] = [checkpoint_n] * len(layers)
    verb_results = pd.DataFrame(results_ridge)
    row_labels = [f"Layer {i+1}" for i in range(len(verb_results))]
    verb_results.set_index(pd.Index(row_labels), inplace=True)

    return verb_results
        
