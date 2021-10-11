#!usr/bin/env python

#imports
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

#loads and returns the dataset
def load_dataset():
    data = pd.read_csv('data/drug200.csv')
    return data

#plots the number of instances of each target in a bar graph
def plot_distribution(data):
    #get list of drug types from dataset (stored as a list so it is sortable)
    drug_types = list(data.Drug.unique())
    drug_types.sort()
    #count the number of each drug type
    target_count = [np.count_nonzero(data.Drug.values == d) for d in drug_types]
    #plot the data
    graph = pyplot.bar(range(len(drug_types)), target_count)
    pyplot.title('drug instance distribution')
    pyplot.xlabel('drug type')
    pyplot.ylabel('number of articles')
    pyplot.bar_label(graph, drug_types)
    pyplot.savefig('drug-distribution.pdf')

#splits the dataset into training and testing data, and converts all ordinal/nominal features to numeric ones
def preprocess_dataset(data):
    # numericize data
    #list of ordinal values, in order
    ordinals = ['LOW', 'NORMAL', 'HIGH']
    labels = data.pop('Drug')   #pd.get_dummies(data.pop('Drug'))
    data = pd.get_dummies(data,columns=['Sex'])
    data.BP = pd.Categorical(data.BP, categories=ordinals, ordered=True).codes
    data.Cholesterol = pd.Categorical(data.Cholesterol, categories=ordinals, ordered=True).codes

    # split data into training/testing sets
    split_data = {}
    split_data['train_data'], split_data['test_data'], split_data['train_labels'], split_data['test_labels'] = train_test_split(data, labels)
    return split_data

#runs the classifier for the given model
def train_classifier(split_data, model):
    model.fit(split_data['train_data'], split_data['train_labels'])

#returns (accuracy, macro F1, weighted F1)
def get_acc_f1(actual, predicted):
    acc_score = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average='macro')
    weighted_f1 = f1_score(actual, predicted, average='weighted')

    return {'accuracy': acc_score, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}

def generate_report(split_data, model, model_name):
    report = ""

    # compute the predicted labels for the test set
    predicted_labels = model.predict(split_data['test_data'])

    #(a): model description
    report += '\n\n(a)\n{name}\n{under}\n'.format(name=model_name, under = '='*len(model_name))
    # also add model parameters
    if(type(model) == GridSearchCV):
        report += 'params: {params}\n'.format(params=model.best_estimator_.get_params())

    #(b): confusion matrix
    cmatrix = confusion_matrix(split_data['test_labels'], predicted_labels)
    report += '(b)\n{matrix}\n'.format(matrix=cmatrix)

    #(c): precision, recall, and F1-measure for each class
    clreport = classification_report(split_data['test_labels'], predicted_labels)
    report += '(c)\n{rep}\n'.format(rep=clreport)

    #(d): accuracy, macro-avg F1 and weighted-avg F1
    amw = get_acc_f1(split_data['test_labels'], predicted_labels)
    report += '(d)\nAccuracy:\t{acc}\nMacro F1:\t{mac}\nWeighted F1:\t{wei}\n'.format(acc=amw['accuracy'], mac=amw['macro_f1'], wei=amw['weighted_f1'])

    return report

# runs the given model ten times and does some analysis on the results.
# returns a report on the results as a string.
def run_ten_times(split_data, model_type, model_name, params=None):
    accuracies = []
    macro_f1s = []
    weighted_f1s = []

    for i in range(10):
        model = model_type(**params)
        train_classifier(split_data, model)
        predicted_labels = model.predict(split_data['test_data'])
        amw = get_acc_f1(split_data['test_labels'], predicted_labels)
        accuracies.append(amw['accuracy'])
        macro_f1s.append(amw['macro_f1'])
        weighted_f1s.append(amw['weighted_f1'])

    #compute avgs for each stat
    avg_accuracy = np.average(accuracies)
    avg_macro_f1 = np.average(macro_f1s)
    avg_weighted_f1 = np.average(weighted_f1s)

    #compute std deviation for each stat
    std_accuracy = np.std(accuracies)
    std_macro_f1 = np.std(macro_f1s)
    std_weighted_f1 = np.std(weighted_f1s)

    #now, generate report
    report = ''
    full_name = model_name + ", average over 10 runs"
    report += '\n\n(a)\n{name}\n{under}\n\n'.format(name=full_name, under = '='*len(full_name))

    # add average values to report
    report += 'Average metrics:\n\tAccuracy:\t{acc}\n\tMacro F1:\t{mac}\n\tWeighted F1:\t{wei}\n'.format(acc=avg_accuracy, mac=avg_macro_f1, wei=avg_weighted_f1)
    report += 'Std. Dev. for metrics:\n\tAccuracy:\t{acc}\n\tMacro F1:\t{mac}\n\tWeighted F1:\t{wei}\n'.format(acc=std_accuracy, mac=std_macro_f1, wei=std_weighted_f1)

    #done!
    return report

def save_report(fn, contents):
    f = open(fn, 'w')
    f.write(contents)
    f.close()

def main():
    data = load_dataset()
    plot_distribution(data)
    split_data = preprocess_dataset(data)

    # run each of the classifiers
    dtree_params = {'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 100],
                    'min_samples_split': [2,5,10]}
    mlp_params = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
                  'solver': ['adam', 'sgd'],
                  'hidden_layer_sizes': [(50,20), (7,7,7,7)]}

    models = [[GaussianNB, {}, "Gaussian Naive Bayes"],
              [DecisionTreeClassifier, {}, "Default Decision Tree"],
              [GridSearchCV, {'estimator': DecisionTreeClassifier(), 'param_grid': dtree_params}, "Grid Search Decision Tree"],
              [Perceptron, {}, "Default Perceptron"],
              [MLPClassifier, {'hidden_layer_sizes': (100)}, "1x100 Sigmoid Multi-Layer Perceptron"],
              [GridSearchCV, {'estimator': MLPClassifier(), 'param_grid': mlp_params}, "Grid Search Multi-Layer Perceptron"]]

    report = ''

    # run first instance of the model
    for model_type in models:
        model = model_type[0](**model_type[1])
        train_classifier(split_data, model)
        report += generate_report(split_data, model, model_type[2])

    # run each model ten times and add to report
    for model_type in models:
        report += run_ten_times(split_data, model_type[0], model_type[2], model_type[1])

    save_report("drugs-performance.txt", report)


if __name__ == '__main__':
    main()
