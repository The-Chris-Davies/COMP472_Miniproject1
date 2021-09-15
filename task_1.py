#!usr/bin/env python

#imports
import numpy as np
from matplotlib import pyplot
from sklearn import datasets, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#loads & returns the dataset
def load_bbc_data():
    return datasets.load_files('data/BBC', encoding='latin1')

#plots the number of instances of each target in a bar graph
def plot_distribution(data):
    target_count = [np.count_nonzero(data.target == i) for i in range(len(data.target_names))]
    graph = pyplot.bar(range(len(data.target_names)), target_count)
    pyplot.title('article instance distribution')
    pyplot.xlabel('article type')
    pyplot.ylabel('number of articles')
    pyplot.bar_label(graph, data.target_names)
    pyplot.savefig('BBC-distribution.pdf')

#runs feature extraction on the corpus and splits it for training and testing
def preprocess_and_split(data):
    vectorizer = CountVectorizer(input=data.data)
    processed_data = vectorizer.fit_transform(data.data)
    train_data, test_data, train_labels, test_labels = train_test_split(processed_data, data.target, test_size=0.2)
    return {'training_data': train_data, 'training_labels':train_labels, 'test_data': test_data, 'test_labels': test_labels}

def train_and_report(split_data, smoothing = None):
    if smoothing != None:
        model = naive_bayes.MultinomialNB(alpha=smoothing)
    else:
        model = naive_bayes.MultinomialNB()

    model.fit(split_data['training_data'], split_data['training_labels'])

    #now that the model is trained, we need to start writing the performance document

    #obsolete - it would be embarrasing if this were to be here in the final version
    print('model score:\t', model.score(split_data['test_data'], split_data['test_labels']))

def main():
    data = load_bbc_data()
    plot_distribution(data)
    split_data = preprocess_and_split(data)

    train_and_report(split_data)

if __name__ == '__main__':
    main()
