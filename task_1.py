#!usr/bin/env python

#imports
import numpy as np
from matplotlib import pyplot
from sklearn import datasets, naive_bayes, metrics
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

def train_and_report(split_data, model_name, smoothing = None):
    #create and train the model
    if smoothing != None:
        model = naive_bayes.MultinomialNB(alpha=smoothing)
    else:
        model = naive_bayes.MultinomialNB()
    model.fit(split_data['training_data'], split_data['training_labels'])

    #now that the model is trained, we need to start writing the performance document
    report = ''

    #(a): model description
    report += '(a)\n{name}\n{under}\n'.format(name=model_name, under = '='*len(model_name))

    #(b): confusion matrix
    report += '(b)\n'

    #(c): precision, recall, and F1-measure of each class
    report += '(c)\n'

    #(d): accuracy_score and f1_score
    report += '(d)\n'

    #(e): prior stability for each class
    report += '(e)\n'

    #(f): size of vocabulary
    report += '(f)\n'

    #(g): number of word-tokens in each class
    report += '(g)\n'

    #(h): number of word-tokens in the corpus
    report += '(h)\n'

    #(i): number and % of words with freq=0 in each class
    report += '(i)\n'

    #(j): number and % of words with freq=0 in the corpus
    report += '(j)\n'

    #(k): 2 favorite words + their log-prob
    report += '(k)\n'

    report += '\n\n'
    return report

    #obsolete - it would be embarrasing if this were to be here in the final version ;)
    #print('model score:\t', model.score(split_data['test_data'], split_data['test_labels']))

def save_report(fn, contents):
    f = open(fn, 'w')
    f.write(contents)
    f.close()

def main():
    data = load_bbc_data()
    plot_distribution(data)
    split_data = preprocess_and_split(data)

    #the model instances to attempt
    attempts = [{'model_name' : 'default MultinomialNB attempt 1'},
                {'model_name' : 'default MultinomialNB attempt 2'},
                {'model_name' : 'MultinomialNB alpha=0.0001', 'smoothing' : 0.0001},
                {'model_name' : 'MultinomialNB alpha=0.9', 'smoothing' : 0.9}]
    report = ''
    for attempt in attempts:
        report += train_and_report(split_data, **attempt)

    save_report('bbc-performance.txt', report)

if __name__ == '__main__':
    main()
