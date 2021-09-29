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
    return {'training_data': train_data, 'training_labels':train_labels, 'test_data': test_data, 'test_labels': test_labels, 'all_data': processed_data, 'all_labels': data.target, 'label_names': data.target_names, 'vectorizer': vectorizer}

#creates a feature vector representing all the words in the given class
def get_class_word_vector(label, data, target_class):
    arr = sum(data[label==target_class])
    return arr

def train_and_report(split_data, data, model_name, smoothing = None):
    #create and train the model
    if smoothing != None:
        model = naive_bayes.MultinomialNB(alpha=smoothing)
    else:
        model = naive_bayes.MultinomialNB()
    model.fit(split_data['training_data'], split_data['training_labels'])

    #get prediction for test set
    predicted_labels = list(map(model.predict, split_data['test_data']))

    #count the number of instances of each class
    target_count = [np.count_nonzero(data.target == i) for i in range(len(data.target_names))]

    #now that the model is trained, we need to start writing the performance document
    report = ''

    #(a): model description
    report += '(a)\n{name}\n{under}\n'.format(name=model_name, under = '='*len(model_name))

    #(b): confusion matrix
    cmatrix = metrics.confusion_matrix(split_data['test_labels'], predicted_labels)
    report += '(b)\n{matrix}\n'.format(matrix=cmatrix)

    #(c): precision, recall, and F1-measure of each class
    report += '(c)\n{rep}'.format(rep=metrics.classification_report(split_data['test_labels'], predicted_labels, target_names=split_data['label_names']))


    #(d): accuracy_score and f1_score
    acc_score = metrics.accuracy_score(split_data['test_labels'], predicted_labels)
    m_f1 = metrics.f1_score(split_data['test_labels'], predicted_labels, average='macro')
    w_f1 = metrics.f1_score(split_data['test_labels'], predicted_labels, average='weighted')
    report += '(d)\nAccuracy_score:\t{acc}\nMacro-Average F1:\t{m_f1}\nWeighted-Average F1:\t{w_f1}\n'.format(acc=acc_score, m_f1=m_f1, w_f1=w_f1)

    #(e): prior probability for each class
    report += '(e)\n' + '\n'.join([split_data['label_names'][i] + '\t' + str(target_count[i]/len(split_data['all_labels'])) for i in range(len(split_data['label_names']))]) + '\n'

    #(f): size of vocabulary
    report += '(f)\nVocabulary size:\t{vsize}\n'.format(vsize=len(split_data['vectorizer'].vocabulary_))

    #(g): number of word-tokens in each class
    report += '(g)\n' + '\n'.join([split_data['label_names'][i] + '\t' + str(np.sum(get_class_word_vector(split_data['all_labels'], split_data['all_data'], i))) for i in range(len(split_data['label_names']))]) + '\n'

    #(h): number of word-tokens in the corpus
    report += '(h)\nWords in the Corpus:\t{wc}\n'.format(wc= sum(map(np.sum, split_data['all_data'])))

    #(i): number and % of words with freq=0 in each class
    pretty_count = []
    for i in range(len(split_data['label_names'])):
        class_vec = get_class_word_vector(split_data['all_labels'], split_data['all_data'], i)
        #   using .toarray() because the sparse matrix was not working as expected
        zero_count = np.count_nonzero(class_vec.toarray()==0)
        perc = zero_count / class_vec.toarray().size
        pretty_count += ['{label}:\t{count}\t{perc}%'.format(label=split_data['label_names'][i], count=zero_count, perc=perc)]
    report += '(i)\n' + '\n'.join(pretty_count) + '\n'


    #(j): number and % of words with freq=1 in the corpus
    corpus = sum(split_data['all_data']).toarray()
    one_count = np.count_nonzero(corpus==1)
    perc = one_count / corpus.size
    report += '(j)\n{ones}\t{perc}%\n'.format(ones=one_count, perc=perc)

    #(k): 2 favorite words + their log-prob
    word1 = 'log'
    word2 = 'probability'  #runners up: aaa, 40052308090
    word1_ind = split_data['vectorizer'].vocabulary_[word1]
    word1_probs = " ".join([split_data['label_names'][i] + ":" + str(model.feature_log_prob_[i,word1_ind]) for i in range(len(split_data['label_names']))])
    word2_ind = split_data['vectorizer'].vocabulary_[word2]
    word2_probs = " ".join([split_data['label_names'][i] + ":" + str(model.feature_log_prob_[i,word2_ind]) for i in range(len(split_data['label_names']))])
    report += '(k)\n{w1}:\n{arr1}\n{w2}:\n{arr2}\n'.format(w1=word1, w2=word2, arr1=word1_probs, arr2=word2_probs)

    #end report
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
        report += train_and_report(split_data, data, **attempt)

    save_report('bbc-performance.txt', report)

if __name__ == '__main__':
    main()
