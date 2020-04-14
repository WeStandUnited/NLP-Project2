from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation, preprocessing

import numpy as np
import itertools
import functools
from collections import defaultdict
from collections import Counter

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preprocessing.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


def corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """

    corpus_counts = defaultdict(float)
    for pos, curr_label in enumerate(y):
        if curr_label == label:
            for word in x[pos]:
                corpus_counts[word] += x[pos][word]
    return corpus_counts


# deliverable 3.2
def estimate_pxy(x,y,label,alpha,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param alpha: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    c = corpus_counts(x,y,label)

    de = defaultdict(float)

    denom = (len(vocab) * alpha) + sum(c.values())



    for word in vocab:

        if word not in c:
            c[word] = 0

    for word2 in vocab:
        de[word2] = np.log((c[word2]+alpha)/denom)




    return de

def estimate_nb(x,y,alpha):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    (eg. [Counter({'donald': 1,
         'trump': 1,
         'hillary': 1,
         'clinton': 1,
         'final': 1,
         'debate': 1})...]

    :param y: list of labels
    (eg. [real real real real real...]
    :param alpha: smoothing constant

    :returns: weights
    ("real","obama"): .5 ,
    :rtype: defaultdict
    """


    #P(word|label) = (P(word|label) * P(word)) / P(word)


    labels = set(y)
    wordcount = list()
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    de = defaultdict(float)
    vocab = set()

    for i in x:
        for j in i.keys():
            vocab.add(j)

    for l in y:
        doc_counts[l] +=1# count of documents and such



    for label in labels:
        de[(label, OFFSET)] = np.log(doc_counts[label] / sum(doc_counts.values()))
        prob =  estimate_pxy(x,y,label,alpha,vocab)

        for i in prob:
            de[(label,i)] = prob[i]



    return de

    


def find_best_smoother(x_tr,y_tr,x_dv,y_dv,alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    raise NotImplementedError
    







