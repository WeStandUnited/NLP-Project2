import operator
from collections import defaultdict, Counter
from oswegonlp.preprocessing import conll_seq_generator
from oswegonlp.constants import OFFSET, START_TAG, END_TAG, UNK

argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]

def get_tag_word_counts(trainfile):
    """
    Produce a Counter of occurences of word for each tag
    
    Parameters:
    trainfile: -- the filename to be passed as argument to conll_seq_generator
    :returns: -- a default dict of counters, where the keys are tags.
    """
    all_counters = defaultdict(lambda: Counter())


    # Put some code here!

    for i,(word,tag) in enumerate(conll_seq_generator(trainfile)):

        for j,k in zip(word,tag):
           all_counters[k][j] += 1
           #all_counters[j][k] += 1






    return all_counters

def get_tag_to_ix(input_file):
    """
    creates a dictionary that maps each tag (including the START_TAG and END_TAG to a unique index and vice-versa
    :returns: dict1, dict2
    dict1: maps tag to unique index
    dict2: maps each unique index to its own tag
    """
    tag_to_ix={}
    for i,(words,tags) in enumerate(conll_seq_generator(input_file)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    #adding START_TAG and END_TAG
    #if START_TAG not in tag_to_ix:
    #    tag_to_ix[START_TAG] = len(tag_to_ix)
    #if END_TAG not in tag_to_ix:
    #    tag_to_ix[END_TAG] = len(tag_to_ix)
    
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    return tag_to_ix, ix_to_tag


def get_word_to_ix(input_file, max_size=100000):
    """
    creates a vocab that has the list of most frequent occuring words such that the size of the vocab <=max_size, 
    also adds an UNK token to the Vocab and then creates a dictionary that maps each word to a unique index, 
    :returns: vocab, dict
    vocab: list of words in the vocabulary
    dict: maps word to unique index
    """
    vocab_counter=Counter()
    for words,tags in conll_seq_generator(input_file):
        for word,tag in zip(words,tags):
            vocab_counter[word]+=1
    vocab = [ word for word,val in vocab_counter.most_common(max_size-1)]
    vocab.append(UNK)
    
    word_to_ix={}
    ix=0
    for word in vocab:
        word_to_ix[word]=ix
        ix+=1
    
    return vocab, word_to_ix



def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    weights = defaultdict(float)
    weights[('NOUN'),OFFSET] = 1.
    return weights

def get_most_common_word_weights(trainfile):
    """
    Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training file, the weights should be set so that the output tag is Noun.
    
    Parameters:
    trainfile: -- training file
    :returns: -- classification weights
    :rtype: -- defaultdict
This function should return a set weights such that each word should get the tag that is most frequently associated
with it in the training data. If the word does not appear in the training data, the weights should be set so that the
tagger outputs the most common tag in the training data.
For the out of vocabulary words, you need to think on how to set the weights so that you tag them by the most common tag.
    """

    weights = defaultdict(float)
    weights[('NOUN', OFFSET)] = 0.75


    all_counters = get_tag_word_counts(trainfile)# All counters is a default dict of words and there counts

    for tag in all_counters:
        for word in all_counters[tag]:# Specfic counter for a sepcific tag
            
            weights[(sorted([(all_counters[comparison][word], comparison) for comparison in all_counters], reverse = True)[0][1], word)] = 1.

    return weights


def get_tag_trans_counts(input_file):
    """compute a dict of counters for tag transitions
    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict
    """

    tot_counts = defaultdict(lambda: Counter())

    for index, (words, tags) in enumerate(conll_seq_generator(input_file)):
        for index, tag in enumerate(tags):
            if index == 0:
                tot_counts[START_TAG].update([tag])
            if index == len(tags) - 1:
                tot_counts[tag].update([END_TAG])
            else:
                tot_counts[tag].update([tags[index + 1]])

    return dict(tot_counts)






