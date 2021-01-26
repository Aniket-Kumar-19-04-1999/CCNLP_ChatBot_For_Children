import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    takes a sentence and returns array of words.
    "Hi how are you ?" => ["Hi","how","are","you","?"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    This function provides the root form of the word.
    ["organize", "organizes", "organizing"] => ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    returns bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    tokenized_sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag of words = [0, 1, 0, 1, 0, 0, 0]
    """
    # words is stemmed while tokenized_sentence is not
    sentence_words_stemmed = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    # words is array of all words in corpus. bag of words is vector
    # representing which words our sentence has.
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words_stemmed: 
            bag[idx] = 1

    return bag