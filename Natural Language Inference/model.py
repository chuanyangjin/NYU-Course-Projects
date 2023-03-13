import json
import collections
import argparse
import random
import numpy as np

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW features of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    dict = {}
    for sentence in [ex["sentence1"], ex["sentence2"]]:
        for word in sentence:
            dict[word] = dict.get(word, 0) + 1
    return dict


def extract_custom_features(ex):
    """Design your own features.
    """
    dict = {}
    for word in ex["sentence2"]:
        if word not in ex["sentence1"]:
            dict[word] = dict.get(word, 0) + 7      # "non-entailment" words
    for word in ex["sentence1"] + ex["sentence2"]:
        dict[word] = dict.get(word, 0) - 1          # other words
    return dict


def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    weights = {}
    for i in range(num_epochs):
        for ex in train_data:
            x = feature_extractor(ex)
            y = ex["gold_label"]
            y_hat = predict(weights, x)
            error = y - y_hat
            for word in x.keys():
                weights[word] = weights.get(word, 0) + learning_rate * error * x[word]

    return weights


def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    word2ind = {}
    for token in tokens:
        if token not in word2ind.keys():
            word2ind[token] = len(word2ind.keys())
    
    co_mat = np.zeros([len(word2ind.keys()), len(word2ind.keys())])

    for i in range(len(tokens)):
        for j in range(i - window_size, i + window_size + 1):
            if (j != i) and (j >= 0) and (j < len(tokens)):
                co_mat[word2ind[tokens[i]]][word2ind[tokens[j]]] += 1

    return word2ind, co_mat


def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    U, S, VH = np.linalg.svd(co_mat, hermitian=True)
    for i in range(min(embed_size, len(S))):
        U[:, i] *= S[i]
    return U[:, :embed_size]


def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    word_embedding = embeddings[word_ind]
    similarities = []
    for embedding in embeddings:
        if metric == 'dot':
            similarities.append(np.dot(word_embedding, embedding))
        else:
            similarities.append(np.dot(word_embedding, embedding) / (np.linalg.norm(word_embedding) * np.linalg.norm(embedding)))
    top_ind = reversed(np.argsort(similarities)[-k-1:])
    topk_words = []
    for ind in top_ind:
        for word in word2ind.keys():
            if (word2ind[word] == ind) and (ind != word_ind):
                topk_words.append(word)
    topk_words = topk_words[:k]
    return topk_words