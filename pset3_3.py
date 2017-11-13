"""
Created on Sun Nov 12 18:42:08 2017

pset3_3

@author: shenhao
"""

import os
#path = "/Volumes/Transcend/Dropbox (MIT)/2017 Fall/6.867/psets/pset3"
#os.chdir(path)


from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import numpy as np


# part 1

n_features = 1000
n_components = 10
n_top_words = 5


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1, subset = 'all', 
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data
print("done in %0.3fs." % (time() - t0))


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()


# Fit LDA
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (len(data_samples), n_features))
lda = LatentDirichletAllocation(n_topics=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))


# Analyze LDA model results
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
                    
               


# part 2
# choose two categories out of 20 categories, and analyze their topic distribution
# apply the trained model to the two dataset and see the topic distribution
religion_dataset = fetch_20newsgroups(shuffle=True, random_state=1, subset = 'all', 
                             categories = ['soc.religion.christian'], 
                             remove=('headers', 'footers', 'quotes'))
baseball_dataset = fetch_20newsgroups(shuffle=True, random_state=1, subset = 'all', 
                             categories = ['rec.sport.baseball'], 
                             remove=('headers', 'footers', 'quotes'))

tf_religion = tf_vectorizer.fit_transform(religion_dataset.data)
tf_baseball = tf_vectorizer.fit_transform(baseball_dataset.data)

religion_topic_distribution = lda.transform(tf_religion)
baseball_topic_distribution = lda.transform(tf_baseball)

print("The indices of the most likely topics of religion dataset are: ", np.argsort(religion_topic_distribution.mean(0))[n_components - 3:])
print("The indices of the most likely topics of baseball dataset are: ", np.argsort(baseball_topic_distribution.mean(0))[n_components - 3:])








