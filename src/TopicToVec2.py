from collections import defaultdict
from random import choice, random, randint
#from math import log
from sys import argv
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import operator
import math
import os
import sys
import time
import plotly.plotly as py
from plotly.graph_objs import *

# sets the valid words (words which appear in corpus and have word vecs)
# Vetted
def setValidWords():

	global wordPadding
	global wordVecs
	global words

	print "* loading valid words and their vectors"
	badWords = [] #['jacqueline', 'landis']

	# reads words
	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			for token in tokens[2:]:
				if token not in words and token not in badWords:
					words.add(token)
	f.close()

	print "\tloaded " + str(len(words)) + " unique words!"

	# loads word vecs (only the valid words)
	f = open(vecs, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			word = tokens[0]

			if word in words:
				wordVecs[word] = np.array(map(float, tokens[1:]))

	# remove unnecessary wordVecs (ones not used in the doc)
	print "\tloaded " + str(len(wordVecs.keys())) + " wordvecs"
	for w in wordVecs.keys():
		if w not in words:
			print "**** ERROR, somehow we loaded a wordvec for a word that's not in our corpus"
			exit(1)
	f.close()

	wordPadding = float(1.0 / len(words))

	donthave = 0
	for w in words:
		if w not in wordVecs.keys():
			donthave += 1
			words.remove(w)

	if donthave > 0:
		print "we dind't have vecs for " + str(donthave) + " words"
		print "i think we shrunk the words{} set to " + str(len(words))
		exit(1)

	# performs PCA to get the 50 most important dimensions
	print "shrinking dims to: " + str(reducedDim)
	pcaVecs = []
	wordOrders = {}
	i=0
	for w in words:
		wordOrders[w] = i
		pcaVecs.append(wordVecs[w])
		i += 1

	pca = PCA(n_components=reducedDim)
	pcaVecs = pca.fit_transform(pcaVecs)

	for w in wordOrders.keys():
		wordVecs[w] = wordVecs[w][0:reducedDim] #pcaVecs[wordOrders[w]] TODO: don't leave this
	pcaVecs = []

# Vetted
def setRandomTopicsPerWordToken():

	global docs

	print "* randomly assigning an initial topic to each word"
	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			wordTopicPairs = []
			for token in tokens[2:]:
				if token in words:
					wordTopicPairs.append((token, choice(topics)))
				else:
					print "**** " + str(token) + " not found!"					
			docs[docName] = wordTopicPairs
	f.close()
	print "* done"


# requires topicToWords[] to have been set, for these are the only words we'll consider printing
# reqires p_w_t to have been updated
def getTopicalWords(k):

	global topicToWords
	global p_w_t

	wordSet = set()
	for w in topicToWords[k]:
		wordSet.add(w)

	print "\ttopic " + str(k) + " had " + str(len(wordSet)) + " unique words assigned to the topic"
	topWords = sorted(p_w_t[k].items(), reverse=True, key=operator.itemgetter(1))
	topCount = 0
	i_ = 0
	printList = []
	while topCount < 20 and i_ < len(topWords):
		if topWords[i_][0] in wordSet:
			printList.append(topWords[i_])
			topCount += 1
		i_ += 1

	return printList



def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
    	#print "removing item " + str(ind)# + " bc " + str(L[ind]) + " = " + str(arr)
    	#print " but [1]: " + str(L[ind+1])
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def getEuclideanDistance(w1,w2):
	return getCosSim(w1,w2)

def getCosSim(w1,w2):
	sum = 0
	for i in range(len(w1)):
		sum += math.pow(w1[i] - w2[i], 2)
	sum = math.sqrt(sum)
	return sum


# Vetted
def PLSA():

	global reducedDim
	global ptd
	global pwt
	global ntw
	global nto

	ntw = defaultdict(lambda: 0)
	nto = defaultdict(lambda: 0)

	print docs.keys()
	words = set()
	for i in docs.keys(): # iterates over docs
		print "# words: " + str(len(docs[i]))
		print i
		for j in docs[i]: # iterates over each (word,topic) pair
			ndt[(i,j[1])] +=1
			ndo[i] += 1
			ntw[(j[1],j[0])] +=1
			nto[j[1]] += 1
			words.add(j[0])

	nwords = len(words)
	ntopics = len(topics)

	# Functions for computing P(t,d) and P(w,t)
	ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)
	pwt = lambda w, t: float(ntw[(t, w)] + alpha) / float(nto[t] + alpha * nwords)


	for _ in range(iterations):
		print "iter: " + str(_)
		loglikelihood = 0

		for i in docs.keys():

			print "iter: " + str(_) + "; doc: " + i + " (" + str(len(docs[i])) + " words)"

			x=0
			for j in docs[i]:
				#print j
				ndt[(i,j[1])] -= 1
				ndo[i] -= 1
				ntw[(j[1],j[0])] -= 1
				nto[j[1]] -= 1

				# Create our distribution.
				dist = [ptd(k, i) * pwt(j[0], k) for k in topics]
				sdist = sum(dist)

				loglikelihood += math.log(sum(dist))

				# Normalize our distribution.
				ndist = map(lambda t: float(t) / sdist, dist)

				r = random()
				for k in range(len(ndist)):
					r -= ndist[k]
					if r < 0:
						break
				docs[i][x] = (j[0],k)

				# increment the counts
				ndt[(i, k)] += 1
				ndo[i] += 1
				ntw[(k, j[0])] += 1
				nto[k] += 1
				#print "picked " + str(k)
				x += 1

		# done going through all docs for the given iteration
		print "\n** The probabilites of topics for doc8 (with " + str(len(docs['doc8'])) + " words):"
		for t in topics:
			print "Topic " + str(t) + ":", ptd(t, 'doc8')

		print "\n2. The probabilites of topics for doc17 (with " + str(len(docs['doc17'])) + " words):"
		for t in topics:
			print "Topic " + str(t) + ":", ptd(t, 'doc17')

		print "\n2. The probabilites of topics for doc18 (with " + str(len(docs['doc18'])) + " words):"
		for t in topics:
			print "Topic " + str(t) + ":", ptd(t, 'doc18')
		print "\n2. The probabilites of topics for doc20 (with " + str(len(docs['doc20'])) + " words):"
		for t in topics:
			print "Topic " + str(t) + ":", ptd(t, 'doc20')

		print "\n3. The most probable 15 words for each topic:"
		for t in topics:
			# My poor man's argmaxn (argmax for the top n items)...
			rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
			rs.sort(reverse = True)
			print "Topic " + str(t) + ":", map(lambda r: r[1], rs[:15])

		print "**** likelihood for iter " + str(_) + ": " + str(loglikelihood)

	print "*** printing mean (weighted and unweighted) vectors for each topic, along w/ the word2vecs that are most similar to each"
	for t in topics:
	# My poor man's argmaxn (argmax for the top n items)...
		rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
		rs.sort(reverse = True)
		topwords = map(lambda r: r[1], rs[:min(len(rs),nbestwords)])
		print "len " + str(len(topwords))
		# constructs the weighted and unweighted mean vectors for the 'topwords' (which is a specified length)
		sumvec = np.zeros(reducedDim)
		sumwvec = np.zeros(reducedDim)
		for w in topwords:
			sumwvec = np.add(sumwvec, pwt(w,t)*wordVecs[w])
			sumvec = np.add(sumvec, wordVecs[w])
			#print "adding " + str(wordVecs[w])
		#print sumvec

		# finds closest WEIGHTED word2vec
		bestWW = ""
		minW = 9999999
		for w in wordVecs:
			curCS = sp.spatial.distance.cosine(wordVecs[w], sumwvec)
			if curCS < minW:
				minW = curCS
				bestWW = w

		bestW = ""
		minW = 9999999
		for w in wordVecs:
			curCS = sp.spatial.distance.cosine(wordVecs[w], sumvec)
			if curCS < minW:
				minW = curCS
				bestW = w
		print "\nTopic " + str(t) + ":\n---------"
		print "closest word2vec (unweighted): " + str(bestW)
		print "closest word2vec (weighted): " + str(bestWW)
		#print "Top N per P(W|Z): " + str(topwords)


# Vetted
def constructGaussians():

	global mixtureVecs
	global mixtureMeans
	global mixtureCovs
	global topicToWords
	global p_w_t
	global words

	print "* running constructGaussians"
	nwords = len(words)
	print "nwords: " + str(nwords)
	pwt = lambda w, t: float(ntw[(t, w)] + alpha) / float(nto[t] + alpha * nwords)

	for t in topics:
		rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
		rs.sort(reverse = True)
		topwords = map(lambda r: r[1], rs[:min(len(rs),nbestwords)])
		print "topic " + str(t) + " len " + str(len(topwords))

		for w in topwords:
			print " adding " + str(w)
			topicToWords[t].append(w) # append to topic's words
			curVecs = []
			if t in mixtureVecs.keys():
				curVecs = mixtureVecs[t]
			#print "raw: " + str(wordVecs[w])
			#print " * " + str(pwt(w,t))
			#print "scaled: " + str(np.multiply(pwt(w,t),wordVecs[w]))
			curVecs.append(np.array(np.multiply(pwt(w,t),wordVecs[w]))) #[np.random.randint(1,30), np.random.randint(30,50), np.random.randint(50,70)]
			mixtureVecs[t] = curVecs

	for k in mixtureVecs.keys():
		print "top " + str(k) + " has " + str(len(mixtureVecs[k])) + " vecs and " + str(len(topicToWords[k])) + " topicToWords items"	

	# calculates initial covs and means for each k-mixtures
	print "* calculating initial cov matrix and mean vectors for each of the k mixtures..."
	for k in range(len(topics)):
		x = np.array(mixtureVecs[k]).T
		mixtureCovs[k] = np.cov(x)
		print "# cov " + str(len(mixtureCovs[k]))
		print "dimension of cov[" + str(k) + "]: " + str(len(mixtureCovs[k][0]))
		print "cov shape: " + str(mixtureCovs[k].shape)
		#print mixtureCovs[k]

		mixtureMeans[k] = np.mean(x, axis=1)
		#print "mixtureMeans[" + str(k) + "]: " + str(mixtureMeans[k])
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))
				# finds closest WEIGHTED word2vec
		bestWW = ""
		minW = 9999999
		for w in wordVecs:
			curCS = sp.spatial.distance.cosine(wordVecs[w], mixtureMeans[k])
			if curCS < minW:
				minW = curCS
				bestWW = w
		print "\nTopic " + str(t) + ":\n---------"
		print "closest word2vec (weighted) to GAUSSIAN!: " + str(bestWW)

	# initializes the p(w|z) normalized probs; we only update each topic's probs once we update the other memberships
	for k in range(ntopics):
		print "* init'ing p(w|z) for mixture " + str(k)
		wp = updateWordProbs(k)
		p_w_t[k] = wp
		
	print "done!"


if __name__ == "__main__":
	iterations = 3
	alpha = 0.35
	updateProb = 1
	topics = range(15)
	reducedDim = 300
	maxKMeansIterations = 5
	wordPadding = 0
	nbestwords = 15 # how many words per topic to base our mean vector on

	input = "../input/docs/news_500-filtered.txt"
	vecs = "../input/word2vec/GoogleNews_news.txt"
	
	## GLOBAL VARIABLES ##
	docs = {}
	wordVecs = {}
	words = set()
	p_w_t = {} # stores a list for topics.  each index is for a specific topic and contains a map (word -> prob)
	topicToWords = defaultdict(list) # stores the words currently assigned to the given topic

	# represents the k-mixtures
	ndt = defaultdict(lambda: 0)
	ndo = defaultdict(lambda: 0)

	mixtureVecs = defaultdict(np.array)
	mixtureCovs = {}
	mixtureMeans = {}
	ntopics = len(topics)

	print "running with: \n\titerations: "+ str(iterations) + "\n\talpha: " + str(alpha) + "\n\t# topics: " + str(len(topics))

	setValidWords()

	# initializes each word token to a randomly chosen topic
	setRandomTopicsPerWordToken()

	# runs PLSA
	PLSA()

	# run TopicToVec
	constructGaussians()
	#TopicToVec()

