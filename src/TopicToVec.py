from collections import defaultdict
from random import choice, random
#from math import log
from sys import argv
import numpy as np
from sklearn.decomposition import PCA
import operator
import math
import os
import sys
import time

def setRandomTopicsPerWordToken():
	print "running PLSA with: \n\titerations: "+ str(iterations) + "\n\talpha: " + str(alpha) + "\n\t# topics: " + str(len(topics))

	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			wordTopicPairs = []
			for token in tokens[2:]:
				wordTopicPairs.append((token, choice(topics)))
			docs[docName] = wordTopicPairs

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

def getCosSim(w1,w2):
	sum = 0
	for i in range(len(w1)):
		sum += math.pow(w1[i] - w2[i], 2)
	sum = math.sqrt(sum)
	return sum

def updateWordProbs(k):

	a = np.linalg.slogdet(mixtureCovs[k])[1]
	b = np.exp(a)
	#print "a: " + str(a) + " b: " + str(b)
	first = float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*(np.float64(b))))
	#print "\tb: " + str(b) + " and total first part: " + str(first)

	wordProbs = {}
	sum = 0
	inv = np.linalg.inv(mixtureCovs[k])
	#print "inv: " + str(inv)
	for w in words:
		second = math.exp(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[k]),inv),(wordVecs[w] - mixtureMeans[k])))
		
		#print "second: " + str(second)
		prob = first * second
		wordProbs[w] = prob
		sum += prob

	for w in words:
		wordProbs[w] = float(wordProbs[w] / sum)
		#print "wp: " + str(wordProbs[w])
	return wordProbs

def TopicToVec():

	current_milli_time = lambda: int(round(time.time() * 1000))


	# gets the words in our corpus
	for i in docs.keys():
		for j in docs[i]: # iterates over each (word,topic) pair
			words.add(j[0])
	print "* loaded " + str(len(words)) + " unique word types"


	# loads word vecs (only the valid words)
	f = open(vecs, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			word = tokens[0]

			if word in words:
				wordVecs[word] = np.array(map(float, tokens[1:]))


	# performs PCA to get the 50 most important dimensions
	pcaVecs = []
	wordOrders = {}
	i=0
	for w in words:
		print w
		wordOrders[w] = i
		pcaVecs.append(wordVecs[w])
		i += 1

	pca = PCA(n_components=reducedDim)
	pcaVecs = pca.fit_transform(pcaVecs)

	for w in wordOrders.keys():
		wordVecs[w] = wordVecs[w][0:50] #pcaVecs[wordOrders[w]] TODO: don't leave this
	pcaVecs = []


	# initialize mixture memberships
	print docs.keys()
	for i in docs.keys(): # iterates over docs
		print "# words: " + str(len(docs[i]))
		print i
		for j in docs[i]: # iterates over each (word,topic) pair
			ndt[(i,j[1])] +=1
			ndo[i] += 1
			mixtureTokens[j[1]].append(j[0])

			curVecs = []
			if j[1] in mixtureVecs.keys():
				curVecs = mixtureVecs[j[1]]
			curVecs.append(np.array(wordVecs[j[0]])) #[np.random.randint(1,30), np.random.randint(30,50), np.random.randint(50,70)]
			mixtureVecs[j[1]] = curVecs

	print "# words in docs: " + str(len(words))
	# remove unnecessary wordVecs (ones not used in the doc)
	print "loaded " + str(len(wordVecs.keys())) + " word vecs"
	for w in wordVecs.keys():
		if w not in words:
			del wordVecs[w]

	print "loaded " + str(len(wordVecs.keys())) + " word vecs"
	

	# performs PCA to get the 50 most important dimensions
	# pca = PCA(n_components=50)
	# pcaVecs = pca.fit_transform(pcaVecs)
	# #print "now: " + str(pcaVecs[0])
	qWords = ['school', 'month', 'paris']
	for w in qWords:
		regCS = {}
		pcaCS = {}
		print "looking at: " + str(w)
		print "reg: " + str(wordVecs[w])
		#print "pca: " + str(pcaVecs[wordOrders[w]])

		for o in words:
			regCS[o] = getCosSim(wordVecs[w], wordVecs[o])
			#pcaCS[o] = getCosSim(pcaVecs[wordOrders[w]], pcaVecs[wordOrders[o]])

		sorted_reg = sorted(regCS.items(), key=operator.itemgetter(1))
		#sorted_pca = sorted(pcaCS.items(), key=operator.itemgetter(1))
		print "words closes to: " + str(w)
		print "reg!"
		for i in range(20):
			print str(sorted_reg[i])# + ": " + str(regCS[sorted_reg[i]])
		#print "pca!"
		#for i in range(20):
		#	print str(sorted_pca[i])# + ": " + str(pcaCS[sorted_pca[i]])


	# calculates initial covs and means for each k-mixtures
	print "* calculating initial cov matrix and mean vectors for each of the k mixtures..."
	for k in range(len(topics)):
		x = np.array(mixtureVecs[k]).T
		mixtureCovs[k] = np.cov(x)
		print "# cov " + str(len(mixtureCovs[k]))
		print "dimension of cov[0]: " + str(len(mixtureCovs[k][0]))
		#print mixtureCovs[k]

		mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)
		print "# means: " + str(len(mixtureMeans[k]))
		print mixtureMeans[k]
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))
	print "done!"

	nwords = len(words)
	ntopics = len(mixtureTokens.keys())

	print "# topics: " + str(ntopics)

	# Functions for computing P(t,d) and P(w,t)
	ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)


	pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*(np.float64(np.exp(np.linalg.slogdet(mixtureCovs[t])[1]))+0.000001))) * math.exp(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[t]),np.linalg.inv(mixtureCovs[t])),(wordVecs[w] - mixtureMeans[t])))
	diff = lambda w, t: float(np.linalg.norm(wordVecs[w] - mixtureMeans[t]))

	#pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*np.linalg.det(mixtureCovs[t]))) # * float(math.exp(-0.5* (np.subtract(x - mixtureVecs[t]).T) * np.linalg.inv(mixtureCovs[t]) * (np.substract(x - mixtureVecs[t]))))

	# initializes the p(w|z) normalized probs; we only update each topic's probs once we update the other memberships
	p_w_t = [] # stores a list for topics.  each index is for a specific topic and contains a map (word -> prob)
	for k in range(ntopics):
		print "* init'ing p(w|z) for mixture " + str(k)
		wp = updateWordProbs(k)
		p_w_t.append(wp)


	for _ in range(iterations):
		print "iter: " + str(_)
		loglikelihood = 0

		for k in range(ntopics):
			print "mixturevecs[" + str(k) + "] has: " + str(len(mixtureVecs[k]))

		for i in docs.keys():
			print "iter: " + str(_) + "; doc: " + i + " (" + str(len(docs[i])) + " words)"
			tokenNum=0

			# TODO: is mixtureVecs correct?
			# i want to find the new assignment, while keeping track of which item to remove from which mixture, and the new assignment, too
			# then, make a function that updates all of these.
			# call said function when tokenNum % updateFreq == 0 and at end of going hrough full doc
			t1a = []
			t1b = []
			t2 = []
			t3 = []
			t4 = []

			for j in docs[i]:

				curWord = j[0]
				curTopic = j[1]

				# remove word's topic assignment from doc and global counts
				a = current_milli_time()
				ndt[(i,curTopic)] -= 1
				ndo[i] -= 1
				t1a.append(current_milli_time() - a)

				#print "word, topic: " + str(curWord) + "," + str(curTopic)
				
				curWordVec = wordVecs[curWord]  #mixtureVecs[curTopic][tokenNum]

				a = current_milli_time()
				removearray(mixtureVecs[curTopic], curWordVec)

				#mixtureVecs[curTopic].pop(mixtureVecs[curTopic].index(curWordVec))
				#mixtureVecs[curTopic] = np.delete(mixtureVecs[curTopic],tokenNum)
				#print "length of words is now: " + str(len(mixtureVecs[curTopic]))

				# recalulate the mean and cov for the given topic/mixture
				x = np.array(mixtureVecs[curTopic]).T
				mixtureCovs[curTopic] = np.cov(x)
				mixtureMeans[curTopic] = np.mean(np.array(mixtureVecs[curTopic]).T, axis=1)
				t1b.append(current_milli_time() - a)

				a = current_milli_time()
				dist = [ptd(k, i) * p_w_t[k][curWord] for k in topics] #pwt(curWord, k) for k in topics] #
				sdist = sum(dist)

				loglikelihood += math.log(sum(dist))

				# Normalize our distribution.
				ndist = map(lambda t: float(t) / sdist, dist)
				r = random()
				for k in range(len(ndist)):
					r -= ndist[k]
					if r < 0:
						break
				
				t2.append(current_milli_time() - a)
				#print str(docs[i])
				#print "doc " + str(i) + "; trying to set index " + str(tokenNum) + " to " + str((curWord,k)) + str(dist[k]/sdist)
				#print dist
				#print sdist
				docs[i][tokenNum] = (curWord,k)

				# increment the counts
				ndt[(i, k)] += 1
				ndo[i] += 1

				# updates the newly assigned mixture
				#print "mixture vec of topic " + str(k) + " was: " + str(mixtureVecs[k])
				a = current_milli_time()
				mixtureVecs[k].append(curWordVec)
				#print "cov of " + str(k) + " was: " + str(mixtureCovs[k])
				#print "mean of " + str(k) + " was: " + str(mixtureMeans[k])

				x = np.array(mixtureVecs[k]).T
				mixtureCovs[k] = np.cov(x)
				mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)

				t3.append(current_milli_time() - a)

				a = current_milli_time()
				# update p(w|z)

				if random() < updateProb:
					wp = updateWordProbs(k)
					p_w_t.insert(k,wp)
				#print "pwt[" + str(k) + "] was " + str(p_w_t[k])

				#print "pwt[" + str(k) + "] now " + str(p_w_t[k])
				t4.append(current_milli_time() - a)
				#print "cov now: " + str(mixtureCovs[k])
				#print "mean now: " + str(mixtureMeans[k])
				tokenNum += 1

				# print "** avg times"
				# print str(float(sum(t1a)/len(t1a))) + " from " + str(len(t1a))
				# print str(float(sum(t1b)/len(t1b))) 
				# print str(float(sum(t2)/len(t2)))
				# print str(float(sum(t3)/len(t3)))
				# print str(float(sum(t4)/len(t4))) + " from " + str(len(t4))
			print "\n** The probabilites of topics for doc8 (with " + str(len(docs['doc8'])) + " words):"
			for t in topics:
				print "Topic " + str(t) + ":", ptd(t, 'doc8')
			for t in topics:
				topWords = sorted(p_w_t[t].items(), reverse=True, key=operator.itemgetter(1))
				print "Topic " + str(t) + ":" + str(topWords[0:20])
				print "Topic mean: " + str(mixtureMeans[t])
		print "**** likelihood for iter " + str(_) + ": " + str(loglikelihood)

				#return

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
			topWords = sorted(p_w_t[t].items(), key=operator.itemgetter(1))
			print "Topic " + str(t) + ":" + topWords[0:20]



def PLSA():
	#ndt = defaultdict(lambda: 0)
	ntw = defaultdict(lambda: 0)
	#ndo = defaultdict(lambda: 0)
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
		print str(loglikelihood)
	#print ndt[('doc13',0)]
	#print ndo['doc13']
	#print ntw[(0,'africa')]
	#print nto[0]

	print "\n2. The probabilites of topics for doc1 (with " + str(len(docs['doc1'])) + " words):"
	for t in topics:
		print "Topic " + str(t + 1) + ":", ptd(t, 'doc1')

	print "\n3. The most probable 15 words for each topic:"
	for t in topics:
		# My poor man's argmaxn (argmax for the top n items)...
		rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
		rs.sort(reverse = True)
		print "Topic " + str(t + 1) + ":", map(lambda r: r[1], rs[:15])

if __name__ == "__main__":
	iterations = 20
	alpha = 0.35
	updateProb = .10
	topics = range(12)
	reducedDim = 50

	input = "../input/docs/news_50-filtered.txt"
	vecs = "../input/word2vec/GoogleNews_news.txt"

	## GLOBAL VARIABLES ##
	docs = {}
	wordVecs = {}
	words = set()

	# represents the k-mixtures
	ndt = defaultdict(lambda: 0)
	ndo = defaultdict(lambda: 0)
	mixtureTokens = defaultdict(list)
	mixtureVecs = defaultdict(np.array)
	mixtureCovs = {}
	mixtureMeans = {}
	ntopics = 0
	# initializes each word token to a randomly chosen topic
	setRandomTopicsPerWordToken()

	# runs PLSA
	#PLSA()

	# run TopicToVec
	TopicToVec()