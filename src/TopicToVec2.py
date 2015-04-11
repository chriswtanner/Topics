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
	badWords = ['jacqueline', 'landis']

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

def updateWordProbs(k):

	global mixtureMeans
	global mixtureCovs

		#a = math.log(max(0.00001,np.linalg.det(mixtureCovs[k] * 1000) + 0.00001)) - math.log(math.pow(1000,reducedDim))
		#a = np.linalg.slogdet(mixtureCovs[k])[1]
		#b = np.exp(a)
		#b = 1
		#print "a: " + str(a) + " b: " + str(b)
		#try:
		#	first = float(1.0 / (math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(b))))) # + 0.0000000001))

	#print "wp: " + str(wordPadding)
	#exit(1)
	try:

		#firstCovVal = mixtureCovs[k][0][0]
		#scale = float(1.0/firstCovVal)

		#covsScaled = mixtureCovs[k] * scale

		# logdet
		#det = 1 #np.exp(np.linalg.slogdet(mixtureCovs[k])[1])

		# calculates it the standard way
		#det = np.linalg.det(mixtureCovs[k])
		#detScaled = np.exp(math.log(np.linalg.det(covsScaled)) - math.log(math.pow(scale,reducedDim)))

		#first = float(1.0 / (math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(det)))))
		firstScaled = 1 #float(1.0 / (math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(det)))))

		#print "det: " + str(det) + "; 2pi^k: " + str(math.pow(2*math.pi,reducedDim)) + " mult: " + str(math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(det)))) + "; 1 over that: " + str(first)

		#print "first: " + str(first) + " first2: " + str(firstScaled) + "; detScaled: " + str(detScaled)
		#exit(1)

		b = np.linalg.inv(mixtureCovs[k])
		#bScaled = np.linalg.inv(covsScaled)

		wordProbs = {}
		sum = 0

		numVoidWords = 0
		# FOLLOWING IS FOR EUC DISTANCE MEAN WAY
		tmpE = {}
		xvals = []
		# cosVals = []
		origWay = {}
		noInv = {}
		for w in words:
			c = (wordVecs[w] - mixtureMeans[k])
			a = c.T
			x = -0.5 * np.dot(np.dot(a,b),c)
			tmpE[w] = x
			xvals.append(x)

		# 	y = getEuclideanDistance(wordVecs[w],mixtureMeans[k])
		# 	tmpE[w] = y
		# 	cosVals.append(y)
		median = np.median(xvals)
		padding = float(-1.0 * median)
		#maxX = np.max(cosVals)
		#minX = np.min(cosVals)
		#rangeX = maxX - minX

		#print "padding: " + str(padding)
		#exit(1)
		for w in words:

			try:
				#prob = float(1.01 - (tmpE[w] - minX) / rangeX)
				#prob = math.exp(float(1.0 / (1.0 + math.exp(-1.0*(tmpE[w] - median)/2.0)))) - 0.999999 #tmpE[w] + padding)
				c = (wordVecs[w] - mixtureMeans[k])
				a = c.T

				epart = math.exp(tmpE[w] + padding)
				#epart = math.exp(-0.5 * np.dot(np.dot(a,bScaled),c))
				#epartScaled = math.pow(math.exp(-0.5 * np.dot(np.dot(a,bScaled),c)), float(1.0 / math.pow(scale, reducedDim)))
				#probScaled = first * detScaled * math.pow(scale, float(reducedDim/2.0)) * epartScaled
				#print "to the power of: " + str(float(1.0 / math.pow(scale, reducedDim)))
				#print "ebefore power raising: " + str(math.exp(-0.5 * np.dot(np.dot(a,bScaled),c)))
				#print "probScaled: " + str(probScaled) + "; detScaled: " + str(detScaled) + "; invScaled: " + str(bScaled) + "; X of e^(x): " + str(-0.5 * np.dot(np.dot(a,bScaled),c)) + "; epart: " + str(epartScaled)
				prob = firstScaled * epart

				#print prob
				if prob == 0:
					print "firstpart: " + str(firstScaled) + "; epart: " + str(epart)
					print "0!!"
					#exit(1)
				wordProbs[w] = prob
				sum += prob
				# except OverflowError:
				# 	numVoidWords += 1

				# 	print "**** ERROR overflow: "
				# 	print "mc: " + str(mixtureCovs[k])
				# 	print "inv: " + str(inv)
				# 	print "w - mixture: " + str(wordVecs[w] - mixtureMeans[k])
				# 	print "(w-mixture) * inv: " + str(np.dot((wordVecs[w] - mixtureMeans[k]),inv))
				# 	print "-0.5 * (w-mixture) * inv * (w-mixture: " + str(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[k]),inv),(wordVecs[w] - mixtureMeans[k])))
				# 	exit(1)
				# 	wordProbs[w] = 0 #wordPadding
				# 	sum += wordPadding
				#print "second: " + str(second)
			except OverflowError:
				#print "tmp: " + str(tmpE[w]) + "; mid: " + str(midX) + "x of e^(x): " + str(float(1.0 / (1.0 + math.exp(-1.0*(tmpE[w] - median)/2.0))) - 1)
				wordProbs[w] = 0
				print "AHH, OVERFLOW!"
					#prob: " + str(prob) + "; firstScaled: " + str(firstScaled) + "; epart: " + str(epart) + " det: " + str(det) + "; x was: " + str(tmpE[w]) + "; x now: " + str(tmpE[w] + padding)
				#exit(1)


		if numVoidWords == len(words):
			print "*** all words had zero mass!!!  here's the inv" + str(inv)
			#print " see: " + str(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[k]).T,inv),(wordVecs[w] - mixtureMeans[k])))
			exit(1) 
		for w in words:
			wordProbs[w] = float(wordProbs[w] / sum)
	
		#print "topwords:" + str(sorted(wordProbs.items(), reverse=True, key=operator.itemgetter(1)))
		return wordProbs
		#except ZeroDivisionError:
		#	print "divide by 0; a: " + str(a) + "; b: " + str(b)# + "; mixtureCovs: " + str(mixtureCovs[k])
		#	exit(1)
		#print " and the slogdet of it" + str(np.linalg.slogdet(mixtureCovs[k])[1])
	except ValueError:

		print "*** VALUE ERROR!"
		print "det: " + str(np.linalg.det(covsScaled))
		print "rest: " + str(math.pow(scale,reducedDim))
		print "log(det): " + str(math.log(np.linalg.det(covsScaled)))
		print "x of e^x: " + str(math.log(np.linalg.det(covsScaled)) - math.log(math.pow(scale,reducedDim)))
		detScaled = np.exp(math.log(np.linalg.det(covsScaled)) - math.log(math.pow(scale,reducedDim)))
		print "error: det:" + str(detScaled)

		#first = float(1.0 / (math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(det)))))
		firstScaled = float(1.0 / (math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(detScaled)))))
		print "firstScaled: " + str(firstScaled)
		#print "det: " + str(det) + "; 2pi^k: " + str(math.pow(2*math.pi,reducedDim)) + " mult: " + str(math.sqrt(math.pow(2*math.pi,reducedDim)*(np.float64(det)))) + "; 1 over that: " + str(first)

		#print "first: " + str(first) + " first2: " + str(firstScaled) + "; detScaled: " + str(detScaled)
		#exit(1)

		b = np.linalg.inv(mixtureCovs[k])


		print "inv: " + str(b)
		#print "log of that: " + str(math.log(np.linalg.det(mixtureCovs[k] * 1000)))
		exit(1)
	#print "\tb: " + str(b) + " and total first part: " + str(first)

	




def TopicToVec():

	current_milli_time = lambda: int(round(time.time() * 1000))

	print "* loaded " + str(len(words)) + " unique word types"

	# initialize mixture memberships
	#print docs.keys()
	for i in docs.keys(): # iterates over docs
		print "# words: " + str(len(docs[i]))
		print i
		for j in docs[i]: # iterates over each (word,topic) pair
			ndt[(i,j[1])] +=1
			ndo[i] += 1
			#mixtureTokens[j[1]].append(j[0])

			topicToWords[j[1]].append(j[0]) # adds to topic's word list

			curVecs = []
			if j[1] in mixtureVecs.keys():
				curVecs = mixtureVecs[j[1]]
			curVecs.append(np.array(wordVecs[j[0]])) #[np.random.randint(1,30), np.random.randint(30,50), np.random.randint(50,70)]
			mixtureVecs[j[1]] = curVecs
	
	for k in mixtureVecs.keys():
		print "top " + str(k) + " has " + str(len(mixtureVecs[k])) + " vecs"



	print "# words in docs: " + str(len(words))

	#print "jac " + str(wordVecs['jacqueline'])
	#print "landis " + str(wordVecs['landis'])


	# prints words closes to qWords
	qWords = ['school', 'month', 'paris']
	# for w in qWords:
	# 	regCS = {}
	# 	pcaCS = {}
	# 	print "looking at: " + str(w)
	# 	print "reg: " + str(wordVecs[w])
	# 	#print "pca: " + str(pcaVecs[wordOrders[w]])

	# 	for o in words:
	# 		regCS[o] = getCosSim(wordVecs[w], wordVecs[o])
	# 		#pcaCS[o] = getCosSim(pcaVecs[wordOrders[w]], pcaVecs[wordOrders[o]])

	# 	sorted_reg = sorted(regCS.items(), key=operator.itemgetter(1))
	# 	#sorted_pca = sorted(pcaCS.items(), key=operator.itemgetter(1))
	# 	print "words closes to: " + str(w)
	# 	print "reg!"
	# 	for i in range(20):
	# 		print str(sorted_reg[i])# + ": " + str(regCS[sorted_reg[i]])
	# 	#print "pca!"
	# 	#for i in range(20):
	# 	#	print str(sorted_pca[i])# + ": " + str(pcaCS[sorted_pca[i]])


	# calculates initial covs and means for each k-mixtures
	print "* calculating initial cov matrix and mean vectors for each of the k mixtures..."
	for k in range(len(topics)):
		x = np.array(mixtureVecs[k]).T
		mixtureCovs[k] = np.cov(x)
		print "# cov " + str(len(mixtureCovs[k]))
		print "dimension of cov[" + str(k) + "]: " + str(len(mixtureCovs[k][0]))
		print "cov shape: " + str(mixtureCovs[k].shape)
		#print mixtureCovs[k]

		mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)
		print "mixtureMeans[" + str(k) + ": " + str(mixtureMeans[k])
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))
	print "done!"

	nwords = len(words)
	ntopics = len(topicToWords.keys())

	print "# topics: " + str(ntopics)

	# Functions for computing P(t,d) and P(w,t)
	ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)
	#pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*(np.float64(np.exp(np.linalg.slogdet(mixtureCovs[t])[1]))))) * math.exp(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[t]),np.linalg.inv(mixtureCovs[t])),(wordVecs[w] - mixtureMeans[t])))
	#diff = lambda w, t: float(np.linalg.norm(wordVecs[w] - mixtureMeans[t]))

	# initializes the p(w|z) normalized probs; we only update each topic's probs once we update the other memberships
	for k in range(ntopics):
		print "* init'ing p(w|z) for mixture " + str(k)
		wp = updateWordProbs(k)
		p_w_t[k] = wp

	for t in topics:

		#
		wordSet = set()
		for w in topicToWords[t]:
			wordSet.add(w)

		print "\ttopic " + str(t) + " had " + str(len(wordSet)) + " unique words assigned to the topic"
		topWords = sorted(p_w_t[t].items(), reverse=True, key=operator.itemgetter(1))
		topCount = 0
		i_ = 0
		printList = []
		while topCount < 20 and i_ < len(topWords):

			if topWords[i_][0] in wordSet:
				printList.append(topWords[i_])
				topCount += 1
			i_ += 1

		print "most probable words " + str(t) + ":" + str(printList)

		print "Topic " + str(t) + ":" + str(topWords[0:20])


	exit(1)

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

			numTopicChanges = 0

			for j in docs[i]:

				curWord = j[0]
				curTopic = j[1]

				# remove word's topic assignment from doc and global counts
				a = current_milli_time()
				ndt[(i,curTopic)] -= 1
				ndo[i] -= 1
				t1a.append(current_milli_time() - a)

				# removes word from topic's lists of words
				if curWord in topicToWords[curTopic]:
					topicToWords[curTopic].remove(curWord)

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
				dist = [ptd(k, i) * p_w_t[k][curWord] for k in topics]
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
				#print "ndist: " + str(ndist)

				docs[i][tokenNum] = (curWord,k)

				# increment the counts
				ndt[(i, k)] += 1
				ndo[i] += 1

				if curTopic != k:
					print "topic changed from " + str(curTopic) + " to " + str(k)
					numTopicChanges += 1

				# adds word to topic's list of words
				topicToWords[k].append(curWord)

				# updates the newly assigned mixture
				#print "mixture vec of topic " + str(k) + " was: " + str(len(mixtureVecs[k]))
				a = current_milli_time()
				mixtureVecs[k].append(curWordVec)
				#print "now: " + str(len(mixtureVecs[k]))
				#print "cov of " + str(k) + " was: " + str(mixtureCovs[k])
				#print "mean of " + str(k) + " was: " + str(mixtureMeans[k])

				x = np.array(mixtureVecs[k]).T
				mixtureCovs[k] = np.cov(x)
				mixtureMeans[k] = np.mean(x, axis=1)

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

			print "\n2. The probabilites of topics for doc17 (with " + str(len(docs['doc17'])) + " words):"
			for t in topics:
				print "Topic " + str(t) + ":", ptd(t, 'doc17')

			print "\n2. The probabilites of topics for doc18 (with " + str(len(docs['doc18'])) + " words):"
			for t in topics:
				print "Topic " + str(t) + ":", ptd(t, 'doc18')
			print "\n2. The probabilites of topics for doc20 (with " + str(len(docs['doc20'])) + " words):"
			for t in topics:
				print "Topic " + str(t) + ":", ptd(t, 'doc20')
			for t in topics:

				print "Topic " + str(t)

				wordSet = set()
				for w in topicToWords[t]:
					wordSet.add(w)

				print "* had " + str(len(wordSet)) + " unique words assigned to the topic"
				topWords = sorted(p_w_t[t].items(), reverse=True, key=operator.itemgetter(1))
				topCount = 0
				i_ = 0
				printList = []
				while topCount < 20 and i_ < len(topWords):
					#print i_
					#print topWords[i_][0]
					if topWords[i_][0] in wordSet:
						printList.append(topWords[i_][0])
						topCount += 1
					i_ += 1

				print "most probable words " + str(t) + ":" + str(printList)
				#print "Topic mean: " + str(mixtureMeans[t])

			print str(numTopicChanges) + " out of " + str(len(docs[i])) + " words changed topics"

		print "**** likelihood for iter " + str(_) + ": " + str(loglikelihood)

				#return
		# print end of all docs for the given iteration

def makeHeatPlot():
	global heatout
	y2 = []
	for i in range(ntopics):
		y2.append("z" + str(i))

	maxwords = 0
	for i in docs.keys():
		if len(docs[i]) > maxwords:
			maxwords = len(docs[i])

	x2 = []
	for i in range(maxwords):
		x2.append(i)


	print " max words: " + str(maxwords)
	print x2
	print y2

	ztmp = defaultdict(lambda : defaultdict(int))
	ztmp2 = defaultdict(lambda : defaultdict(float))
	print "** making a heatmap plot"
	for i in docs.keys(): # iterates over docs
		print "# words: " + str(len(docs[i]))

		doctopics = [docs[i][j][1] for j in range(len(docs[i]))]
		print doctopics
		j=0
		for j in range(len(docs[i])): # iterates over each (word,topic) pair
			k = docs[i][j][1]
			ztmp[k][j] += 1


	
	# normalizes for each word (across all topics), so sum_z P(z|w) = 1
	for i in range(maxwords):
		sum = 0
		for j in range(ntopics):
			sum += ztmp[j][i]
		
		for j in range(ntopics):
			ztmp2[j][i] = float(ztmp[j][i] / float(sum))
	# un-normalized (raw counts)
	# converts to z (list of lists)
	z2 = []
	for i in range(ntopics):
		tmp = []
		
		# iterates over each word for the given topic
		for j in range(150):
			tmp.append(ztmp2[i][j])
		z2.append(tmp)


	data = Data([
		Heatmap(z=z2,x=x2,y=y2,
			colorscale='YIOrRd'
		)
	])
	layout = Layout(
		title='YIOrRd'
	)
	fig = Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename=heatout)

def printWordToTopicCounts():

	wordToTopics = defaultdict(lambda : defaultdict(int))
	for i in docs.keys(): # iterates over docs
		for j in docs[i]: # iterates over each (word,topic) pair
			w = j[0]
			z = j[1]
			if w in wordsToPrint:
				wordToTopics[w][z] += 1

	for w in wordsToPrint:
		print "word " + str(w)
		topWords = sorted(wordToTopics[w].items(), reverse=True, key=operator.itemgetter(1))
		count = 0
		for i in topWords:
			count += i[1]
		print "count: " + str(count)
		print topWords


# Vetted
def PLSA():

	global wordsToPrint
	global reducedDim
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

	fout = open('plsa_topicLikelihoods.csv', 'w')
	for w in wordsToPrint:
		fout.write("," + w)
	fout.write('\n')

	for _ in range(iterations):
		print "iter: " + str(_)
		loglikelihood = 0

		# prints select words' topic probabilities (i.e., P(W|Z))
		if _ == 0 or _ == 6:
			fout.write(str(_) + "th it:")
			wordTopicDists = {}
			for gw in wordsToPrint:
				dist = [pwt(gw, k) for k in topics]
				sdist = sum(dist)
				ndist = map(lambda t: float(t) / sdist, dist)
				ndist.sort(reverse=True)
				wordTopicDists[gw] = ndist
			
			for it in range(ntopics):
				for gw in wordsToPrint:
					fout.write("," + str(wordTopicDists[gw][it]))
				fout.write("\n")

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
	fout.close()

	for t in topics:
	# My poor man's argmaxn (argmax for the top n items)...
		rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
		rs.sort(reverse = True)
		topwords = map(lambda r: r[1], rs[:min(len(rs),nbestwords)])

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
		print "Topic " + str(t) + ":\n---------"
		print "closest word2vec (unweighted): " + str(bestW)
		print "closest word2vec (weighted): " + str(bestWW)
		#print "Top N per P(W|Z): " + str(topwords)



if __name__ == "__main__":
	iterations = 15
	alpha = 0.35
	updateProb = 1
	topics = range(15)
	reducedDim = 300
	maxKMeansIterations = 5
	wordPadding = 0
	nbestwords = 9999999 # how many words per topic to base our mean vector on

	input = "../input/docs/news_1000-filtered.txt"
	vecs = "../input/word2vec/GoogleNews_news.txt"
	heatout = "heatmap_plsa_" + str(len(topics)) + "norm."

	wordsToPrint = ['america', 'bank', 'durning', 'time', 'bill']
	
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

	#printTopicVecs()
	# prints topic counts for each of the 'good words' (i.e., wordsToPrint[])
	#printWordToTopicCounts()

	# run TopicToVec
	#TopicToVec()

