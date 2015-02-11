from collections import defaultdict
from random import choice, random, randint
#from math import log
from sys import argv
import numpy as np
from sklearn.decomposition import PCA
import operator
import math
import os
import sys
import time

def setTopicPerWordTokenViaKMeans():

	validWords = []
	# reads words
	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			for token in tokens[2:]:
				if token not in words:
					validWords.append(token)
					words.add(token)
	f.close()

	print str(len(validWords)) + " unique words!"

	# loads word vecs (only the valid words)
	f = open(vecs, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			word = tokens[0]

			if word in words:
				wordVecs[word] = np.array(map(float, tokens[1:]))


	# remove unnecessary wordVecs (ones not used in the doc)
	print "loaded " + str(len(wordVecs.keys())) + " word vecs"
	for w in wordVecs.keys():
		if w not in words:
			del wordVecs[w]
			del words[w]

	print "loaded " + str(len(wordVecs.keys())) + " word vecs"
	f.close()

	# performs PCA to get the 50 most important dimensions
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

	# randomly picks k-centroids
	centroids = {}
	print "nto" + str(ntopics)
	for i in range(ntopics):
		randWord = validWords[randint(0,len(validWords)-1)]
		print "picking rand word: " + str(randWord)
		centroids[i] = wordVecs[randWord]
		print centroids[i]


	# performs k-means
	wordToTopics = defaultdict(lambda: -1)
	topicToWords = defaultdict(list)

	madeChange = True
	i = 0
	while madeChange and i < maxKMeansIterations:
		madeChange = False
		
		print "k-means iteration: " + str(i)

		# assigns each word to its closest k-mean centroid
		for w in words:
			curAssignment = wordToTopics[w]

			# find closest k-means centroid
			bestEuc = 999999
			bestTop = -1
			for k in range(ntopics):
				curEuc = getCosSim(wordVecs[w], centroids[k])
				if curEuc < bestEuc:
					bestEuc = curEuc
					bestTop = k

			if bestTop != curAssignment:
				madeChange = True

				wordToTopics[w] = bestTop
				if w in topicToWords[curAssignment]:
					#print "remove len was " + str(len(topicToWords[curAssignment]))
					topicToWords[curAssignment].remove(w)
					#print "remove len now " + str(len(topicToWords[curAssignment]))
				#print "len was " + str(len(topicToWords[bestTop]))
				topicToWords[bestTop].append(w)
				#print "len now: " + str(len(topicToWords[bestTop]))

		# update the k-means centroids
		if madeChange:
			for k in range(ntopics):
				tmpvecs = []
				for w in topicToWords[k]:
					tmpvecs.append(np.array(wordVecs[w]))
				print len(tmpvecs)
				centNew = np.mean(np.array(tmpvecs).T, axis=1)
				centroids[k] = centNew
				#print len(centNew)
				#print centNew

		i += 1

	# end k-means initialization;
	# let's construct the mixtures and cov's
	# calculates initial covs and means for each k-mixtures
	print "* calculating initial cov matrix and mean vectors for each of the k mixtures..."
	for k in range(ntopics):

		tmpvecs = []
		for w in topicToWords[k]:
			tmpvecs.append(np.array(wordVecs[w]))

		x = np.array(tmpvecs).T
		mixtureCovs[k] = np.cov(x)
		#print "# cov " + str(len(mixtureCovs[k]))
		#print "dimension of cov[0]: " + str(len(mixtureCovs[k][0]))
		#print "cov shape: " + str(mixtureCovs[k].shape)
		#print mixtureCovs[k]
	
		mixtureMeans[k] = np.mean(np.array(x), axis=1)
		#print "means: " + str(mixtureMeans[k])
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))

	# initializes the p(w|z) normalized probs; we only update each topic's probs once we update the other memberships
	p_w_t = [] # stores a list for topics.  each index is for a specific topic and contains a map (word -> prob)
	for k in range(ntopics):
		print "* init'ing p(w|z) for mixture " + str(k)
		wp = updateWordProbs(k)
		p_w_t.append(wp)

	# reads words; determines the mixture it belongs to, probabilistically
	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			wordTopicPairs = []
			for token in tokens[2:]:
				if token in words:

					# determines the mixture it belongs to, probabilistically
					dist = [p_w_t[k][token] for k in topics]
					#print "dist: " + str(dist)
					sdist = sum(dist)
					# Normalize our distribution.
					ndist = map(lambda t: float(t) / sdist, dist)
					#print "ndist: " + str(ndist)
					blah = []
					for di in ndist:
						blah.append(float(1.0/ntopics) + di)

					sdist = sum(blah)
					ndist = map(lambda t: float(t) / sdist, blah)

					#print "ndist2: " + str(ndist)
					r = random()
					for k in range(len(ndist)):
						r -= ndist[k]
						if r < 0:
							break
					print "word " + str(token) + " was " + str(wordToTopics[token]) + " but now " + str(k)
					wordTopicPairs.append((token, k))
			docs[docName] = wordTopicPairs
	f.close()

	print "size of wordvecs: " + str(len(wordVecs))

	# prints closest words to each mean
	for k in range(ntopics):
		wordDists = {}
		for w in topicToWords[k]:
			#print "wvec: " + str(wordVecs[w])
			#print "cent: " + str(centroids[k])
			curEuc = getCosSim(wordVecs[w], centroids[k])
			wordDists[w] = curEuc

		sorted_words = sorted(wordDists.items(), key=operator.itemgetter(1))
		print "words closes to: " + str(k)
		for i in range(20):
			print str(sorted_words[i])


		sorted_words = sorted(p_w_t[k].items(), reverse=True, key=operator.itemgetter(1))
		print "words most probable from density for: " + str(k)
		for i in range(20):
			print str(sorted_words[i])


	print "done!"
	#exit(1)


def setRandomTopicsPerWordToken():
	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			wordTopicPairs = []
			for token in tokens[2:]:
				wordTopicPairs.append((token, choice(topics)))
			docs[docName] = wordTopicPairs
	f.close()

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
		try:
			second = math.exp(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[k]).T,inv),(wordVecs[w] - mixtureMeans[k])))
			prob = first * second
			wordProbs[w] = prob
			sum += prob
		except OverflowError:
			#print "**** ERROR overflow: "
			#print "inv: " + str(inv)
			#print "w - mixture: " + str(wordVecs[w] - mixtureMeans[k])
			#print "(w-mixture) * inv: " + str(np.dot((wordVecs[w] - mixtureMeans[k]),inv))
			#print "-0.5 * (w-mixture) * inv * (w-mixture: " + str(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[k]),inv),(wordVecs[w] - mixtureMeans[k])))
			wordProbs[w] = 0
		#print "second: " + str(second)


	for w in words:
		wordProbs[w] = float(wordProbs[w] / sum)
		#print "wp: " + str(wordProbs[w])
	return wordProbs

def TopicToVec():

	current_milli_time = lambda: int(round(time.time() * 1000))

	print "* loaded " + str(len(words)) + " unique word types"
	
	topicToWords = defaultdict(list)

	# initialize mixture memberships
	print docs.keys()
	for i in docs.keys(): # iterates over docs
		print "# words: " + str(len(docs[i]))
		print i
		for j in docs[i]: # iterates over each (word,topic) pair
			ndt[(i,j[1])] +=1
			ndo[i] += 1
			mixtureTokens[j[1]].append(j[0])

			topicToWords[j[1]].append(j[0]) # adds to topic's word list

			curVecs = []
			if j[1] in mixtureVecs.keys():
				curVecs = mixtureVecs[j[1]]
			curVecs.append(np.array(wordVecs[j[0]])) #[np.random.randint(1,30), np.random.randint(30,50), np.random.randint(50,70)]
			mixtureVecs[j[1]] = curVecs
			
	print "# words in docs: " + str(len(words))

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
		print "cov shape: " + str(mixtureCovs[k].shape)
		#print mixtureCovs[k]

		mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)
		print "means: " + str(mixtureMeans[k])
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))
	print "done!"

	nwords = len(words)
	ntopics = len(mixtureTokens.keys())

	print "# topics: " + str(ntopics)

	# Functions for computing P(t,d) and P(w,t)
	ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)
	#pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*(np.float64(np.exp(np.linalg.slogdet(mixtureCovs[t])[1]))))) * math.exp(-0.5 * np.dot(np.dot((wordVecs[w] - mixtureMeans[t]),np.linalg.inv(mixtureCovs[t])),(wordVecs[w] - mixtureMeans[t])))
	#diff = lambda w, t: float(np.linalg.norm(wordVecs[w] - mixtureMeans[t]))

	# initializes the p(w|z) normalized probs; we only update each topic's probs once we update the other memberships
	p_w_t = [] # stores a list for topics.  each index is for a specific topic and contains a map (word -> prob)
	for k in range(ntopics):
		print "* init'ing p(w|z) for mixture " + str(k)
		wp = updateWordProbs(k)
		p_w_t.append(wp)

	for t in topics:
		topWords = sorted(p_w_t[t].items(), reverse=True, key=operator.itemgetter(1))
		print "Topic " + str(t) + ":" + str(topWords[0:20])



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
				#print sdist
				docs[i][tokenNum] = (curWord,k)

				# increment the counts
				ndt[(i, k)] += 1
				ndo[i] += 1

				# adds word to topic's list of words
				topicToWords[k].append(curWord)

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
		print "**** likelihood for iter " + str(_) + ": " + str(loglikelihood)

				#return
		# print end of all docs for the given iteration



def PLSA():

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
				print "Topic " + str(t + 1) + ":", map(lambda r: r[1], rs[:15])

		print "**** likelihood for iter " + str(_) + ": " + str(loglikelihood)
	#print ndt[('doc13',0)]
	#print ndo['doc13']
	#print ntw[(0,'africa')]
	#print nto[0]



if __name__ == "__main__":
	iterations = 20
	alpha = 0.35
	updateProb = 1
	topics = range(15)
	reducedDim = 50
	maxKMeansIterations = 14


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
	ntopics = len(topics)

	print "running with: \n\titerations: "+ str(iterations) + "\n\talpha: " + str(alpha) + "\n\t# topics: " + str(len(topics))

	# initializes each word token to a randomly chosen topic
	setRandomTopicsPerWordToken()

	# runs k-means (K = # topics) on word types, then probabilistically assigns each word token to a topic based on this
	#setTopicPerWordTokenViaKMeans()
	#exit(1)
	# runs PLSA
	PLSA()

	# run TopicToVec
	#TopicToVec()

		# a = np.array([1,2])
	# print a.shape

	# b = defaultdict(np.array)
	# b1 = np.array([1,1.3])
	# b2 = np.array([2.1,2.1])
	# b3 = np.array([3.2,3.0])
	# c = []
	# c.append(np.array(b1))
	# c.append(np.array(b2))
	# c.append(np.array(b3))
	# b[0] = c


	# x = np.array(b[0]).T
	# print "x " + str(x.shape)

	# mc = np.cov(x)
	# print "mc: " + str(mc.shape)
	# print "mc : " + str(mc)

	# invv = np.linalg.inv(mc)
	# print "inv: " + str(invv)
	# n1 = np.dot(a.T,invv)
	# print n1.shape
	# print n1
