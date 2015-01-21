from collections import defaultdict
from random import choice, random
#from math import log
from sys import argv
import numpy as np
import math

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

def TopicToVec():

	# loads word vecs
	f = open(vecs, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			word = tokens[0]
			wordVecs[word] = np.array(map(float, tokens[1:]))
	print "loaded " + str(len(wordVecs.keys())) + " word vecs"

	ndt = defaultdict(lambda: 0)
	ndo = defaultdict(lambda: 0)

	# represents the k-mixtures
	mixtureTokens = defaultdict(list)
	mixtureVecs = defaultdict(np.array)
	mixtureCovs = {}
	mixtureMeans = {}

	# initialize mixture memberships
	print docs.keys()
	words = set()
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
			words.add(j[0])

	
	# calculates initial covs and means for each k-mixtures
	print "* calculating initial cov matrix and mean vectors for each of the k mixtures..."
	for k in range(50):
		x = np.array(mixtureVecs[k]).T
		mixtureCovs[k] = np.cov(x)
		print "# cov " + str(len(mixtureCovs[k]))
		print "dimension of cov[0]: " + str(len(mixtureCovs[k][0]))
		#print mixtureCovs[k]

		mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)
		print "# means: " + str(len(mixtureMeans[k]))
		#print mixtureMeans[k]
		#return 
		#print str(k) + ": " + str(len(mixtureTokens[k]))
	print "done!"

	nwords = len(words)
	ntopics = len(mixtureTokens.keys())

	print "# topics: " + str(ntopics)

	# Functions for computing P(t,d) and P(w,t)
	ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)
	pwt = lambda w, t: float(1.0)
	#pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*np.linalg.det(mixtureCovs[t]))) # * math.exp(-0.5 * np.dot(np.dot((wordVec[w] - mixtureMeans[t]),np.linalg.inv(mixtureCovs[t])),(wordVec[w] - mixtureMeans[t])))
	diff = lambda w, t: float(wordVec[w] - mixtureMeans[t])
	#pwt = lambda w, t: float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*np.linalg.det(mixtureCovs[t]))) # * float(math.exp(-0.5* (np.subtract(x - mixtureVecs[t]).T) * np.linalg.inv(mixtureCovs[t]) * (np.substract(x - mixtureVecs[t]))))

	for _ in range(iterations):
		print "iter: " + str(_)
		loglikelihood = 0

		for i in docs.keys():
			print "iter: " + str(_) + "; doc: " + i
			tokenNum=0
			for j in docs[i]:

				curWord = j[0]
				curTopic = j[1]
				# remove word's topic assignment from doc and global counts
				ndt[(i,curTopic)] -= 1
				ndo[i] -= 1

				curWordVec = mixtureVecs[curTopic][tokenNum]
				#print "curVec: " + str(curWordVec)
				#print "mixturevec[curtopic] type: " + str(type(mixtureVecs[curTopic]))
				#print "curvec type: " + str(type(curWordVec))
				#print "length of words was: " + str(len(mixtureVecs[curTopic]))
				#if curWordVec in mixtureVecs[curTopic]:
				#	print "** we in there"
				mixtureVecs[curTopic].pop(tokenNum)
				#mixtureVecs[curTopic] = np.delete(mixtureVecs[curTopic],tokenNum)
				#print "length of words is now: " + str(len(mixtureVecs[curTopic]))

				# recalulate the mean and cov for the given topic/mixture
				# print "cov was: " + str(mixtureCovs[curTopic])
				# print "mean was: " + str(mixtureMeans[curTopic])

				x = np.array(mixtureVecs[curTopic]).T
				mixtureCovs[curTopic] = np.cov(x)
				mixtureMeans[curTopic] = np.mean(np.array(mixtureVecs[curTopic]).T, axis=1)
				# print "cov now: " + str(mixtureCovs[curTopic])
				# print "mean now: " + str(mixtureMeans[curTopic])

				#a = float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*np.linalg.det(mixtureCovs[curTopic])))
				# print "a: " + str(a)
				# print str(len(curWordVec))
				# print mixtureMeans[curTopic].shape
				# print (curWordVec - mixtureMeans[curTopic]).shape
				# print "here"
				#b = (curWordVec - mixtureMeans[curTopic])
				#c = np.linalg.inv(mixtureCovs[curTopic])
				#d = (curWordVec - mixtureMeans[curTopic])
				# print "b: " + str(b)
				# print "c: " + str(c)
				# print "d: " + str(d)
				# print str(b.shape)
				# print str(c.shape)
				# print str(d.shape)
				#e = math.exp(-0.5 * np.dot(np.dot((curWordVec - mixtureMeans[curTopic]),np.linalg.inv(mixtureCovs[curTopic])),(curWordVec - mixtureMeans[curTopic])))
				#prob = float(1.0 / math.sqrt(math.pow(2*math.pi,ntopics)*np.linalg.det(mixtureCovs[curTopic]))) * math.exp(-0.5 * np.dot(np.dot((curWordVec - mixtureMeans[curTopic]),np.linalg.inv(mixtureCovs[curTopic])),(curWordVec - mixtureMeans[curTopic])))

				# Create our distribution.
				for k in topics:
					print "pwt for: " + str(k) + " = " + str(pwt(curWord, k))
				dist = [ptd(k, i) * pwt(curWord, k) for k in topics]
				sdist = sum(dist)

				loglikelihood += math.log(sum(dist))

				# Normalize our distribution.
				ndist = map(lambda t: float(t) / sdist, dist)
				r = random()
				for k in range(len(ndist)):
					r -= ndist[k]
					if r < 0:
						break
				#print str(docs[i])
				#print " trying to set index " + str(tokenNum) + " to " + str((curWord,k))
				docs[i][tokenNum] = (curWord,k)

				# increment the counts
				ndt[(i, k)] += 1
				ndo[i] += 1

				# updates the newly assigned mixture
				#print "mixture vec of topic " + str(k) + " was: " + str(mixtureVecs[k])
				mixtureVecs[k].append(curWordVec)
				#print "cov of " + str(k) + " was: " + str(mixtureCovs[k])
				#print "mean of " + str(k) + " was: " + str(mixtureMeans[k])

				x = np.array(mixtureVecs[k]).T
				mixtureCovs[k] = np.cov(x)
				mixtureMeans[k] = np.mean(np.array(mixtureVecs[k]).T, axis=1)
				#print "cov now: " + str(mixtureCovs[k])
				#print "mean now: " + str(mixtureMeans[k])
				tokenNum += 1
		print "**** likelihood: " + str(loglikelihood)

				#return

		print "\n2. The probabilites of topics for doc1 (with " + str(len(docs['doc1'])) + " words):"
		for t in topics:
			print "Topic " + str(t + 1) + ":", ptd(t, 'doc1')

		print "\n3. The most probable 15 words for each topic:"
		for t in topics:
			# My poor man's argmaxn (argmax for the top n items)...
			rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
			rs.sort(reverse = True)
			print "Topic " + str(t + 1) + ":", map(lambda r: r[1], rs[:15])

		print "\n4. The 15 words closest to the mean for each topic:"
		for t in topics:
			# My poor man's argmaxn (argmax for the top n items)...
			rs = zip([diff(w, t) for w in words], words) # I'm assuming we want unique words.
			rs.sort(reverse = True)
			print "Topic " + str(t + 1) + ":", map(lambda r: r[1], rs[:15])			
		return


def PLSA():
	ndt = defaultdict(lambda: 0)
	ntw = defaultdict(lambda: 0)
	ndo = defaultdict(lambda: 0)
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
	iterations = 10
	alpha = 0.35
	topics = range(50)

	input = "../input/docs/news_1000-filtered.txt"
	vecs = "../input/word2vec/GoogleNews_news.txt"
	
	docs = {}
	wordVecs = {}

	# initializes each word token to a randomly chosen topic
	setRandomTopicsPerWordToken()

	# runs PLSA
	#PLSA()

	# run TopicToVec
	TopicToVec()