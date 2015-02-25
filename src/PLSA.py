from collections import defaultdict
from random import choice, random
from math import log
from sys import argv

iterations = 10 # Instead of iterating until convergence, I hardcode 10 iterations.
alpha = 0.35 # I belive this is near the optimal alpha value based on extensive testing.
topics = range(50) # Assignments says we have 50 topics.

# Parse the wonderfully formatted data into our documents list.
documents = []
with open(argv[1]) as f:
	article = []
	for line in f:
		if line[0] == " ":
			article += line.split()
		else:
			if article != []:
				documents.append(article)
				article = []

# Assign a random topic to each word in each article in our documents list.
documents = [[(word, choice(topics)) for word in document] for document in documents]

# Them counts.
ndt = defaultdict(lambda: 0)
ntw = defaultdict(lambda: 0)
ndo = defaultdict(lambda: 0)
nto = defaultdict(lambda: 0)

# Prime our counts.
words = set() # All unique words.
for i in range(len(documents)):
	for j in range(len(documents[i])):
		ndt[(i, documents[i][j][1])] += 1
		ndo[i] += 1
		ntw[(documents[i][j][1], documents[i][j][0])] += 1
		nto[documents[i][j][1]] += 1
		words.add(documents[i][j][0])

nwords = len(words) # Number of words (Look ma! No types!)
ntopics = len(topics) # Number of topics

# Functions for computing P(t,d) and P(w,t)
ptd = lambda t, d: float(ndt[(d, t)] + alpha) / float(ndo[d] + alpha * ntopics)
pwt = lambda w, t: float(ntw[(t, w)] + alpha) / float(nto[t] + alpha * nwords)

for _ in range(iterations): # Run the algorithm a fixed number of times.
	loglikelihood = 0 # The current log-likelihood.
	for i in range(len(documents)): # For each article...
		for j in range(len(documents[i])): # For each word, topic pair in each article
			# Decrement them counts.
			ndt[(i, documents[i][j][1])] -= 1
			ndo[i] -= 1
			ntw[(documents[i][j][1], documents[i][j][0])] -= 1
			nto[documents[i][j][1]] -= 1

			# Create our distribution.
			dist = [ptd(k, i) * pwt(documents[i][j][0], k) for k in topics]
			sdist = sum(dist)

			# Compute the log-likelihood for this distribution and add it to the global sum.
			loglikelihood += log(sum(dist))

			# Normalize our distribution.
			ndist = map(lambda t: float(t) / sdist, dist)

			# Pick a random index from our distribution
			r = random()
			for k in range(len(ndist)):
				r -= ndist[k]
				if r < 0:
					break
			# Yay for Python leaking variables!

			# Update the word with the new topic.
			documents[i][j] = (documents[i][j][0], k)

			# Increment them counts.
			ndt[(i, k)] += 1
			ndo[i] += 1
			ntw[(k, documents[i][j][0])] += 1
			nto[k] += 1

# Note: For alpha = 0.35, we converge at around 7 iterations.
print "\n1. The log-likelihood of the data at convergence (with alpha = " + str(alpha) + \
	" and iterations = " + str(iterations) + "):"
print str(loglikelihood)

print "\n2. The probabilites of topics for article 17 (with " + str(len(documents[16])) + " words):"
for t in topics:
	print "Topic " + str(t + 1) + ":", ptd(t, 16)

print "\n3. The most probable 15 words for each topic:"
for t in topics:
	# My poor man's argmaxn (argmax for the top n items)...
	rs = zip([pwt(w, t) for w in words], words) # I'm assuming we want unique words.
	rs.sort(reverse = True)
	print "Topic " + str(t + 1) + ":", map(lambda r: r[1], rs[:15])