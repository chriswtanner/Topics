#!/usr/bin/python
# written by Chris Tanner

# reads in a MALLET-formatted file, while only considering docs that
# 
from sets import Set

def loadFilteredDocs(input, vocab):
	lexicon = []
	docs = {}

	# loads lexicon
	f = open(vocab, 'r')
	for line in f:
		line = line.strip().lower()
		lexicon.append(line)

	print "* lexicon has " + str(len(lexicon)) + " words"

	badTokens = Set([])

	f = open(input, 'r')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			print docName
			curValidTokens = []
			for token in tokens[2:]:
				if (token in lexicon):
					curValidTokens.append(token)
				#else:
				#	if (token not in badTokens):
				#		badTokens.add(token)
			if (len(curValidTokens) >= minNumWordsPerDoc):
				docs[docName] = curValidTokens
			else:
				print "doc " + docName + " only had " + str(len(curValidTokens)) + " valid tokens"
	print len(docs.keys())
if __name__ == "__main__":
	input = "../input/docs/news1000-mallet.txt"
	vocab = "../input/word2vec/GoogleNews_Vocab.txt"
	minNumWordsPerDoc = 100

	docs = loadFilteredDocs(input, vocab)


