#!/usr/bin/python
# written by Chris Tanner

# reads in a MALLET-formatted file, while only considering docs that
# 
from sets import Set

def loadFilteredDocs(input, vocab, output):
	lexicon = []
	docs = {}

	# loads lexicon
	f = open(vocab, 'r')
	for line in f:
		line = line.strip().lower()
		lexicon.append(line)

	print "* lexicon has " + str(len(lexicon)) + " words"

	badTokens = Set([]) # mallet tokens not in Vec format
	goodTokens = Set([]) # mallet tokens in Vec format

	f = open(input, 'r')
	fout = open(output, 'w')
	for line in f:
		tokens = line.strip().split(" ")
		if (len(tokens) > 0):
			docName = tokens[0]
			print docName
			curValidTokens = []
			for token in tokens[2:]:
				if (token in lexicon):
					curValidTokens.append(token)

			if (len(curValidTokens) >= minNumWordsPerDoc):
				docs[docName] = curValidTokens
				fout.write(docName + " " + docName)
				for token in curValidTokens:
					fout.write(" " + token)
					goodTokens.add(token)
				fout.write("\n")
			else:
				print "doc " + docName + " only had " + str(len(curValidTokens)) + " valid tokens"
	print "wrote " + str(len(docs.keys())) + " docs to " + output
	fout.close()
	f.close()

	print "total # good tokens: " + str(len(goodTokens))
	fout = open(newvecs, 'w')

	f = open(vecs, 'r')
	written = Set([])

	# tries writing the lowercased-exact-matches first
	for line in f:
		tokens = line.strip().split(" ")
		token = tokens[0]
		if token in goodTokens and token not in written:
			fout.write(token)
			for vals in tokens[1:]:
				fout.write(" " + vals)
			fout.write("\n")
			written.add(token)
	f.close()

	f = open(vecs, 'r')
	for line in f:
		tokens = line.strip().lower().split(" ")
		token = tokens[0]
		if token in goodTokens and token not in written:
			fout.write(token)
			for vals in tokens[1:]:
				fout.write(" " + vals)
			fout.write("\n")
			written.add(token)
	f.close()
	print "we just wrote new vec file: " + str(len(written)) + " words"
	fout.close()

if __name__ == "__main__":

	# change this chunk per run
	input = "../input/docs/acl_1001-mallet.txt"
	output = "../input/docs/acl_1001-filtered.txt"
	newvecs = "../input/word2vec/GoogleNews_acl1k.txt"

	vocab = "../input/word2vec/GoogleNews_Vocab.txt"
	vecs = "../input/word2vec/GoogleNews.txt"


	minNumWordsPerDoc = 100

	loadFilteredDocs(input, vocab, output)


