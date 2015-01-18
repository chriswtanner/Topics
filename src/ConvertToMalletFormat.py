#!/usr/bin/python
# written by Chris Tanner

# converts eugene's file format to MALLET format

if __name__ == "__main__":
	input = "../data/news1000.txt"
	output = "../data/news1000-mallet.txt"

	f = open(input, 'r')
	out = open(output, 'w')
	curFileTokens = []
	filenum = 0
	for line in f:
		line = line.strip()
		if (line.isdigit() == True and len(line) > 1):

			# print it
			if (len(curFileTokens) > 0):
				out.write("doc" + str(filenum) + " doc" + str(filenum))

				for t in curFileTokens:
					out.write(" " + t)
				out.write("\n")
			else:
				print "line:",line
			filenum = filenum + 1

			curFileTokens = []
		else:
			tokens = line.split(" ")
			for t in tokens:
				curFileTokens.append(t)

	if (len(curFileTokens) > 0):
		out.write("doc" + str(filenum) + " doc" + str(filenum))

		for t in curFileTokens:
			out.write(" " + t)
		out.write("\n")