import sys
import nltk
import collections
import os
from nltk.collocations import *
import operator
import pickle

threshold=11

mPS = nltk.stem.porter.PorterStemmer()

def getdep(s):
	k=s.split('(')
	l=k[1].split(')')[0].split(',')
	m=l[0].split('-')[0]
	n=l[1].split('-')[0]
	n=n[1:]
	#n=mPS.stem(n)
	#m=mPS.stem(m)
	return k[0],m,n

def get_seeds(lsttup, threshold):
	nn=[]
	for k,_ in lsttup:
		#nn.append(mPS.stem(k))
		nn.append(k)
	
	c=collections.Counter(nn)
	if threshold>=0:
		return c.most_common(1)[0][0], [k[0] for k in c.most_common(threshold+1) if k[0]!=c.most_common(1)[0][0]]
	else:
		return c.most_common()


fp=open('parsed_review_10.txt','r')

parsed_well=[[]]
line='0'
nouns=[]
adjectives=[]
ctr=0
while True:
	#print ctr,
	sys.stdout.write("\r%d"%ctr)
	sys.stdout.flush()
	sent_line=fp.readline()		#sentence
	if not sent_line:
		break
	# POS Tag, Get NN and JJ (of sentence)
	tokens = nltk.word_tokenize(sent_line)
	for i in range(0,len(tokens)):
		tokens[i] = tokens[i].lower()

	tagged=nltk.pos_tag(tokens)
	
	nouns += [(t,ctr) for t,k in tagged if k[:2]=='NN' and t not in ['i']]
	adjectives += [(t,ctr) for t,k in tagged if k[:2]=='JJ' and t not in ['i']]
	#
	r_line=fp.readline()
	line=r_line.strip()
	if not r_line:
		break
	while len(line)>0:
		parsed_well[ctr].append(line)
		r_line=fp.readline()
		line=r_line.strip()
	parsed_well.append([])
	if not r_line:
		break
	ctr+=1

fp.close()
print
print "Dependancy creation done."

#print nouns
#print adjectives

prod, feats = get_seeds(nouns, threshold)
#print prod, feats

temp, opins = get_seeds(adjectives,threshold-1)
#print temp,opins

o_seeds = set(opins)		# Contains words with no repetitions
f_seeds = set(feats)		# Contains words with no repetitions
o_list = adjectives	# Contains tuples (word, line number) with multiple copies
f_list = nouns		# Contains tuples (word, line number) with multiple copies

# print "Original Opinions: ", o_seeds
# print "Original Features: ", f_seeds


print "-"*80
print
print "Starting Double Propogation"

# Better than calling Porter Stemmer all the time.
stem={}
for f,_ in f_list:
	try:
		stem[f]=f
	except:
		pass

stem[prod]=prod

for o,_ in o_list:
	stem[o]=o

#print f_list   'roast' is present in f_list
# Testing Number of lines to be called

allines=set([])
allinessum=0
for _,l in f_list:
	allines.add(l)
for _,l in o_list:
	allines.add(l)
for elem in allines:
	for line in parsed_well[elem]:
		allinessum+=1

print len(allines), allinessum

allinessum=0
for p in parsed_well:
	for line in p:
		allinessum+=1

#print len(parsed_well), allinessum
#raw_input()

ctr=[0,0]
while True:
	ctr[0]+=1
	o_new=set([])
	#o_new=[]
	
	#f_new=set([])
	f_new=[]
	
	# Feature Words
	for f,lineno in f_list:
		rels={}
		p_line=parsed_well[lineno]
		
		for line in p_line:
			rel, p1, p2 = getdep(line)
			if rel in rels.keys():
				rels[rel].append((p1,p2))
			else:
				rels[rel]=[(p1,p2)]
		
		for line in p_line:
			ctr[1]+=1
			rel, p1, p2 = getdep(line)
			if rel=='ROOT':
				break
			# R11:
			if p1==f and p2 in o_seeds and rel=='amod':
				#if f=='roasts' and p1=='roasts' and p2=='great' and rel=='amod':
				#	print 'yyes'
				#	print stem.get(f)
				#	print stem.get(prod)
				if f not in f_seeds and stem.get(f)!=stem.get(prod):
				#	if f =='roasts':
				#		print 'yyyes'
				#		print f_new
					try: f_new.add(f)
					except: f_new.append(f)
				#if f=='roasts':
				#	print f_new
			# R12:
			if p2==f and stem.get(p1) == stem.get(prod) and rel=='nsubj':
				if 'amod' in rels.keys():
					for q1,q2 in rels['amod']:
						if stem.get(q1) == stem.get(prod) and q2 in o_seeds:
							if f not in f_seeds and stem.get(f)!=stem.get(prod):
								try: f_new.add(f)
								except: f_new.append(f)
			# R31:
			if stem.get(p2)==stem.get(f) and p1 in f_seeds and rel=='conj:and':
				if f not in f_seeds and stem.get(f)!=stem.get(prod):
					try: f_new.add(f) 
					except: f_new.append(f)
			# R31':
			if stem.get(p1)==stem.get(f) and p2 in f_seeds and rel=='conj:and':
				if f not in f_seeds and stem.get(f)!=stem.get(prod):
					try: f_new.add(f) 
					except: f_new.append(f)
			# R32:
			if stem.get(p2)==stem.get(f) and p1 == 'has' and rel=='dobj':
				if 'nsubj' in rels.keys():
					for q1,q2 in rels['nsubj']:
						if q1 == 'has' and q2 in f_seeds:
							if f not in f_seeds and stem.get(f)!=stem.get(prod):
								try: f_new.add(f) 
								except: f_new.append(f)


	# Opinion Words
	for o,lineno in o_list:
		rels={}
		p_line=parsed_well[lineno]
		
		for line in p_line:
			rel, p1, p2 = getdep(line)
			if rel in rels.keys():
				rels[rel].append((p1,p2))
			else:
				rels[rel]=[(p1,p2)]
		
		for line in p_line:
			ctr[1]+=1
			rel, p1, p2 = getdep(line)
		
			# R21:
			if p2==o and p1 in f_seeds and rel=='amod':
				if o not in o_seeds and stem.get(o)!=stem.get(prod):
					o_new.add(o)
			# R22:
			if p2==o and p1 == prod and rel=='amod':
				if 'nsubj' in rels.keys():
					for q1,q2 in rels['nsubj']:
						if stem.get(q1) == stem.get(prod) and q2 in f_seeds:
							if o not in o_seeds and stem.get(o)!=stem.get(prod):
								o_new.add(o)
			# R41:
			if p1==o and p2 in o_seeds and rel=='conj:and':
				if o not in o_seeds and stem.get(o)!=stem.get(prod):
					o_new.add(o)
			# R41':
			if p2==o and p1 in o_seeds and rel=='conj:and':
				if o not in o_seeds and stem.get(o)!=stem.get(prod):
					o_new.add(o)
			# R42:
			if p2==o and stem.get(p1) == stem.get(prod) and rel=='amod':
				if o not in o_seeds and stem.get(o)!=stem.get(prod):
					o_new.add(o)




	#Convert opinion list to set
	
	c=collections.Counter(f_new)
	#print c.most_common()
	f_new = set([key for key in c.keys() if c[key]>0])
	
	# print "Features:",f_new
	# print "Opinion:",o_new
	
	o_seeds = o_seeds | o_new
	f_seeds = f_seeds | f_new
	
	if len(f_new)==0 and len(o_new)==0:
		break




print "Double Propogation Complete."
#print o_seeds
#print f_seeds

#unigrams = o_seeds | f_seeds

#bigram_measures = nltk.collocations.BigramAssocMeasures()
#finder = BigramCollocationFinder.from_words(unigrams)
#print finder.nbest(bigram_measures.pmi,10)

#scored = finder.score_ngrams(bigram_measures.raw_freq)
#print sorted(bigram for bigram, score in scored)

our_oc = set(o_seeds)
our_fc = set(f_seeds)





pop_analysis = []

for f in our_fc:
	pop_analysis.append([f,0,[]])
	for p_line in parsed_well:
		flag=0
		for l in p_line:
			if f in l:
				flag=1
		if flag==0:
			continue
		for l in p_line:
			_, p1, p2 = getdep(l)
			if p1 in our_oc:
				pop_analysis[-1][2].append(p1)
			if p2 in our_oc:
				pop_analysis[-1][2].append(p2)
	pop_analysis[-1][1]=len(pop_analysis[-1][2])

pop_analysis = sorted(pop_analysis, key = operator.itemgetter(1), reverse = True)

# print pop_analysis

for i in xrange(len(pop_analysis)):
	c=collections.Counter(pop_analysis[i][2])
	pop_analysis[i][2]=c.most_common()
	# print pop_analysis[i][2]

ctr=0
for f,freq,ol in pop_analysis:
	print f, freq, ol
	ctr+=1
	if ctr==10:
		break

f = open('file.pkl', 'r')
cpd_words = pickle.load(f)
# print cpd_words
f.close()

max_prob = 0.
for f, freq, ol in pop_analysis:
	#print f
	for word,count in ol:
		#print word
		cond_prob = cpd_words[f].prob(word)
		if max_prob < cond_prob and count > 10:
			max_prob = cond_prob
			word1 = word
			word2 = f
	break
	break

print word1, word2