import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import pickle

stop = stopwords.words('english')

dataset = pd.read_csv("Reviews.csv",index_col=0)

words_seq = []
for i in range(1,len(dataset.Text)+1):
	try:
		if (i%1000 == 0):
			print i, len(dataset.Text)
		sent = dataset.Summary[i]   
		tokens = sent.lower().split()
		words_seq.extend([word for word in tokens if word not in string.punctuation])
		sent_text = sent_tokenize(dataset.Text[i])
		for sent in sent_text:
			letters_only = re.sub("[^a-zA-Z]"," ",sent)    
			tokens = letters_only.lower().split() 
			for word in tokens:
				if word not in stop:
					if word not in string.punctuation:
						words_seq.extend([word])
	except:
		pass


print "complete" 
print len(words_seq)

fd_single_word = nltk.FreqDist(words_seq)


 
word_bigrams = nltk.bigrams(words_seq)
condition_pairs = ( (w0,w1) for w0,w1 in word_bigrams )
cfd_words = nltk.ConditionalFreqDist(condition_pairs)
cpd_words = nltk.ConditionalProbDist(cfd_words, nltk.MLEProbDist)


f = open('file.pkl', 'w')
pickle.dump(cpd_words, f)
f.close()

