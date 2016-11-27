
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import recall_score

from string import maketrans

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


df = pd.read_csv("Reviews.csv", index_col = 0)       ## Read data from reviews.csv file
print("Number of reviews = {}".format(len(df)))

#print df.Score

df.Score = df.Score.apply(lambda x : 'pos' if x > 3 else 'neg')   ## Tagged the review pos or neg based on score

print df.groupby('Score')['Summary'].count()

###### Splitting the dataset based on Labels ########
def splitPosNeg(Summaries):
	neg = df.loc[Summaries['Score']=='neg']
	pos = df.loc[Summaries['Score']=='pos']
	return [pos,neg]

[pos,neg] = splitPosNeg(df)

######  Preprocessing ########
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = maketrans(string.punctuation,' '*len(string.punctuation))

def preprocessing(line):
	tokens=[]
	if type(line) is str:
		line = line.translate(translation)
		line = nltk.word_tokenize(line.lower())
		for t in line:
			stemmed = lemmatizer.lemmatize(t)
			tokens.append(stemmed)
	return ' '.join(tokens)

pos_data = []
neg_data = []


for p in pos['Summary']:
	pos_data.append(preprocessing(p))

for n in neg['Summary']:
	neg_data.append(preprocessing(n))

data = pos_data + neg_data
labels = np.concatenate((pos['Score'].values, neg['Score'].values))

[Data_train,Data_test,Train_labels,Test_labels] = train_test_split(data,labels , test_size=0.25, random_state=20160121,stratify=labels)

t = []
for line in Data_train:
	l = nltk.word_tokenize(line)
	for w in l:
		t.append(w)

word_features = nltk.FreqDist(t)
print (len(word_features))

vec_all = CountVectorizer()
ctr_features_all = vec_all.fit_transform(Data_train)

tf_vec_all = TfidfTransformer()
tr_features_all = tf_vec_all.fit_transform(ctr_features_all)

cte_features_all = vec_all.transform(Data_test)
te_features_all = tf_vec_all.transform(cte_features_all)

svd = TruncatedSVD(n_components=200)
ctr_features_truncated = svd.fit_transform(ctr_features_all)
cte_features_truncated = svd.transform(cte_features_all)

models = {'Logistic': linear_model.LogisticRegression(C=1e5)}

results_svd = pd.DataFrame()

foldnum = 0
tfprediction = {}
cprediction = {}

for name, model in models.items():
	model.fit(ctr_features_truncated, Train_labels)
	tfprediction[name] = model.predict(cte_features_truncated)
	tfaccuracy = metrics.accuracy_score(tfprediction[name],Test_labels)

	results_svd.loc[foldnum,'Model']=name
	results_svd.loc[foldnum,'TF-IDF Accuracy']=tfaccuracy
	foldnum = foldnum+1

print(results_svd)