import pandas as pd

datatrain=pd.read_csv('data untuk training\hasilsteming2.csv')
datatrain=datatrain[0:4300]
datatrain.sentiment[datatrain.sentiment==0]='net'
datatrain.sentiment[datatrain.sentiment==1]='pos'
datatrain.sentiment[datatrain.sentiment==-1]='neg'
datatrain.head()
len(datatrain)


from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
count_vector= CountVectorizer(ngram_range=(1,1))
x_train_counts = count_vector.fit_transform(datatrain['tweetText'])
x_train_counts.toarray()


#x_train_counts.toarray().transpose()
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer().fit(x_train_counts)
X_train_tf = tf_transformer.transform(x_train_counts)
X_train_tf.toarray()


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

X_train_tfidf.toarray()
#X_train_tfidf.data

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.cross_validation import train_test_split as tts
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import numpy as np
from sklearn import svm
from sklearn import metrics
from imblearn.over_sampling import ADASYN
from imblearn.ensemble import BalanceCascade
from imblearn.over_sampling import RandomOverSampler
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 


sm = SMOTE()
X_res, y_res = sm.fit_sample(X_train_tf.toarray(), datatrain['sentiment'])
X_res, y_res=sm.fit_sample(X_res,y_res)
print ('Data sentmen asli {}'.format (Counter(datatrain['sentiment'])))
print('Resampled dataset shape {}'.format(Counter(y_res)))

clf=svm.LinearSVC()
#clf = svm.SVC(decision_function_shape='ovo')
X_train, X_test, y_train, y_test = tts(X_res, y_res,test_size=0.2)
clf.fit(X_train,y_train)
predicted=clf.predict(X_test)
print(metrics.classification_report(y_test, predicted))
presisi_svm_smote=metrics.precision_score(y_test, predicted,average='macro')
recall_svm_smote=metrics.recall_score(y_test, predicted,average='macro')
f1_svm_smote=metrics.f1_score(y_test, predicted,average='macro')
akurasi_svm_smote=metrics.accuracy_score(y_test, predicted)
print "Presisi:",presisi_svm_smote 
print "Recall:", recall_svm_smote
print "F1-Score:", f1_svm_smote
print "Akurasi:", akurasi_svm_smote

