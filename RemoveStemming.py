#====================================================HILANGKAN TANDA BACA==============================
import string
string.punctuation
#'!Hello.'.strip(string.punctuation)
#'Hello'
data4=data3.copy()
remove=string.punctuation
#remove = remove.replace("@", "") 

for i in range(len(data4)):
    sent=data4['tweetText'].iloc[i]
    kd=' '.join(word.strip(remove) for word in sent.split())
    data4['tweetText'].iloc[i]=kd
data4.head()


import nltk 
#nltk.download()
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords 
reader=pd.read_excel('stopword_id.xls',header=None)
#print reader[0][2]
cachedStopWords = set(stopwords.words("english"))
#more_stopwords = {'oh', 'will', 'hey', 'yet', ...}
#STOPWORDS = STOPWORDS.union(more_stopwords)
cachedStopWords.update(reader[0][:])

#print len(data5)                       
#sent='kenapa yg ini harus!'
data5=data4.copy()
for i in range(len(data5)):
    sent=data5['tweetText'].iloc[i]
    #new_str=" ".join("".join([" " if ch in string.punctuation else ch for ch in sent]).split())
    kt=" ".join([word for word in sent.split() if word not in cachedStopWords])
    data5['tweetText'].iloc[i]=kt

data5.head()


import time
start_time=time.time()
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
data7=data6
stemmer = factory.create_stemmer()
for i in range(len(data7)):
    sent=data7['tweetText'].iloc[i]
    output = stemmer.stem(sent)
    data7['tweetText'].iloc[i]=output
print("--- %s detik --- atau %s menit" % ((time.time() - start_time),(time.time() - start_time)/60))

data7.head()