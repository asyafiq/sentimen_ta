import time

import re
import string
import csv
start_time = time.time()
reader = csv.reader(open('normalisasi.csv', 'r'))

#data2=pd.read_excel('redatakaryaakhir/pegitraveloka.xls')
data2=data_.copy()
d = {}
for row in reader:
    k,v= row
    d[string.lower(k)] = string.lower(v)
    #print d[k]
pat = re.compile(r"\b(%s)\b" % "|".join(d))
for i in range(len(data2)):
    text = string.lower(data2['tweetText'].iloc[i])
    
    text = pat.sub(lambda m: d.get(m.group()), text)
    #print text
    
    data2['tweetText'].iloc[i]=text
data2.head()
#print("--- %s detik --- atau %s menit" % ((time.time() - start_time),(time.time() - start_time)/60))


#===============================hilangkan url
pattern=r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
#r'^https?:\/\/.*[\r\n]*'
data3=data2.copy()
for i in range(len(data3)):
    data3['tweetText'].iloc[i] = re.sub(pattern,'', data3['tweetText'].iloc[i], flags=re.MULTILINE)
data3.head()