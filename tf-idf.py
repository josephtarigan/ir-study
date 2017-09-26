import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

d1 = 'shipment of gold damaged in a fire'
d2 = 'delivery of silver arrived in a silver truck'
d3 = 'shipment of gold arrived in a truck'
q = 'gold silver truck'
n = 3

# normalise
d1 = d1.lower()
d2 = d2.lower()
d3 = d3.lower()
q = q.lower()

# convert term string into Terms
terms = q.split(' ')

# dtm
vectorizer = CountVectorizer(d1)
dtm = vectorizer.fit_transform([d1,d2,d3,q])
vocabs = vectorizer.get_feature_names()
dtm = dtm.toarray()
q = dtm[3]
dtm = np.resize(dtm, (3, dtm.shape[1]))

df = np.sum(dtm, axis=0)
idf = np.log10(n/df)

a = np.multiply(dtm, idf)
rsv = np.dot(a,q)

print('RSV : ' + str(rsv))

tfidf = np.multiply(df,idf)

print ('Result : 2, 3, 1')