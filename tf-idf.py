import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# joseph at tarigan at binus dot ac dot id
# 1801622663

# documents
d1 = 'shipment of gold damaged in a fire'
d2 = 'delivery of silver arrived in a silver truck'
d3 = 'shipment of gold arrived in a truck'
n = 3

# query
q = 'gold silver truck'

# normalise the words
d1 = d1.lower()
d2 = d2.lower()
d3 = d3.lower()
q = q.lower()

# convert query string into terms
terms = q.split(' ')

# dtm
vectorizer = CountVectorizer(d1)
tfm = vectorizer.fit_transform([d1,d2,d3,q])
vocabs = vectorizer.get_feature_names()
tfm = tfm.toarray()

# query doc-term
q = tfm[3]
# documents doc-term
tfm = np.resize(tfm, (3, tfm.shape[1]))

# flatten the dtm to create df
df = np.sum(tfm, axis=0)
# calculate idf
idf = np.log10(n/df)

# normalise the tf-idf
w = np.divide(np.multiply(tfm,np.log10(np.divide(n,df))), np.sqrt(np.sum(np.multiply(np.power(tfm, 2), np.power(np.log10(np.divide(n,df)), 2)))))

# create A matrix, for comparison purpose
A = np.multiply(tfm, idf)
q = np.multiply(q, idf)

# calculate rsv 
rsv = np.dot(A,q)
rsvw = np.dot(w, q)

print('RSV : ' + str(rsv))
print('Normalised RSV : ' + str(rsvw))
print('Result : 2, 3, 1')

'''RSV : [ 0.17609126  0.52827378  0.35218252]
Normalised RSV : [ 0.1714372   0.51431159  0.34287439]
Result : 2, 3, 1
'''