import gensim as gs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

doc1 = 'Lady love Lovely to see Lady you Saved my life from Late and drawn Cold and torn The sky that will take me from you'
doc2 = 'Lady day Sing to me Say you\'ll stay Warm and cold and Lady true Hope and time The things that hold my life'
doc3 = 'Lady take these days from me my heart Then I will go Tomorrow calls A little time apart From everything But I don\'t want to leave a thing at all Don\'t let me be here Without my clothes Without life This melody Melody of love Melody of you Lady take these days from me my heart'
doc4 = 'Then I will go Tomorrow calls A little time apart From everything But I don\'t want to leave a thing at all Don\'t let me be here Without my clothes Without life This melody Melody of love Melody of you'
queryString = 'Lady you save my life'

# normalise
doc1 = doc1.lower()
doc2 = doc2.lower()
doc3 = doc3.lower()
doc4 = doc4.lower()
queryString = queryString.lower()

# convert term string into Terms
terms = queryString.split(' ')

# dtm
vectorizer = CountVectorizer(doc1)
dtm = vectorizer.fit_transform([doc1,doc2,doc3,doc4,queryString])
vocabs = vectorizer.get_feature_names()
dtm = dtm.toarray()
q = dtm[4]
dtm = np.resize(dtm, (4, dtm.shape[1]))

# svd
u, s, v = np.linalg.svd(np.transpose(dtm))
vt = np.transpose(v)

# specify sample dimension
k = 3

# reduce the dimension
ur = np.resize(u, (u.shape[0],k))
sr_flat = np.resize(s, (1,k))
sr = np.array((
                [(sr_flat[0][0]),0,0], 
                [0,(sr_flat[0][1]),0], 
                [0,0,(sr_flat[0][2])]
            ))
vtr = np.resize(vt, (k, vt.shape[1]))

# calculate q term matrix
qr = np.dot(np.dot(np.transpose(q), ur), sr)
print(vtr)
print(q)
print(vocabs)
print(np.dot(qr, vtr))