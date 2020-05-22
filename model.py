

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('reviews.csv')
df=data
df=df[['Reviews','Rating']]
df=df[df['Rating']!=3]
df=df.reset_index(drop=True)
df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data=[]

for i in range(0,1000):
        review=df["Reviews"][i]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)
        
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500,min_df=5,ngram_range=(1,2))
X = cv.fit_transform(df["Reviews"].apply(lambda x: np.str_(x))) 
y=df.iloc[:,2].values
y.shape
y=y.reshape(-1,1)

import pickle
pickle.dump(cv, open("cv.pkl", "wb"))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()



model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 1500))

model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=10)
model.save('mymodel.h5')