#BoW = Bag of Word which is a model in natural language processing, basically used to extract the features from text so
# that the text can be used in modeling such that in machine learning algorithms

from sklearn.feature_extraction.text import CountVectorizer
Sentences=['We are using the Bag of Word model', 'Bag of Word model is used for extracting the features.']
vectorizer = CountVectorizer()
features_text = vectorizer.fit_transform(Sentences).todense()
print(vectorizer.vocabulary_)
