#sometimes is important to predict whether a given sentence belongs to the category email, news, sports, computers

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

category_map = {'talk.religion.misc': 'Religion', 'rec.autos': 'Autos','rec.sport.hockey':'Hockey','sci.electronics':'Electronics', 'sci.space': 'Space'}
training_data = fetch_20newsgroups(subset='train', categories=category_map.keys(), shuffle=True, random_state=5)
#building a count vectorizer and extract the term counts
vectorizer_count = CountVectorizer()
train_tc = vectorizer_count.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
#defining TEST data
input_data = [
    'Discovery was a space shuttle',
    'Hindu, Christian, Sikh all are religions',
    'We must have to drive safely',
    'Puck is a disk made of rubber',
    'Television, Microwave, Refrigrated all uses electricity'
]

classifier = MultinomialNB().fit(train_tfidf, training_data.target)
#transforming the input data using the count vectorizer
input_tc = vectorizer_count.transform(input_data)
input_tfidf = tfidf.transform(input_tc)
predictions = classifier.predict(input_tfidf)

for sent, category in zip(input_data, predictions):
    print("\nInput Data:", sent, '\n Category:', category_map[training_data.target_names[category]])