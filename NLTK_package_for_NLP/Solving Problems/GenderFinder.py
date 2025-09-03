#a classifier would be trained to find the gender(male or female) by providing the names
import random
import nltk
nltk.download('names')
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

#extract the last N letters form the input word, the letters will act as features
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return{'feature': last_n_letters.lower()}

if __name__ == '__main__':
    #create the training data using labeled names (male as well as female) available in NLTK
    male_list = [(name, 'male') for name in names.words('male.txt')]
    female_list = [(name, 'female') for name in names.words('female.txt')]
    data = (male_list + female_list)
    random.seed(5)
    random.shuffle(data)

    namesInput = ['Giorgia', 'Leonardo', 'Simone', 'Federica']
    train_sample = int(len(data) * 0.8)
    #iteratinf through different lengths so that the accuracy can be compared
    for i in range(1, 6):
        print('\nNumber of end letters:', i)
        features = [(extract_features(n, i), gender) for (n, gender) in data]
        train_data, test_data = features[:train_sample], features[train_sample:]
        classifier = NaiveBayesClassifier.train(train_data)
        accuracy_classifier = round(100*nltk_accuracy(classifier, test_data), 2)
        print('Accuracy: '+ str(accuracy_classifier) + " %")
        for name in namesInput:
            print(name, '==>', classifier.classify(extract_features(name, i)))