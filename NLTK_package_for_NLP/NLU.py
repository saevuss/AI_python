#Natural Language Processing (NLP) refers to AI method of communicating with intelligent systems using a natural language
# such as English
#there are two components of NLP:
#   - NLU = natural language understanding which involves mapping the given input in NL into useful representations and
#           analyzing different aspect of the language
#   - NLG = natural language generation which is the process of producing meaningful phrases and sentences in form of
#           natural language from some internal representation, it involves text planning, sentence planning and text realization

#NLTK = natural language toolkit package

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize #this package divides the input text into words
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer #prende una parola e la riduce alla sua radice, tipo latticino --> latte
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#DT means the determinant, VBP means the verb, JJ means the adjective, IN means the preposition and NN means the noun
sentence = [("a", "DT"),("clever", "JJ"),("fox","NN"),("was","VBP"),("jumping","VBP"),("over","IN"),("the","DT"),( "wall","NN")]

#it is needed the grammar in the form of regular expression
grammar = "NP:{<DT>?<JJ>*<NN>}"
#parser to parse the grammar
parser_chunking = nltk.RegexpParser(grammar)
output_chunk = parser_chunking.parse(sentence) #parsing the sentence
output_chunk.pretty_print()
output_chunk.draw()
print(output_chunk)

