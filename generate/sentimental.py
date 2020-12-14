import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy 
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
import unicodedata
from better_profanity import profanity
from textblob import TextBlob

tokenizer = ToktokTokenizer()

def remove_special_char(text):
    #replace special characters with ''
    text = re.sub('[^\w\s]', '', text)
    #remove chinese character
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    #remove numbers
    text = re.sub('\d+', '', text)
    text = re.sub('_', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
    
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    text = [token for token in tokens if token.lower() not in stopword_list]
    return " ".join(text)

def stem_text(text):
    stemmer = nltk.porter.PorterStemmer()
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
def normalize_corpus(doc):
    
    normalized_corpus = []
    # normalize each document in the corpus
        # remove special character and normalize docs
    doc = remove_special_char(doc)
    # remove stopwards 
    doc = remove_stopwords(doc)
    # lemmatize docs    
    doc = lemmatize_text(doc)
    normalized_corpus.append(doc)
        
    return normalized_corpus

def remove_swear(text):
    censored_text = profanity.censor(text, '*')
    return censored_text

def get_sentiment(text):
    normalized_corpus = normalize_corpus(text)
    #print(normalized_corpus)
    
    sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in normalized_corpus]
    sentiment_category_tb = ['positive' if score > 0 
                                 else 'negative' if score < 0 
                                     else 'neutral' 
                                         for score in sentiment_scores_tb]

    return sentiment_category_tb
sentence = 'Go fuck yourself you piece of shit, you scum'
sentiment = get_sentiment(sentence)
print('Sentiment of your sentence is ',sentiment[0])
sentence_swear_removed = remove_swear(sentence)
print('Your text after removing swa',sentence_swear_removed)
sentence = remove_special_char(sentence)

# p_word = []
# n_word = []
# neu_word = []
# for word in sentence.split():
#     testimonial = TextBlob(word)
#     if testimonial.sentiment.polarity >= 0.5:
#         p_word.append(word)
#     elif testimonial.sentiment.polarity <= -0.5:
#         n_word.append(word)
#     else:
#         neu_word.append(word)

# if sentiment[0]=='negative':
#     print('you have negative words in your text \' ',' '.join(n_word),' \'.Try replacing these words')
# elif sentiment[0]=='positive':
#     print('Your sentence is positive')

# print(n_word)
# print(neu_word)
# print(p_word)


