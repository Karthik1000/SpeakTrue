def speech():
    import speech_recognition as sr       ### Using HMM model as API
    from langdetect import detect  
    from textblob import TextBlob

    r1 = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio_data = r1.listen(source, timeout=7)
        print("Recognizing...")
        try:
            #text = r.recognize_google(audio_data)# api's::--  recognize_bing(): Microsoft Bing Speech , recognize_google_cloud(): Google Cloud Speech , 
            li = []                                         #recognize_ibm(): IBM Speech to Text
            x = (r1.recognize_google(audio_data))
            print(x)
            z = (TextBlob(x))
            lang = (z.detect_language())
            x = (r1.recognize_google(audio_data, language=lang))
            li.append(x)
            y = detect(x)
            li.append(y)
            return li
            #print(text)
        except:
            #print("Sorry, please try again...")
            return 0


def text_translate(inp,ln):
    from googletrans import Translator
    import googletrans
    k=googletrans.LANGUAGES
    if len(inp) == 2:
      lng1 = inp[1]
    else:
      lng1 = ln
    
    for key, value in k.items():
        if lng1 == key:
            x=key
    lng2 = 'english'
    for key, value in k.items():
        if lng2 == value:
            y=key
    if len(inp) == 2:
      text1=inp[0]
    else:
      text1 = inp
    translator = Translator()
    translated = translator.translate(text1, src=x, dest=y)
    #print("The translated text in",lng2,"is : ",translated.text.lower())
    return ((translated.text)).lower()

def text_translate_1(inp,ln):
    from googletrans import Translator
    import googletrans
    k=googletrans.LANGUAGES
    lng1 = inp[1]
    for key, value in k.items():
        if lng1 == key:
            x=key
    lng2 = ln
    for key, value in k.items():
        if lng2 == value:
            y=key
    text1=inp[0]
    translator = Translator()
    translated = translator.translate(text1, src=x, dest=y)
    #print("The translated text in",lng2,"is : ",translated.text.lower())
    return ((translated.text)).lower()


def sub_text_generation(txt,data_set,hd5_file):
  import tensorflow as tf
  import pandas as pd
  import numpy as np
  import re                               
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D
  from tensorflow.keras.models import Model
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models import Sequential


  from keras.models import load_model

  with open(data_set, encoding="utf8") as story:
      story_data = story.read()

  import re                                

  def clean_text(text):
      text = re.sub(r',', '', text)
      text = re.sub(r'\'', '',  text)
      text = re.sub(r'\"', '', text)
      text = re.sub(r'\(', '', text)
      text = re.sub(r'\)', '', text)
      text = re.sub(r'\n', '', text)
      text = re.sub(r'“', '', text)
      text = re.sub(r'”', '', text)
      text = re.sub(r'’', '', text)
      text = re.sub(r'\.', '', text)
      text = re.sub(r';', '', text)
      text = re.sub(r':', '', text)
      text = re.sub(r'\-', '', text)

      return text

  # cleaning the data
  lower_data = story_data.lower()           

  split_data = lower_data.splitlines()      

  final = ''                                

  for line in split_data:
      line = clean_text(line)
      final += '\n' + line

  #print(final)

  final_data = final.split('\n')       
  #print(final_data)

  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences

  # Instantiating the Tokenizer
  max_vocab = 1000000
  tokenizer = Tokenizer(num_words=max_vocab)
  tokenizer.fit_on_texts(final_data)

  # Getting the total number of words of the data.
  word2idx = tokenizer.word_index
  #print(len(word2idx))
  #print(word2idx)
  vocab_size = len(word2idx) + 1       
  #print(vocab_size)
  #n-grams
  #The sentence ['my name is shiva ji and happy'] will have sequence as [112, 113, 114, 7, 5, 190, 75]
  # * [112, 113], 
  # * [112, 113, 114], 
  # * [112, 113, 114, 7], 
  # * [112, 113, 114, 7, 5], 
  # * [112, 113, 114, 7, 5, 190],
  # * [112, 113, 114, 7, 5, 190, 75], 

  input_seq = []

  for line in final_data:
      token_list = tokenizer.texts_to_sequences([line])[0]
      for i in range(1, len(token_list)):
          n_gram_seq = token_list[:i+1]
          input_seq.append(n_gram_seq)

  #print(input_seq)

  # Getting the maximum length of sequence for padding purpose
  max_seq_length = max(len(x) for x in input_seq)
  #print(max_seq_length)

  # Padding the sequences and converting them to array
  input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_length, padding='pre'))
  #print(input_seq)

  # Taking xs and labels to train the model.

  xs = input_seq[:, :-1]        
  labels = input_seq[:, -1]     

  from tensorflow.keras.utils import to_categorical
  ys = to_categorical(labels, num_classes=vocab_size)


  model = load_model(hd5_file)
  def predict_words(seed, no_words):
    for i in range(no_words):
      token_list = tokenizer.texts_to_sequences([seed])[0]
      token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
      predicted = np.argmax(model.predict(token_list), axis=1)

      new_word = ''

      for word, index in tokenizer.word_index.items():
        if predicted == index:
          new_word = word
          break
      seed += " " + new_word
    #print(seed)
    return (seed)

  seed_text = txt
  next_words = 5
  x = predict_words(seed_text, next_words)
  return (x)


def sentiment_analysis(inp_text):
    import numpy as np 
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    import spacy 
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    import nltk
    from nltk.tokenize.toktok import ToktokTokenizer
    import sqlite3
    import os
    import re
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    import unicodedata
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

    def get_sentiment(text):
        normalized_corpus = normalize_corpus(text)
        #print(normalized_corpus)
        sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in normalized_corpus]
        sentiment_category_tb = ['positive' if score > 0 
                                    else 'negative' if score < 0 
                                        else 'neutral' 
                                            for score in sentiment_scores_tb]

        return sentiment_category_tb
    return (get_sentiment(inp_text))





def text_generation(txt):
  
  x = sub_text_generation(txt,'./alice_epoch=56_accuracy=97.txt','./generator.h5')
  print('\n First sentence for given input : ')
  
  print(x)
  print()
  ln = input('Please enter the language to which the text is to be translated :  ' )
  print('translating to {} ...'.format(ln))
  print()
  lis = []
  lis.append(x)
  lis.append('en')
  print(text_translate_1(lis,ln))

  
  y = sub_text_generation(txt,'./harry_potter_epoch=60_accuracy=96.txt','./generator_2.h5')
  print('\n Second sentence for given input : ')
  print()
  print(y)
  print('translating to {} ...'.format(ln))
  print()
  lis = []
  lis.append(y)
  lis.append('en')
  print(text_translate_1(lis,ln))


  #print('\n Third sentence for given input : ')
  z = sub_text_generation(txt,'./Twilight_epoch=57_accuracy=98.txt','./generator_3.h5')
  print('\n Third sentence for given input : ')
  print()
  print(z)
  print('translating to {} ...'.format(ln))
  print()
  lis = []
  lis.append(z)
  lis.append('en')
  print(text_translate_1(lis,ln))
  tg = []
  tg.append(x)
  tg.append(y)
  tg.append(z)
  return tg


def not_final_module():
  #print('not final module')
  sp1 = speech()
  if sp1 != 0:
      z = sp1     ##['re-again input +input of the language',''hi']  shivaji
      return z[0]
  else:
      print('\n \n not_final_module::: please try again !! may be there is some external noise')  
  

def final_module():
  sp = speech()
  if sp != 0:
      y = sp     ##[''input of the language',''hi'] my name is
  else:
      print('\n \n please try again !! may be there is some external noise')

  try:
      txt1 = ((text_translate(y,0)).lower())
      
  except:
      print('Please try again!!!')


  res = (text_generation(txt1))
  k = 0
  comb = ['1','2','3','4','5']
  while(1):

    x = int(input('Please enter ur choice 1 or 2 or 3 or 4 or 5 for break: '))
    if x == 1 or x == 2 or x==3:
      #res = (text_generation(txt2))
      print()
      print('final output....')
      print(res[x-1])
      print('Sentiment analysis: ---  ')
      print(sentiment_analysis(res[x-1]))
      break
      
    elif x == 5:
      print()
      print('final output....')
      print(txt2)
      print('Sentiment analysis: ---  ')
      print(sentiment_analysis(txt2))
      break
    else:
      comb[k] = not_final_module()
      if k == 0:
        comb[k] = y[0] + ' ' + comb[k]
      else:
        comb[k] = comb[k-1] + ' ' + comb[k]
      txt2 = ((text_translate(comb[k],y[1])).lower())
      res = (text_generation(txt2))
      print()
      print(res)
      k+=1


final_module()
