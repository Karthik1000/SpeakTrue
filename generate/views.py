from django.shortcuts import render
from textblob import TextBlob
import speech_recognition as sr
# import langdetect.detector
from langdetect import detect
from django.http import HttpResponse

from googletrans import Translator
import googletrans


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

import re                                

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

import os
from google.cloud import translate_v2 as translate

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"E:\7th_sem\NLP\Project_working_area\django_implementation\text-generate-main\generate\hip-heading-283511-c145f68ba57b.json"
#from googletrans import Translator
#import googletrans
######### sentimental
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


#############
def text_translate(txt,lang):
    k=googletrans.LANGUAGES
    lng2 = lang
    for key, value in k.items():
        if lng2 == value:
            y=key



    translate_client = translate.Client()

    text = txt
    target = y

    output = translate_client.translate(
        text,
        target_language=target
    )

    return (output['translatedText'])



# Create your views here.
def home(request):
    return render(request,'generate/home.html')

def speech():
    # import speech_recognition as sr       ### Using HMM model as API
    # from langdetect import detect  
    # from textblob import TextBlob

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


# def text_translate(inp,ln):
#     # from googletrans import Translator
#     # import googletrans
#     k=googletrans.LANGUAGES
#     if len(inp) == 2:
#       lng1 = inp[1]
#     else:
#       lng1 = ln
    
#     for key, value in k.items():
#         if lng1 == key:
#             x=key
#     lng2 = 'english'
#     for key, value in k.items():
#         if lng2 == value:
#             y=key
#     if len(inp) == 2:
#       text1=inp[0]
#     else:
#       text1 = inp
#     translator = Translator()
#     translated = translator.translate(text1, src=x, dest=y)
#     #print("The translated text in",lng2,"is : ",translated.text.lower())
#     return ((translated.text)).lower()

# def text_translate_1(inp,ln):
#     # from googletrans import Translator
#     # import googletrans

#     k=googletrans.LANGUAGES
#     lng1 = inp[1]
#     for key, value in k.items():
#         if lng1 == key:
#             x=key
#     lng2 = ln
#     for key, value in k.items():
#         if lng2 == value:
#             y=key
#     text1=inp[0]
#     translator = Translator()
#     translated = translator.translate(text1, src=x, dest=y)
#     #print("The translated text in",lng2,"is : ",translated.text.lower())
#     return ((translated.text)).lower()
# def text_translate_2(inp,ln):
#     # from googletrans import Translator
#     # import googletrans

#     k=googletrans.LANGUAGES
#     lng1 = inp[1]
#     for key, value in k.items():
#         if lng1 == key:
#             x=key
#     lng2 = ln
#     for key, value in k.items():
#         if lng2 == value:
#             y=key
#     text1=inp[0]
#     translator = Translator()
#     translated = translator.translate(text1, src=x, dest=y)
#     #print("The translated text in",lng2,"is : ",translated.text.lower())
#     return ((translated.text)).lower()
# def text_translate_3(inp,ln):
#     # from googletrans import Translator
#     # import googletrans

#     k=googletrans.LANGUAGES
#     lng1 = inp[1]
#     for key, value in k.items():
#         if lng1 == key:
#             x=key
#     lng2 = ln
#     for key, value in k.items():
#         if lng2 == value:
#             y=key
#     text1=inp[0]
#     translator = Translator()
#     translated = translator.translate(text1, src=x, dest=y)
#     #print("The translated text in",lng2,"is : ",translated.text.lower())
#     return ((translated.text)).lower()


def sub_text_generation(txt,data_set,hd5_file):
  # import tensorflow as tf
  # import pandas as pd
  # import numpy as np
  # import re                               
  # from tensorflow.keras.preprocessing.text import Tokenizer
  # from tensorflow.keras.preprocessing.sequence import pad_sequences
  # from tensorflow.keras.utils import to_categorical
  # from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D
  # from tensorflow.keras.models import Model
  # from tensorflow.keras.optimizers import Adam
  # from tensorflow.keras.models import Sequential


  # from keras.models import load_model

  with open(data_set, encoding="utf8") as story:
      story_data = story.read()

  # import re                                

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

  # from tensorflow.keras.preprocessing.text import Tokenizer
  # from tensorflow.keras.preprocessing.sequence import pad_sequences

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

  # from tensorflow.keras.utils import to_categorical
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
  sentence = inp_text       #########################input text
  sentiment = get_sentiment(sentence)
  lis = []
  print('Sentiment of your sentence is ',sentiment[0])
  lis.append(sentiment[0])          ###negative or positive
  sentence_swear_removed = remove_swear(sentence)
  print('Your text after removing swa',sentence_swear_removed)
  lis.append(sentence_swear_removed)    ## sentence
  return lis

  #sentence = remove_special_char(sentence)





def text_generation(txt,lang):
  
  x = sub_text_generation(txt,'generate/alice_epoch=56_accuracy=97.txt','generate/generator.h5')
  print('\n First sentence for given input : ')
  
  print(x)
  print()
  #ln = input('Please enter the language to which the text is to be translated :  ' )
  ln = lang
  print('translating to {} ...'.format(ln))
  print()
  lis = []
  lis.append(x)
  lis.append('en')
  print(text_translate_1(lis,ln))

  
  y = sub_text_generation(txt,'generate/harry_potter_epoch=60_accuracy=96.txt','generate/generator_2.h5')
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
  z = sub_text_generation(txt,'generate/Twilight_epoch=57_accuracy=98.txt','generate/generator_3.h5')
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
  

def record(request):
  sp = speech()
  if sp != 0:
      y = sp     ##[''input of the language',''hi'] my name is
      #return y
      k=googletrans.LANGUAGES
      lng2 = y[1]
      for key, value in k.items():
          if lng2 == key:
              z=value

      msg = y[0]
      language = z
  else:
      print('\n \n please try again !! may be there is some external noise')
      msg = 'Please try again'
      language = 'Not Recognized'
  return render(request,'generate/record.html',{'msg':msg,'lang':language})
def translation(request):  
      #y = record()
  if request.method == 'POST':
    a = request.POST['message']
    b = request.POST['language']
    print('hello')
    a = str(a)
    b = 'english'
    print(a,b)
    # y = []

    # y.append(a)
    # y.append(b)
    # print(y)
    txt1 = ((text_translate(a,b)))
    print(txt1)
    return render(request,'generate/trans.html',{'msg':txt1})

      
      #text1 = 'please try again!!'
  #return render(request,'generate/trans.html',{'msg':text1})
def text_gen(request):
      #y = record()
  if request.method == 'POST':
    a = request.POST['message']
    b = request.POST['language']
    print('hello12')
    a = str(a)
    b = str(b)
    print(a,b)
    #y = []

    #y.append(a)
    #y.append(b)
    #print(y)
    #res = (text_generation(a,b))
    txt = a
    lang = b
    x = sub_text_generation(txt,'generate/alice_epoch=56_accuracy=97.txt','generate/generator.h5')
    print('\n First sentence for given input : ')
    
    print(x)
    print()
    #ln = input('Please enter the language to which the text is to be translated :  ' )
    ln = lang
    print('translating to {} ...'.format(ln))
    print()
    # lis_1 = []
    # lis_1.append(x)
    # lis_1.append('en')
    #print(text_translate_1(lis,ln))
    res_0 = text_translate(x,ln)
    print(res_0)

    
    y = sub_text_generation(txt,'generate/harry_potter_epoch=60_accuracy=96.txt','generate/generator_2.h5')
    print('\n Second sentence for given input : ')
    print()
    print(y)
    print('translating to {} ...'.format(ln))
    print()
    # lis_2 = []
    # lis_2.append(y)
    # lis_2.append('en')
    #print(text_translate_1(lis,ln))
    res_1 = text_translate(y,ln)
    print(res_1)


    #print('\n Third sentence for given input : ')
    z = sub_text_generation(txt,'generate/Twilight_epoch=57_accuracy=98.txt','generate/generator_3.h5')
    print('\n Third sentence for given input : ')
    print()
    print(z)
    print('translating to {} ...'.format(ln))
    print()
    # lis_3 = []
    # lis_3.append(z)
    # lis_3.append('en')
    #print(text_translate_1(lis,ln))
    res_2 = text_translate(z,ln)
    print(res_2)
    #if (res_0!=None and res_1!=None and res_2!=None):

    #print(res)
    msg = a

    return render(request,'generate/gen.html',{'msg1':res_0,'msg2':res_1,'msg3':res_2,'msg4':msg})
    # else:
    #   return render(request,'generate/trans.html',{'msg':a})

def record_1(request):
  print('record_1......')
  if request.method == 'POST':
    a = request.POST['message']

    sp = speech()
    if sp != 0:
        y = sp     ##[''input of the language',''hi'] my name is
        #return y
        print('record_1...')
        k=googletrans.LANGUAGES
        lng2 = y[1]
        for key, value in k.items():
            if lng2 == key:
                z=value

        # lis = []
        # lis.append(a)
        # lis.append(y[1])
        print(z)
        print(a)
        #txt1 = ((text_translate(a,z)))
        #print(txt1)
        msg = a + ' ' + y[0]
        print(msg)
        language = z
    else:
        print('\n \n please try again !! may be there is some external noise')
        msg = 'Please try again'
        language = 'Not Recognized'
  return render(request,'generate/record.html',{'msg':msg,'lang':language})


def final(request): 
  print('final_msg......')
  if request.method == 'POST':
    a = request.POST['final_msg']
    z = 'english'
    print(a,z)
    txt1 = text_translate(a,z)
    print(txt1)
    msg = txt1
  else:
      print('\n \n please try again !! did not accept the request')
      msg = 'Please try again!!'
      #language = ''
  return render(request,'generate/final.html',{'msg':msg})


def sentimental(request):
  print('Sentimental......')
  if request.method == 'POST':
    a = request.POST['final_senti']
    a = str(a)
    output = sentiment_analysis(a)
    msg = output[1]
    status = output[0]
  else:
      print('\n \n please try again !! We did not get correct input for sentimental analysis')
      msg = 'Please try again'
      status = 'Neutral'
  return render(request,'generate/sentimental.html',{'msg':msg,'status':status})





  







def func():
  res = (text_generation(txt1))
  print('helloooooo')
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
      return render(request,'generate/record.html',{'msg':res[x-1],'lang':sentiment_analysis(res[x-1])})
      break
      
    elif x == 5:
      print()
      print('final output....')
      print(txt2)
      print('Sentiment analysis: ---  ')
      print(sentiment_analysis(txt2))
      return render(request,'generate/record.html',{'msg':txt2,'lang':sentiment_analysis(txt2)})
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
      language = 'final_output'
      k+=1
      return render(request,'generate/record.html',{'msg':res,'lang':language})



# # def record(request):
# #     r1 = sr.Recognizer()
# #     with sr.Microphone() as source:
# #         print("Say something...")
# #         audio_data = r1.listen(source, timeout=7)
# #         print("Recognizing...")
# #         try:
# #             li = []
# #             x = (r1.recognize_google(audio_data))
# #             print(x)
# #             z = (TextBlob(x))
# #             lang = (z.detect_language())
# #             x = (r1.recognize_google(audio_data, language=lang))
# #             li.append(x)
# #             y = detect(x)
# #             li.append(y)
# #             print(li)
# #             msg = x
# #             language = y
# #         except:
# #             print('Sorry')
# #             msg = 'Please try again'
# #             language = 'Not Recognized'
# #     return render(request,'generate/record.html',{'msg':msg,'lang':language})