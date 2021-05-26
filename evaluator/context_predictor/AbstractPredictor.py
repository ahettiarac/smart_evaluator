import numpy as np
import pandas as pd
import re
import io
import json
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer,tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
from sklearn.model_selection import train_test_split
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

class AbstractPredictor:

    data = pd.DataFrame(index=[0], columns=['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary','Text'])
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

    max_text_length = 30
    max_summary_length = 8
    vocab_size = 0
    x_vocab_size = 0
    reverse_target_word_index = None
    reverse_source_word_index = None 
    stop_words = set(stopwords.words('english'))                      

    def __init__(self, csvPath=''):
        if csvPath != '':
            self.data = pd.read_csv(csvPath, nrows=100000)
            self.data.drop_duplicates(subset=['Text'],inplace=True)
            self.data.dropna(axis=0, inplace=True)
            self.data.info()

    def preprocess_text(self, column_name):
        preprocessed_text = []
        for text in self.data[column_name]:
            cleaned_text = text.lower() # convert to lower case text
            cleaned_text = BeautifulSoup(cleaned_text, "lxml").text # remove html tags from text
            cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)
            cleaned_text = re.sub('"','',cleaned_text)
            cleaned_text = ' '.join([self.contraction_mapping[word] if word in self.contraction_mapping else word for word in cleaned_text.split(" ")]) #remove contractions
            cleaned_text = re.sub(r"'s\b","",cleaned_text) #removing 's from text
            cleaned_text = re.sub("[^a-zA-Z]"," ", cleaned_text) # remove words in paranthesis
            cleaned_text = re.sub('[m]{2,}','mm', cleaned_text) # remove punctuations and special characters
            tokens = [word for word in cleaned_text.split() if not word in self.stop_words]
            long_words = []
            for token in tokens:
                if len(token) > 1:
                    long_words.append(token) # removing short words
            preprocessed_text.append(" ".join(long_words).strip())
        return preprocessed_text,long_words

    def pre_arrange_text_data(self,data):
        preprocessed_text = np.array(data['cleaned_text'])
        preprocessed_summary = np.array(data['cleaned_summary'])
        short_text = []
        short_summary = []
        for i in range(len(preprocessed_text)):
            if(len(preprocessed_summary[i].split()) <= self.max_summary_length and len(preprocessed_text[i].split()) <= self.max_text_length):
                short_text.append(preprocessed_text[i])
                short_summary.append(preprocessed_summary[i])
        df = pd.DataFrame({'text': short_text, 'summary': short_summary})
        df['summary'] = df['summary'].apply(lambda x: 'sostock '+x+' eostock')
        return train_test_split(np.array(df['text']), np.array(df['summary']), test_size=0.1,random_state=0,shuffle=True)

    def arrange_model_data(self, x_train, x_test, y_train, y_test):
        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_train))
        vocab_size,rare_words_count = self.find_rare_word_count(x_tokenizer)
        x_tokenizer = Tokenizer(num_words=(vocab_size-rare_words_count))
        x_tokenizer.fit_on_texts(list(x_train))
        x_train_seq = x_tokenizer.texts_to_sequences(x_train)
        x_test_seq = x_tokenizer.texts_to_sequences(x_test)
        x_train = pad_sequences(x_train_seq,maxlen=self.max_text_length,padding='post')
        x_test = pad_sequences(x_test_seq,maxlen=self.max_text_length,padding='post')
        self.x_vocab_size = x_tokenizer.num_words + 1
        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(list(y_train))
        vocab_size,rare_words_count = self.find_rare_word_count(y_tokenizer)
        y_tokenizer = Tokenizer(num_words=(vocab_size-rare_words_count))
        y_tokenizer.fit_on_texts(list(y_train))
        y_train_seq = y_tokenizer.texts_to_sequences(y_train)
        y_test_seq = y_tokenizer.texts_to_sequences(y_test)
        y_train = pad_sequences(y_train_seq,maxlen=self.max_summary_length,padding='post')
        y_test = pad_sequences(y_test_seq,maxlen=self.max_summary_length,padding='post')
        self.vocab_size = y_tokenizer.num_words + 1
        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        x_tokenizer_json = x_tokenizer.to_json()
        with io.open('x_tokenizer.json','w',encoding='utf-8') as f:
            f.write(json.dumps(x_tokenizer_json, ensure_ascii=False))
        y_tokenizer_json = y_tokenizer.to_json()
        with io.open('y_tokenizer.json','w',encoding='utf-8') as f:
            f.write(json.dumps(y_tokenizer_json, ensure_ascii=False))    
        return x_train,x_test,y_train,y_test

    def load_reverse_index(self,x_tokenizer_path,y_tokenizer_path):
        x_tokenizer = None
        y_tokenizer = None
        with open(y_tokenizer_path) as f:
            data = json.load(f)
            y_tokenizer = tokenizer_from_json(data)
            self.reverse_target_word_index = y_tokenizer.index_word
        with open(x_tokenizer_path) as f:
            data = json.load(f)
            x_tokenizer = tokenizer_from_json(data)
            self.reverse_source_word_index = x_tokenizer.index_word
        return x_tokenizer, y_tokenizer                   

    def find_rare_word_count(self,tokenizer):
        threshold = 4
        vocab_size = 0
        rare_words_count = 0
        for key,value in tokenizer.word_counts.items():
            vocab_size = vocab_size+1
            if(value < threshold):
              rare_words_count=rare_words_count+1  
        return vocab_size,rare_words_count 

    def remove_empty_rows(self,x_train,x_test,y_train,y_test):
        index=[]
        for i in range(len(y_train)):
            count=0
            for k in y_train[i]:
                if k!=0:
                    count = count + 1
            if count==2:
                index.append(i)

        y_train=np.delete(y_train, index,axis=0)
        x_train=np.delete(x_train, index, axis=0)

        index = []
        for i in range(len(y_test)):
            count=0
            for k in y_test[i]:
                if k!=0:
                    count = count + 1
            if count==2:
                index.append(i)

        y_test=np.delete(y_test, index, axis=0)
        x_test=np.delete(x_test, index, axis=0)
        return x_train,x_test,y_train,y_test

    def seq2summary(self,input_seq):
        sentence = ''
        for word in input_seq:
            if ((word != 0 and word != self.reverse_source_word_index['sostock']) and word != self.reverse_source_word_index['eostock']):
                sentence = sentence + self.reverse_source_word_index[word] + ' '
        return sentence

    def seq2text(self,input_seq):
        sentence = ''
        for word in input_seq:
            if (word != 0):
                sentence = sentence + self.reverse_target_word_index[word] + ' '
        return sentence      

    def get_input_seq(self,paragraph,x_tokenizer_path,y_tokenizer_path):
        preprocessed_text = ""
        paragraph = paragraph.lower() # convert to lower case text
        cleaned_text = BeautifulSoup(paragraph, "lxml").text # remove html tags from text
        paragraph = re.sub(r'\([^)]*\)', '', cleaned_text)
        paragraph = re.sub('"','',paragraph)
        paragraph = ' '.join([self.contraction_mapping[word] if word in self.contraction_mapping else word for word in paragraph.split(" ")]) #remove contractions
        paragraph = re.sub(r"'s\b","",paragraph) #removing 's from text
        paragraph = re.sub("[^a-zA-Z]"," ", paragraph) # remove words in paranthesis
        paragraph = re.sub('[m]{2,}','mm', paragraph) # remove punctuations and special characters
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in paragraph.split() if not word in stop_words]
        long_words = []
        for token in tokens:
            if len(token) > 1:
                long_words.append(token) # removing short words
        preprocessed_text = " ".join(long_words).strip()
        short_text_list = []
        x_tokenizer,y_tokenizer = self.load_reverse_index(x_tokenizer_path,y_tokenizer_path)
        short_text_list.append(preprocessed_text)
        print("short text length: {}".format(len(short_text_list)))
        x_seq = x_tokenizer.texts_to_sequences(short_text_list)
        x_seq = pad_sequences(x_seq, maxlen=self.max_text_length,padding='post')
        return x_seq,y_tokenizer,x_tokenizer    



