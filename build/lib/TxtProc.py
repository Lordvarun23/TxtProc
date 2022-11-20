import string
import pandas as pd
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


class Preprocess:
    def __init__(self, df, col_name, y_col):
        '''
        Constructor Function
        args:
          df: dataframe containing text that needs to preproc
          col_name: name of column contating text data
          y_col: name of column contating text data
        return:
          None
        '''

        self.df = df
        self.col = col_name
        self.y_col = y_col

    def listToString(self, s):
        str1 = ""
        for ele in s:
            str1 += ele
        return str1

    def remove_punctuation(self, text):
        punctuationfree = "".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    def remove_stopwords(self, text):
        stopwords = nltk.corpus.stopwords.words('english')
        output = [i for i in text if i not in stopwords]
        return output

    def tokenization(self, text):
        tokens = re.split('W+', text)
        return tokens

    def stemming(self, text):
        # defining the object for stemming
        porter_stemmer = PorterStemmer()
        stem_text = [porter_stemmer.stem(word) for word in text]
        return stem_text

    def lemmatizer(self, text):
        wordnet_lemmatizer = WordNetLemmatizer()
        lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        return lemm_text

    def preprocess(self):
        '''
        Function to preprocess text. It included removing stop words,punctuations,Lower casing,Tokenization,Stemming,Lemmatization, Bag of words(Binary)
        args:
        None
        return:
        Pandas Dataframe of binary bag of words
        '''
        data = self.df
        print(">>>>Step 1: Removing Punctuations")
        data['clean_msg'] = data[self.col].apply(lambda x: self.remove_punctuation(x))
        print(">>>>Step 1 Successfully completed")
        print(">>>>Step 2: Lower casing all the letters")
        data['msg_lower'] = data['clean_msg'].apply(lambda x: x.lower())
        print(">>>>Step 2 Successfully completed")
        print(">>>>Step 3: Tokenization")
        data['msg_tokenied'] = data['msg_lower'].apply(lambda x: self.tokenization(x))
        print(">>>>Step 3 Successfully completed")
        print(">>>>Step 4: Removing stop words")
        data['no_stopwords'] = data['msg_tokenied'].apply(lambda x: self.remove_stopwords(x))
        print(">>>>Step 4 Successfully completed")
        print(">>>>Step 5: Stemming")
        data['msg_stemmed'] = data['no_stopwords'].apply(lambda x: self.stemming(x))
        print(">>>>Step 5 Successfully completed")
        print(">>>>Step 5: Lemmetization")
        data['msg_lemmatized'] = data['no_stopwords'].apply(lambda x: self.lemmatizer(x))
        data['msg_lemmatized'] = data['no_stopwords'].apply(lambda x: self.listToString(x))
        print(">>>>Step 6 Successfully completed")
        print(">>>>Step 7: Bag of words")

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data["msg_lemmatized"])
        df_bow_sklearn = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        df_bow_sklearn[self.y_col] = data[self.y_col]

        print(">>>>Step 7 Successfully completed")
        print(">>>>All text Preprocessing completed.")

        return df_bow_sklearn

