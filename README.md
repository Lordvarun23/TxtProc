# TxtProc
TxtProc is a Python A Basic Text Pre-Processing Library. This package contains several useful functions for working with text data, including text cleaning, normalization, and feature extraction.
## Features

This package includes the following features:

    Text cleaning and normalization
    Tokenization
    Stopword removal
    Stemming and lemmatization
    Feature extraction (e.g., Bag of Words, TF-IDF)

## Installation

To install TxtProc, simply run the following command in your terminal:
  `pip install -i https://test.pypi.org/simple/ TxtProc`
  
## Usage
```
import TxtProc as tp

df = pd.read_csv("https://raw.githubusercontent.com/the-fang/Spam-mail-filtering/master/spamham.csv")
proc = tp.Preprocess(df,"Message","Category")
final = proc.preprocess()
```
### Output
```

>>>>Step 1: Removing Punctuations
>>>>Step 1 Successfully completed
>>>>Step 2: Lower casing all the letters
>>>>Step 2 Successfully completed
>>>>Step 3: Tokenization
>>>>Step 3 Successfully completed
>>>>Step 4: Removing stop words
>>>>Step 4 Successfully completed
>>>>Step 5: Stemming
>>>>Step 5 Successfully completed
>>>>Step 5: Lemmetization
>>>>Step 6 Successfully completed
>>>>Step 7: Bag of words
>>>>Step 7 Successfully completed
>>>>All text Preprocessing completed.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
