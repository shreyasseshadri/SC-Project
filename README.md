# SC-Project
Project for Soft Computing Course

[Link to paper to be implemented](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8314136)

Datasets:
-  STANFORD LARGE MOVIE REVIEW DATASET (IMDB)
    - [Kaggle link](https://www.kaggle.com/c/sentiment-classification-on-large-movie-review)
    - [Standford Link](http://ai.stanford.edu/~amaas/data/sentiment/)
  
- STANFORD SENTIMENT TREEBANK DATASET (SSTb)
  - [Code for generating dataset](https://github.com/JonathanRaiman/pytreebank)

# Instructions
- Create a seperate directory under root directory called 'data'.It is added in .gitignore.This directory will be used to       store all data from now on so it wont be pushed to the repo.
- Download the standford link dataset(imdb) and place it under data directory
- Go through the README file given by them.
- Run make_csv.py as follows
```
python3 make_csv.py <path to directory containing .txt files> <path where u want .csv file to be stored>
```

eg:
```
python3 make_csv.py ./data/imdb/test/pos ./data/imdb/test/pos.csv
```