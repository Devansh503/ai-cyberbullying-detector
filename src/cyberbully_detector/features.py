from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def build_tfidf_and_labels(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
    X_train = vect.fit_transform(df_train["text_clean"]) 
    X_val = vect.transform(df_val["text_clean"]) 
    X_test = vect.transform(df_test["text_clean"]) 
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["category"]) 
    y_val = le.transform(df_val["category"]) 
    y_test = le.transform(df_test["category"]) 
    return vect, le, X_train, y_train, X_val, y_val, X_test, y_test
