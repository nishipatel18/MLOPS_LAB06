import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def titanic_dataset():
    df = sns.load_dataset('titanic')
    df.drop(['class','who','adult_male','deck','embark_town','alive','alone'], axis=1, inplace=True)
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['sex'] = df['sex'].map({'male':1,'female':0})
    df['embarked'] = LabelEncoder().fit_transform(df['embarked'])
    X = df.drop('survived', axis=1)
    y = df['survived']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_and_compile_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy')
    return model
