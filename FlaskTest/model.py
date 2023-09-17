import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import pickle


tfvector = TfidfVectorizer(ngram_range=(2, 3))
randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')

def set_up_model():
    plt.style.use('ggplot')
    df = pd.read_csv('all-data.csv', encoding="ISO-8859-1")
    train = df

    #Removing punctuations 
    data=train.iloc[:,1]
    data.replace("[^a-zA-Z]", " ",regex=True, inplace=True)

    # Fit and transform the 'text' column
    tfidf_matrix = tfvector.fit_transform(df['Top1'])

    # Use the 'Label' column as target labels
    labels = df['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

    # implement RandomForest Classifier
    randomclassifier.fit(X_train, y_train)
    pickle.dump(tfvector, open('tfvector.pkl','wb'))
    pickle.dump(randomclassifier, open('model.pkl','wb'))


if __name__ == "__main__":
    set_up_model()


