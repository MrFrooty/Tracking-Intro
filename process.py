import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('coords.csv', skipinitialspace=True)

#convert the strings to an int value: 0 represents Happy and 1 represents Sad
label_encoder = LabelEncoder()
df["class_encoded"] = label_encoder.fit_transform(df["class"])
df.drop("class", axis=1, inplace=True)

x = df.drop('class_encoded',axis=1) #features
y = df['class_encoded'] #target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
