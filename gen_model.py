import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pickle 

df = pd.read_csv('coords.csv', skipinitialspace=True)

x = df.drop('class', axis=1)  # Features
y = df['class']  # Target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

# Create pipeline objects with 4 algorithms
pipelines = {
    'lr': make_pipeline(LogisticRegression(solver='liblinear', max_iter=1000)),
    'rc': make_pipeline(RidgeClassifier()),
    'rf': make_pipeline(RandomForestClassifier()),
    'gb': make_pipeline(GradientBoostingClassifier()),
}

# Training step and storing it in the fit_models dictionary
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model

# Print accuracy for each model
for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, "accuracy score:", accuracy_score(y_test, yhat))
    
# Save the trained models using pickle
for algo, model in fit_models.items():
    filename = f'{algo}_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
