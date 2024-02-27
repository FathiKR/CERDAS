from joblib import dump
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

#load data 
iris = datasets.load_iris(return_X_y=True)

#split features & target
X = iris[0]
y = iris[1]


#create a pipeline
clf_pipeline = [('scaling', MinMaxScaler()), 
                ('clf', DecisionTreeClassifier(random_state=42))]

pipeline = Pipeline(clf_pipeline)


model = pipeline.fit(X, y)
score = model.predict([[3,1,1.1,4]])
prob = model.predict_log_proba([[3,1,1.1,4]])

print(score)
print(prob)
#save to joblib file
# dump(pipeline, './iris_model_base.joblib')