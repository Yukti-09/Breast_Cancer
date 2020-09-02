from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = datasets.load_breast_cancer()
label_names = data['target_names']
labels = data['target']#holds 0 for malignant tumor and 1 for benign tumor
feature_names = data['feature_names']
features = data['data']
train,test,train_labels,test_labels = train_test_split(features,labels,test_size=0.2,random_state=0)

classifier = RandomForestClassifier()
model = classifier.fit(train,train_labels)
pred = classifier.predict(test)
print(accuracy_score(test_labels,pred))
