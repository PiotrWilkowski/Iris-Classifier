import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

column_names=["sepal_length","sepal_width","petal_length", "petal_width", "class"]
iris_data=pd.read_csv("iris.csv", names=column_names)
print(iris_data.head())
print(iris_data.describe())

sns.pairplot(iris_data, hue="class")
plt.show()

X=iris_data.drop("class", axis=1)
y=iris_data["class"]

knn=KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3, random_state=42)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

new_data=pd.DataFrame({"sepal_length":[6],"sepal_width":[3],"petal_length":[1],"petal_width":[2]})
prediction=knn.predict(new_data)
print(prediction[0])