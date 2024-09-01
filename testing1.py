import mlflow
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# init dagshub
dagshub.init(repo_owner='ronylpatil', repo_name='dagshub-testing', mlflow=True)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
max_depth = 5

# apply mlflow
# mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('iris-dt-dagshub')
# adding experiment description
experiment_description = ('training decision tree - dagshub') 
mlflow.set_experiment_tag('mlflow.note.content', experiment_description)

with mlflow.start_run(description = 'Using decision tree - by ronil'):

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # mlflow tracking
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, 'decision tree')
    mlflow.set_tag('author','ronil')
    mlflow.set_tag('model','decision tree')

    print('accuracy', accuracy)
