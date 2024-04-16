# Import Dataset from sklearn
from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# st.set_page_config(page_title="Iris Data Analysis", layout="wide")

# Page Layout
c1, c2 = st.columns([0.3,0.7])

with c2:
    st.markdown("## Iris Data Analysis")

# Load Iris Data
iris = load_iris()

# Creating pd DataFrames
iris_df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
target_df = pd.DataFrame(data= iris.target, columns= ['species'])

def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'
        
target_df['species'] = target_df['species'].apply(converter)

# Concatenate the DataFrames
iris_df = pd.concat([iris_df, target_df], axis= 1)

# Page Layout
c1, c2 = st.columns([0.3,0.7])
with c1:
    st.write('')	

########### Histogram #####################################
st.markdown("### Distribution of Variables")

c1, c2, c3 = st.columns([0.33,0.33,0.33])
with c2:
    fig, axes = plt.subplots(2, 2, figsize=(10,10)) 
      
    axes[0,0].set_title("Sepal Length") 
    axes[0,0].hist(iris_df['sepal length (cm)'], bins=7) 
      
    axes[0,1].set_title("Sepal Width") 
    axes[0,1].hist(iris_df['sepal width (cm)'], bins=5) 
      
    axes[1,0].set_title("Petal Length") 
    axes[1,0].hist(iris_df['petal length (cm)'], bins=6)
      
    axes[1,1].set_title("Petal Width") 
    axes[1,1].hist(iris_df['petal width (cm)'], bins=6)
    
    st.pyplot(fig)

# pairplot of independent variables
st.markdown("### Relationship between Variables")
c1, c2, c3 = st.columns([0.33,0.33,0.33])

with c2:
    ax = sns.pairplot(iris_df, hue= 'species')
    st.pyplot(ax.figure)
    
    # Model
    # Converting Objects to Numerical dtype
    iris_df.drop('species', axis= 1, inplace= True)
    target_df = pd.DataFrame(columns= ['species'], data= iris.target)
    iris_df = pd.concat([iris_df, target_df], axis= 1)

# Page Layout
c1, c2 = st.columns([0.3,0.7])
with c1:
    st.write('')

st.markdown("##### Data")
st.dataframe(iris_df.head())

# Variables
x= iris_df.drop(labels= 'species', axis= 1)
y= iris_df['species']

# Splitting the Dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 20)

# Random Forest Model for Model building
RF = RandomForestClassifier()
    
# Model 
model = RF.fit(x_train, y_train)
    
# Prediction
y_pred = model.predict(x_test)
        
# Accuracy, ROC
from sklearn.metrics import roc_auc_score
# acc_rf = model.score(x_test, y_test)
# roc_rf = roc_auc_score(y_test, y_pred)

# Confusion matrix
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create DataFrame for confusion matrix
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted: Setosa', 'Predicted: Versicolor', 'Predicted: Virginica'],
                           index=['Actual: Setosa', 'Actual: Versicolor', 'Actual: Virginica'])

c1, c2, c3 = st.columns([0.33,0.33,0.33])
with c2:
    # Plot confusion matrix
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
    plt.title('Confusion Matrix for Iris Dataset')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot(plt)

# Metrics Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

c1, c2, c3 = st.columns([0.33,0.33,0.33])
with c2:
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size
    bars = ax.bar(['Accuracy', 'Precision', 'Recall'], [accuracy, precision, recall], color=['blue', 'green', 'red'])  # Different colors for bars
    
    for bar in bars:
        bar.set_width(0.5)
    
    st.pyplot(fig)

def class_flower(i):
    if i == 0:
        return 'The Flower is likely a Setosa'
    if i == 1:
        return 'The flower is likely a Versicolor'
    if i == 2:
        return 'The Flower is likely a Virginica'
    

if st.button('Predict'):    
    # Making Predictions
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.rename(columns={0: 'value'}, inplace=True)
    
    y_pred['value'] = y_pred['value'].apply(class_flower)

    st.dataframe(y_pred)
