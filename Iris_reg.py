# Import Dataset from sklearn
from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
x= iris_df.drop(labels= 'sepal length (cm)', axis= 1)
y= iris_df['sepal length (cm)']

# Splitting the Dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 20)

# Instantiating LinearRegression() Model
lr = LinearRegression()

# Training/Fitting the Model
lr.fit(x_train, y_train)

if st.button('Predict'):    
    # Making Predictions
    lr.predict(x_test)
    y_pred = lr.predict(x_test)
    
    # MAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.metrics import r2_score
    
    MAE =  mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    # R-Squared
    r_squared = r2_score(y_test, y_pred)

    # Adjusted R-Squared
    n = len(y)
    p = x.shape[1]
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    st.write(f"The Mean Absolute Error is, **{MAE}**")
    st.write(f"The R-Squared is, **{r_squared}**")
    st.write(f"The Adjusted R-Squared is, **{adjusted_r_squared}**")
