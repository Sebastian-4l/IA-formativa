#recommendation_system.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('products.csv')

features = data.drop(['product_id', 'product_name'], axis=1)
features2 = data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature column names

labels = data['label']  # Replace with actual label column name

X_train, X_test, y_train, y_test = train_test_split(features2, labels, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

def recommend(product_id, num_recommendations=5):
    product_index = data[data['product_id'] == product_id].index[0]
    product_features = features2.iloc[product_index].values.reshape(1, -1)
    
    similarities = cosine_similarity(product_features, features2)[0]
    
    similar_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]
    
    recommended_products = data.iloc[similar_indices][['product_id', 'product_name']]
    
    return recommended_products