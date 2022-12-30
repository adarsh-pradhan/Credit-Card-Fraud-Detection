# Importing Libraries

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
"""
# Loading dataset
data = pd.read_csv(r"fraudTrain.csv")

testing = data[:10000]

useful_columns = [
  'category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat',
  'merch_long', 'trans_date_trans_time', 'dob', 'is_fraud'
]
sample = data[useful_columns].copy()

# Converting DOB into age

sample['age'] = dt.date.today().year - pd.to_datetime(sample['dob']).dt.year
sample['hour'] = pd.to_datetime(sample['trans_date_trans_time']).dt.hour
sample['day'] = pd.to_datetime(sample['trans_date_trans_time']).dt.dayofweek
sample['month'] = pd.to_datetime(sample['trans_date_trans_time']).dt.month
sample.pop('trans_date_trans_time')
sample.pop('dob')

y = sample.pop('is_fraud')

# Converting categorical data into dummy variables
X = pd.get_dummies(sample, drop_first=True)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

new_X_train, new_y_train = SMOTE().fit_resample(X_train, y_train)
new_X_test, new_y_test = SMOTE().fit_resample(X_test, y_test)

# Model fitting

'''rfc = RandomForestClassifier(random_state=42)
rfc.fit(new_X_train, new_y_train)
print(classification_report(new_y_test, rfc.predict(new_X_test)))
rfc_final=RandomForestClassifier(random_state=42)
rfc_final.fit(new_X,new_y)'''
filename = "rfc_model_4.joblib"
joblib.dump(rfc, filename)"""

loaded_model = joblib.load("rfc_model_4.joblib")

# Saving the model to a pickle file
# pickle.dump(rfc, open('model.pkl', 'wb'))

# A function for taking user input from the web-app


def user_input(category, amt, zip, lat, long, city_pop, merch_lat, merch_long,
               age, hour, day, month):
  inp = [amt, zip, lat, long, city_pop, merch_lat, merch_long, age, hour, day, month]
  arr = ['food_dining', 'gas_transport', 'grocery_net', 'grocery_pos','health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos','personal_care', 'shopping_net', 'shopping_pos', 'travel']
  for i in range(len(arr)):
    if arr[i] == category:
      inp.append(1)
    else:
      inp.append(0)
  inp = np.array(inp)
  return loaded_model.predict(inp.reshape(1, -1))


print(
  user_input('grocery_pos', 281.06, 28611, 35.9946, -81.7266, 885, 36.430124,81.179483, 92, 1, 4, 2))
