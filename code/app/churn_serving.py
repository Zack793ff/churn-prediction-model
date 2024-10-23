# Create a simple Python script that loads the model and applies it to a customer.
import pickle
import numpy as np

# let's put the predict_single function
def predict_single(customer, dv, model):
  X = dv.transform([customer])
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]

# Now we can load our model:
with open('churn-model.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)


# And apply it
customer = { 
 'customerid': '8879-zkjof', 
 'gender': 'female', 
 'seniorcitizen': 0, 
 'partner': 'no', 
 'dependents': 'no', 
 'tenure': 41, 
 'phoneservice': 'yes', 
 'multiplelines': 'no', 
 'internetservice': 'dsl', 
 'onlinesecurity': 'yes', 
 'onlinebackup': 'no', 
 'deviceprotection': 'yes', 
 'techsupport': 'yes', 
 'streamingtv': 'yes', 
 'streamingmovies': 'yes', 
 'contract': 'one_year', 
 'paperlessbilling': 'yes', 
 'paymentmethod': 'bank_transfer_(automatic)', 
 'monthlycharges': 79.85, 
 'totalcharges': 3320.75, 
} 

prediction = predict_single(customer, dv, model)

print('prediction: %.3f' % prediction)

if prediction >= 0.5:
  print('verdict: Churn')
else:
  print('verdict: Not churn')
