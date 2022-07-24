import numpy as np 
import pandas as pd 
from math import  sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.preprocessing import minmax_scale 
import shap
import  time 
t = time.time()

data = pd.read_csv("Multi_Features_Multi_Band_gapsData.csv", header = 0,  delim_whitespace=False, index_col=None)  
data = pd.DataFrame(data) #The train data

data_X = minmax_scale(data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]])
data_X = pd.DataFrame(data_X)
data_y = data.iloc[:,[13]]

#Data Split
#print(data.iloc[::])
train_X=data_X.sample(frac =0.95,random_state=100) #five splits seeds: 20, 40, 60, 80, 100
test_X=data_X.drop(train_X.index)
test_X = test_X.reset_index(drop=True)
train_X= train_X.reset_index(drop=True)


train_y=data_y.sample(frac=0.95,random_state=100) #five splits seeds: 20, 40, 60, 80, 100
test_y=data_y.drop(train_y.index)
test_y = test_y.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)

dummy_trainX = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
dummy_trainX =dummy_trainX.sample(frac =0.95,random_state=100)
dummy_trainX = dummy_trainX.reset_index(drop=True)


# fitting a model (training the XGBoost)
model = xgb.XGBRegressor(objective='reg:squarederror').fit(train_X, train_y)


# Generate the Tree explainer and SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_X)
expected_value = explainer.expected_value

############## visualizations #############
# Generate summary dot plot
shap.summary_plot(shap_values, train_X,title="SHAP summary plot") 

# Generate summary bar plot 
shap.summary_plot(shap_values, train_X,plot_type="bar") 