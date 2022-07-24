import numpy as np 
import pandas as pd 
from math import  sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import minmax_scale 
import  time 
t = time.time()

data = pd.read_csv("Multi_Features_Multi_Band_gapsData.csv", header = 0,  delim_whitespace=False, index_col=None)  
data = pd.DataFrame(data) #The train data

data_X = minmax_scale(data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]])
data_X = pd.DataFrame(data_X)
data_y = data.iloc[:,[13,14]]

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


# fitting a model (training the lightGBM)
model = MultiOutputRegressor(LGBMRegressor()).fit(train_X, train_y)

Train_Target = pd.DataFrame(train_y)
Train_Output  = pd.DataFrame(model.predict(train_X))
#Train_Targ_Output = pd.concat([Train_Target, Train_Output], axis=1)
Train_Targ_Output = pd.concat([Train_Target, Train_Output], axis=1)
Train_Targ_Output.to_csv('Train_lightGBM_5.csv', index=False, header=True, sep='\t')   #change  to 2, 3, 4, 5
Train_Targ_Output =  pd.read_csv("Train_lightGBM_5.csv", header = 0, delim_whitespace=True, index_col=None ) #change  to 2, 3, 4, 5
Train_Targ_Output  =   pd.DataFrame(Train_Targ_Output)
Train_Targ_Output.columns =  ["Actual_BG1",  "Actual_BG2", "Predicted_BG1",  "Predicted_BG2"]
Train_Targ_Output.to_csv('Train_lightGBM_5.csv',  header=True, sep='\t') #change  to 2, 3, 4, 5


dummy_data = pd.DataFrame(dummy_trainX)
dummy_data = dummy_data.reset_index(drop=True)
dummy_data_collect = pd.concat([dummy_data, Train_Target, Train_Output], axis=1)
dummy_data_collect.to_csv('Train_lightGBM_feature_outputs5.csv', index=False, header=True, sep='\t')
dummy_data_collect =  pd.read_csv("Train_lightGBM_feature_outputs5.csv", header = 0, delim_whitespace=True, index_col=None )
dummy_data_collect.columns = [ "Compounds", "PE_element_1",  "PE_element_2",  "PE_element_3",	 "Convalent_Radius_element_1 ",
	 "Convalent_Radius_element_2",  "Convalent_Radius_element_3",	 "FIE_element_1",  "FIE_element_2", 
     "FIE_element_3",	"Row_element_1",  "Row_element_2",  "Row_element_3",  "Direct_Band_Gap_1"	,  "Indirect_Band_Gap_2 ",
      "Predicted_BG1",  "Predicted_BG2"
]
dummy_data_collect.to_csv('Train_lightGBM_feature_outputs5.csv', index=False, header=True, sep=',')

#k =  pd.read_csv("Train_lightGBM_feature_outputs.csv", header = 0, delim_whitespace=True, index_col=None )


Target = pd.DataFrame(test_y )
Output  = pd.DataFrame(model.predict(test_X) )
Test_Targ_Output = pd.concat([Target, Output], axis=1)
Test_Targ_Output.to_csv('Test_lightGBM_5.csv', index=False, header=True, sep='\t') #change  to 2, 3, 4, 5
Test_Targ_Output =  pd.read_csv("Test_lightGBM_5.csv", header = 0, delim_whitespace=True, index_col=None ) #change  to 2, 3, 4, 5
Test_Targ_Output  =   pd.DataFrame(Test_Targ_Output)
Test_Targ_Output.columns = ["Actual_BG1",  "Actual_BG2", "Predicted_BG1",  "Predicted_BG2"]   #change  to 2, 3, 4, 5
Test_Targ_Output.to_csv('Test_lightGBM_5.csv',  header=True, sep='\t') #change  to 2, 3, 4, 5
# print('R² of  model in both training and testing  phases . . . . . . . ')
# print('train R_square', r2_score(model.predict(train_X), train_y), axis=0)
# print('test R_square', r2_score(model.predict(test_X), test_y), axis=0)
train_data_actual_and_prediction= pd.read_csv("Train_lightGBM_5.csv", header = 0,  delim_whitespace=True, index_col=None) 
print(train_data_actual_and_prediction.head())
train_actual_BG1 = train_data_actual_and_prediction.iloc[:,[0]]
train_prediction_BG1 = train_data_actual_and_prediction.iloc[:,[2]]
train_actual_BG2 = train_data_actual_and_prediction.iloc[:,[1]]
train_prediction_BG2 = train_data_actual_and_prediction.iloc[:,[3]]

test_data_actual_and_prediction= pd.read_csv("Test_lightGBM_5.csv", header = 0,  delim_whitespace=True, index_col=None) 
print(test_data_actual_and_prediction.head())
test_actual_BG1 = test_data_actual_and_prediction.iloc[:,[0]]
test_prediction_BG1 = test_data_actual_and_prediction.iloc[:,[2]]
test_actual_BG2 = test_data_actual_and_prediction.iloc[:,[1]]
test_prediction_BG2 = test_data_actual_and_prediction.iloc[:,[3]]

print('RESULT SUMMARY')

print('lightGBM-Model-BG1 for training1')
mae_lightGBM_model_BG1 = mean_absolute_error(train_actual_BG1, train_prediction_BG1)
mse_lightGBM_model_BG1 = mean_squared_error(train_actual_BG1, train_prediction_BG1)
R2_lightGBM_model_BG1  = r2_score(train_actual_BG1, train_prediction_BG1)
print('train_mae_lightGBM_model_BG1', mae_lightGBM_model_BG1)
print('train_mse_lightGBM_model_BG1', mse_lightGBM_model_BG1)
print('train_rmse_lightGBM_model_BG1', np.sqrt(mse_lightGBM_model_BG1))
print('train_R2_lightGBM_model_BG1', R2_lightGBM_model_BG1)

print('lightGBM-Model-BG1 for testing1')
mae_lightGBM_model_BG1_test = mean_absolute_error(test_actual_BG1, test_prediction_BG1)
mse_lightGBM_model_BG1_test = mean_squared_error(test_actual_BG1, test_prediction_BG1)
R2_lightGBM_model_BG1_test  = r2_score(test_actual_BG1, test_prediction_BG1)
print('test_mae_lightGBM_model_BG1', mae_lightGBM_model_BG1_test)
print('test_mse_lightGBM_model_BG1', mse_lightGBM_model_BG1_test)
print('test_rmse_lightGBM_model_BG1', np.sqrt(mse_lightGBM_model_BG1_test))
print('test_R2_lightGBM_model_BG1', R2_lightGBM_model_BG1_test)


print('lightGBM-Model-BG2 for training2')
mae_lightGBM_model_BG2 = mean_absolute_error(train_actual_BG2, train_prediction_BG2)
mse_lightGBM_model_BG2 = mean_squared_error(train_actual_BG2, train_prediction_BG2)
R2_lightGBM_model_BG2  = r2_score(train_actual_BG2, train_prediction_BG2)
print('train_mae_lightGBM_model_BG2', mae_lightGBM_model_BG2)
print('train_mse_lightGBM_model_BG2', mse_lightGBM_model_BG2)
print('train_rmse_lightGBM_model_BG2', np.sqrt(mse_lightGBM_model_BG2))
print('train_R2_lightGBM_model_BG2', R2_lightGBM_model_BG2)

print('lightGBM-Model-BG2 for testing2')
mae_lightGBM_model_BG2_test = mean_absolute_error(test_actual_BG2, test_prediction_BG2)
mse_lightGBM_model_BG2_test = mean_squared_error(test_actual_BG2, test_prediction_BG2)
R2_lightGBM_model_BG2_test  = r2_score(test_actual_BG2, test_prediction_BG2)
print('test_mae_lightGBM_model_BG2', mae_lightGBM_model_BG2_test)
print('test_mse_lightGBM_model_BG2', mse_lightGBM_model_BG2_test)
print('test_rmse_lightGBM_model_BG2', np.sqrt(mse_lightGBM_model_BG2_test))
print('test_R2_lightGBM_model_BG2', R2_lightGBM_model_BG2_test)


elapsed = time.time() - t

print('Elapsed Time is : %.8f seconds ' % (elapsed))


