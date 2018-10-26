# -*- coding: utf-8 -*-
"""
Created by : Apoorba Bibeka
Edited by AB on 2018-10-26
Another change
"""


import os
# Use StringIO so that data can be read like a text file 
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
    
# Use pandas to get familiar with data frame in python
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import seaborn as sns
import numpy as np


print('Current working directory ',os.getcwd())
os.chdir('/Users/Apoorb/Dropbox/PHD_course_work/ISEN 616')
print('Current working directory ',os.getcwd())

# Use StringIO so that data can be read like a text file 
data1=StringIO('''x y
 0.11 0.05
 1.08 0.5 
 1.16 0.54 
 2.75 1.31 
 0.12 0.07 
  0.6 0.28 
 1.55 0.73 
    1 0.46 
 0.61 0.35 
 3.18  1.4 
 2.16 0.91 
 1.82 0.86 
 4.75 2.05 
 1.05 0.58 
 0.92 0.41 
 0.86  0.4 
 0.24 0.14 
 0.01 0.03 
 0.51 0.25 
 2.15 0.96 
 0.53 0.32 
  5.2 2.25 
    0 0.06 
 1.17  0.6 
 6.67  3.1 
 0.04 0.04 
 2.22    1 
 0.05 0.05 
 0.15 0.09 
 0.41 0.25 
 1.45  0.7 
 0.22 0.12 
 2.22    1 
 0.7  0.38 
 2.73 1.63 
 0.02 0.02 
 0.18 0.09 
 0.27 0.14 
 1.25 0.62 
 0.46 0.23 
 0.31 0.17 
 0.75 0.33 
 2.55 1.17 
    1 0.43 
 3.98 1.77 
 1.26 0.58 
  5.4 2.34 
 1.02  0.5 
 3.75 1.62 
  3.7  1.7 
  0.3 0.14 
 0.07 0.06 
 0.58 0.31 
 0.72 0.35 
 0.63 0.29 
 1.55 0.73 
 2.47 1.23 ''')
# Read the data above 
Rainfall_data=pd.read_csv(data1,delimiter=r"\s+",header=0)
# Rename columns
Rainfall_data.columns=['x','y']
# Create a new variable for yi-044xi
Rainfall_data['Y1']=Rainfall_data['y']-0.44*Rainfall_data['x']

# Q9a
plt.plot(Rainfall_data['Y1'],'bo')
plt.title("Residual Plot of $y_i-0.44x_i$")
plt.xlabel("Index")
plt.ylabel("$y_i-0.44x_i$")
plt.savefig('ResidualPlot.png')   # Ans 9a 

X=pd.DataFrame(Rainfall_data['x'])
Y=pd.DataFrame(Rainfall_data['y'])
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
R1=regr.fit(X,Y)

# Print the coefficients
print(R1.intercept_, R1.coef_)

# Make predictions using the testing set
y_pred = regr.predict(X)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.4f"
      % mean_squared_error(Y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y, y_pred))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#!9 b
#***********************************************************
# Model with INtercept
X1 = sm.add_constant(X) # another way to add a constant row for an intercept

# Note the swap of X and y
model = sm.OLS(Y, X1)
results = model.fit()
# Statsmodels gives R-like statistical output
print(results.summary())   ## Ans 9b

#***********************************************************
# Model Without INtercept
# Note the swap of X and y
model = sm.OLS(Y, X)
results = model.fit()
# Statsmodels gives R-like statistical output
print(results.summary())   ## Ans 9b



# =============================================================================
 #*******************************************************************************
 CensusDat = pd.read_csv("Problem13.txt",header=0,sep=r"\s+")
 CensusDat2=CensusDat.set_index("Place")
 CensusDat2_dummy= pd.get_dummies(CensusDat2,"City")
 CensusDat2_dummy.head()
 CensusDat2_dummy.iloc[0:3,0:4]
 # Creeate an empty linear regression model
 Model1=linear_model.LinearRegression()
 feature_cols =["Minority", "Crime", "Poverty", "Language", "High-school", "Housing", "Conventional"]
 
 X_13=CensusDat2_dummy[feature_cols]
 Y_13=CensusDat2_dummy['Undercount']
 X_13=sm.add_constant(X_13)
 model1 = sm.OLS(Y_13, X_13)
 res1 =model1.fit()
 print(res1.summary())   ## Ans 13
 
# fitted values (need a constant term for intercept)
model_fitted_y = res1.fittedvalues
model_residuals =res1.resid
# normalized residuals
model_norm_residuals = res1.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, Y_13, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_residuals[i]));
# =============================================================================
