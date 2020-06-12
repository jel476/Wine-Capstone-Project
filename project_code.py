import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


wine = pd.read_csv('winemag-data-130k-v2.csv')

#Checking for outliers

print(wine.info())

print(wine.describe())


# price seems to have a high max therefore I will analyze further

wine_price_sorted = wine['price'].sort_values()
print(wine_price_sorted.tail(10000))


wine.boxplot(column='price')
plt.show()

#Removing outliers

wine.drop(wine[wine['price']>= 1000].index, inplace = True)

wine.boxplot(column='price')
plt.show()


print(wine.info())

print(wine.describe())

#Removing observations with missing price values

wine.dropna(subset=['price'], inplace=True) 

wine.boxplot(column='price')
plt.show()

print(wine.info())

print(wine.describe())

wine_price_sorted = wine['price'].sort_values()
print(wine_price_sorted.tail(10000))

print(wine_price_sorted.head(10000))


#further removing outliers

z = np.abs(stats.zscore(wine['price']))
print(z)

print(np.where(z > 3))


wine2 = wine[(z < 3)]

print(wine.shape)
print(wine2.shape)

wine2.boxplot(column='price')
plt.show()

# exporting dataframe to new csv

wine.to_csv(r'wine_clean.csv')


# creating  arrays of X and y

arry= wine2['points'].values
arrx= wine2['price'].values

# reshaping the arrays

y= arry.reshape(-1,1)
X= arrx.reshape(-1,1)



# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

# cross validating alpha of Ridge
c_space = np.logspace(-5,-4,50)
param_grid = {'alpha': c_space}

ridge_reg = Ridge()

ridge_reg_cv= GridSearchCV(ridge_reg, param_grid, cv=5 )

ridge_reg_cv.fit(X_train, y_train)

print(ridge_reg_cv.best_params_)

#ridge regression
ridge_reg.fit(X_train,y_train)
ridge_reg.predict(X_test)

print(ridge_reg.score(X_test, y_test))
#log values

y1 = np.log(y)
X1 = np.log(X)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = .2, random_state=42)

ridge_regX1 = Ridge()

ridge_regX1.fit(X1_train,y1_train)
ridge_regX1.predict(X1_test)

print(ridge_regX1.score(X1_test, y1_test))

# adding country 

wine_country = wine2[['country', 'points', 'price']]
winex2 = pd.get_dummies(wine_country, drop_first=True)


y2 = winex2['points'].values
X2 = winex2.drop('points', axis=1).values



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = .2, random_state=42)

c_space = np.logspace(-5,0,50)
param_grid = {'alpha': c_space}


ridge_regX2= Ridge()

ridge_regX2_cv= GridSearchCV(ridge_regX2, param_grid, cv=5 )

ridge_regX2_cv.fit(X2_train, y2_train)

print(ridge_regX2_cv.best_params_)

ridge_reg.fit(X2_train,y2_train)
ridge_reg.predict(X2_test)

print(ridge_reg.score(X2_test, y2_test))


#log country values


y2_log = np.log(y2)
X2_log = X2
X2_log[:,0] = np.log(X2_log[:,0])


X2_log_train, X2_log_test, y2_log_train, y2_log_test = train_test_split(X2_log, y2_log, test_size = .2, random_state=42)

ridge_regX2_log = Ridge()

ridge_regX2_log.fit(X2_log_train,y2_log_train)
ridge_regX2_log.predict(X2_log_test)

print(ridge_regX2_log.score(X2_log_test, y2_log_test))


# Province instead of country 

wine_province = wine2[['province', 'points', 'price']]
winex3 = pd.get_dummies(wine_province, drop_first=True)


y3 = winex3['points'].values
X3 = winex3.drop('points', axis=1).values

y3_log = np.log(y3)
X3_log = X3
X3_log[:,0] = np.log(X3_log[:,0])


X3_log_train, X3_log_test, y3_log_train, y3_log_test = train_test_split(X3_log, y3_log, test_size = .2, random_state=42)

ridge_regX3_log = Ridge()

ridge_regX3_log.fit(X3_log_train,y3_log_train)
ridge_regX3_log.predict(X3_log_test)

print(ridge_regX3_log.score(X3_log_test, y3_log_test))


# adding  winery

wine_winery = wine2[['province', 'points', 'price', 'variety']]
winex4 = pd.get_dummies(wine_winery, drop_first=True)

y4 = winex4['points'].values
X4 = winex4.drop('points', axis=1).values

y4_log = np.log(y4)
X4_log = X4
X4_log[:,0] = np.log(X4_log[:,0])

X4_train, X4_test, y4_train, y4_test = train_test_split(X4_log, y4_log, test_size = .2, random_state=42)


ridge_X4 = Ridge()

ridge_X4.fit(X4_train,y4_train)
ridge_X4.predict(X4_test)

print(ridge_X4.score(X4_test, y4_test))


# delogging y

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = .2, random_state=42)


ridge_X4 = Ridge()

ridge_X4.fit(X4_train,y4_train)
ridge_X4.predict(X4_test)

print(ridge_X4.score(X4_test, y4_test))

# lasso

lasso = Lasso(alpha = 0.00001, normalize=True )

lasso.fit(X4_train,y4_train)
lasso.predict(X4_test)

print(lasso.score(X4_test, y4_test))


#gridsearch CV for lasso

lasso_reg_cv= GridSearchCV(Lasso(), param_grid, cv=5)

lasso_reg_cv.fit(X4_train, y4_train)

print(lasso_reg_cv.best_params_)



# Root Mean Squared Error

rmse = np.sqrt(mean_squared_error(y4_test, lasso.predict(X4_test)))

rmse2 = np.sqrt(mean_squared_error(y4_test, ridge_X4.predict(X4_test)))

print(rmse)
print(rmse2)
