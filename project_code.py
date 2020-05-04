import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
from scipy import stats


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





