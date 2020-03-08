import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np

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

# exporting dataframe to new csv

wine.to_csv(r'wine_clean.csv')
