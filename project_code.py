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



# Checking for duplicates  in the 'country' Column
print(wine['country'].value_counts().sort_index())



#checking for duplicates in the 'designation' Column
designation_count = wine['designation'].value_counts().sort_index()
print(designation_count)


#checking for duplicates in the 'province' Column
province_count = wine['province'].value_counts().sort_index()
print(province_count)


#checking for duplicates in the 'taster_name' Column
print(wine['taster_name'].value_counts().sort_index())


#checking for duplicates in the 'variety' Column
print(wine['variety'].value_counts().sort_index())


#checking for duplicates in the 'winery' Column
print(wine['winery'].value_counts().sort_index())