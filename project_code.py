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


# creating price subrgoups to further analyze data

price_1 = wine2.points[wine2.price < 40]
price_2 = wine2.points[(wine2.price) >= 40 & (wine2.price < 80)]
price_3 = wine2.points[(wine2.price) >= 80 & (wine2.price < 120)]
price_4 = wine2.points[wine2.price <= 120]


# Visualizing the means of the subgroups
price_means=[round(np.mean(price_1),2),round(np.mean(price_2),2),round(np.mean(price_3),2), round(np.mean(price_4),2)]
price_means_labels= ['$0-39','$40-79','$80-120','$120+']


_= sns.barplot(x= price_means_labels, y=price_means)
plt.ylim(80,90)
plt.title('Average Wine Review Score Per Price Range')
plt.ylabel('Price (0-100)')
plt.xlabel('Price Range ($)')
for index, value in enumerate(price_means):
    plt.text(index, value, str(value))
plt.show()


# Bootstrap test between the most expensive and least expensive sub groups

def bs_replicate_1d(data, func, **kwargs):
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample, **kwargs)

def mean_diff(x0, x1):
    m0 = bs_replicate_1d(x0, np.mean)
    m1 = bs_replicate_1d(x1, np.mean)
    return m0 - m1

mean_diffs = [mean_diff(price_1, price_4) for i in np.arange(10000)]

mean_diff_pc = np.percentile(mean_diffs, [2.5, 97.5])
mean_diff_pc

_ = plt.hist(mean_diffs, bins=30)
_ = plt.axvline(x=mean_diff_pc[0], color='r')
_ = plt.axvline(x=mean_diff_pc[1], color='r')
_ = plt.xlabel('Difference of means')
_ = plt.ylabel('Frequency')
_ = plt.title('Difference between means of \$0-39 and \$120+ Review Scores')
plt.show()




