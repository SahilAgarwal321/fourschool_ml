# Last amended: 18th April, 2019
# Myfolder: D:\data\OneDrive\Documents\black_friday

# 1.0 Clear memory and all variables

# 1.1 Import numpy As np
import numpy as np

# 1.2 Import pandas As pd
import pandas as pd

# 1.3 Import pyplot module of matplotlib as plt
import matplotlib.pyplot as plt

# 1.4 Import seaborn As sns
import seaborn as sns

# 1.5 Import os for Operating System related work
import os

# 1.6 Use 'os' to change your working dir to where your
#     black friday csv file is placed
os.chdir('./data_files')


# 1.7 List all files in your current working folder
os.listdir()


# 1.8 Count the number of files in your current working folder
len(os.listdir())


# 1.9  Read the dataset file 'BlackFriday.csv' or
#      black-friday.zip into a variable 'df'

# df = pd.read_csv("black-friday.zip")
df = pd.read_csv("./data_files/BlackFriday.csv")

### Exploration of data

# 2.0 Check the data type of 'df'. Is it
#     a pandas DataFrame or a Pandas Series?
print(type(df))


# 2.1 How much memory does the dataframe 'df' uses?
#     Hint: Use: .memory_usage()
print(df.memory_usage())



# 2.2 What is the shape of df
#     How many rows and columns does it have?
print(df.shape)


# 2.3 What are the column names of df?
print(df.columns)


# 2.4 Get columns names as an array object
#     Hint: Use attribute: .values
print(df.columns.values)



# 2.5 Print few five rows of DataFrame
print(df.head(5))


# 2.6 Print last five rows of DataFrame
print(df.tail(5))



# 2.7 Check data types of each column of df
print(df.dtypes)


# 2.8 User_ID should be 'object' and not int64
#     Make dtype of User_ID to 'object'
df['User_ID'] = df['User_ID'].astype('object')
print(df)


# 2.9 Summarise datatypes by counts of dtypes
#     ie how many are 'object', 'int64' etc
#     Hint: Use method value_counts()
print(df.dtypes.value_counts())



# 3.0 Give a statistical summary of dataframe df
#     In the summary include numeric and categorical (object)
#     datatypes also:
print(df.describe())



# 3.1 Extract just two columns from the dataset:
#     User_ID and Product_ID and also access its
#     rwos from row 10 to 20
print(df.loc[10:20, ['User_ID', 'Product_ID']])



# 3.2 Which columns in the dataset have null values
print(df.columns[df.isna().any()].tolist())

# 3.3 How many and which all Age (ie age-groups) exist?
#     Hint: Use value_counts()
print(len(df['Age'].value_counts()))
print(df['Age'].value_counts())

# 3.4  How many and which all kinds of Occupation exist?
print(len(df['Occupation'].value_counts()))
print(df['Occupation'].value_counts())

# 3.5 How many kinds of City_Category exist?
print(len(df['City_Category'].value_counts()))

# 3.6 How many types of Products (ie Product_ID) exist?
print(len(df['Product_ID'].value_counts()))


# 3.7 How many types of Product_Category_1,
#     Product_Category_2 and Product_Category_3 exist?
print(len(df['Product_Category_1'].value_counts()))
print(len(df['Product_Category_2'].value_counts()))
print(len(df['Product_Category_3'].value_counts()))


#3.8  Which top-10 User_ID occur most frequently
#     Hint: Use: value_counts(), sort_values()
#                and head()
print(df['User_ID'].value_counts().head(10).index.tolist())


# 3.9 Make a barplot of frequency of top-10 User_IDs
#     that occur most frequently on Black Friday
#     Can you order the graph either in
#     decreasing/increasing order of frequency
df['User_ID'].value_counts().head(10).plot(kind='bar')
df['User_ID'].value_counts().head(10).sort_values().plot(kind='bar')


#### Group and summarise data

# 4.0 Find average purchases ('Purchase') per User_ID
#     Hint: Use groupby(), sort_values() and head()
print(df.groupby('User_ID')['Purchase'].mean())


# 4.1 Refer answer to 4.0
#     Plot a barchart of User_IDs average purchases wise
df.groupby('User_ID')['Purchase'].mean().head(10).plot(kind='bar')
# Using head(10) here as showing all values takes a lot of time to load.


# 4.2 Product_ID wise average 'Purchase'?
print(df.groupby('Product_ID')['Purchase'].mean())


# 4.3 Plot top-10 Product_IDs most purchased on an average
df.groupby('Product_ID').size().sort_values(ascending=False).head(10).plot(kind='bar')
df.groupby('Product_ID').size().sort_values().tail(10).plot(kind='bar')


# 4.4 Product_Category_1 wise mean 'Purchase'?
print(df.groupby('Product_Category_1')['Purchase'].mean())


# 4.5 Plot top-10 Product_Category_1 most purchased on an average
df.groupby('Product_Category_1').size().sort_values().tail(10).plot(kind='bar')


# 4.6 Product_Category_2 wise mean 'Purchase'?
print(df.groupby('Product_Category_2')['Purchase'].mean())


# 4.7 Plot top-10 Product_Category_2 most purchased on an average
df.groupby('Product_Category_2').size().sort_values().tail(10).plot(kind='bar')


# 4.8 Product_Category_3 wise mean 'Purchase'?
print(df.groupby('Product_Category_3')['Purchase'].mean())


# 4.9 Plot top-10 Product_Category_3s most purchased on an average
df.groupby('Product_Category_3').size().sort_values().tail(10).plot(kind='bar')


# 5.0 Which Product_Category_1 is more popular in which City_Category?
df1 = df.groupby('City_Category')['Product_Category_1'].apply(lambda x: x.mode()).reset_index(name='Most_Popular_Product_Category')
df1.drop('level_1', axis=1, inplace=True)
print(df1)

# 5.1 Get City_Category wise average 'Purchase'
print(df.groupby('City_Category')['Purchase'].mean())

# 5.2 Get Age wise average 'Purchase'?
print(df.groupby('Age')['Purchase'].mean())


# 5.3 Get City_Category wise and Age wise average 'Purchase'
print(df.groupby(['City_Category', 'Age'])['Purchase'].mean())


#### Relationships between two categories

# 6.0 Is there any relationship between City_Category and Age?
#     Ref: https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
from scipy.stats import chi2_contingency

# 6.1  How confident would you want to be about the existence of any relationship?
#      95% or 90% or 99%......
confidence_level = 0.95         # 95% confident
level_of_significance = 1- confidence_level

# 6.2 Create a cross-table between two categorical variables
table = pd.crosstab(df.City_Category,df.Age)
table


# 6.3 Apply chi-square test of independence and get p_value
_, p_value, _, _ = chi2_contingency(table)


# 6.4 Now examine p_value
if p_value <= level_of_significance:
    print("Categorical variables have relationships")
else:
    print("Categorical variables have no relationships")


# 7.0 Similarly examine if there is any relationship between Age and Occupation?
contingency_table = pd.crosstab(df['Age'], df['Occupation'])
_, p_value, _, _ = chi2_contingency(contingency_table)
if p_value <= level_of_significance:
    print("Categorical variables have relationships")
else:
    print("Categorical variables have no relationships")


# 7.1 And also examine if there is any Gender  and Marital_Status?
contingency_table = pd.crosstab(df['Gender'], df['Marital_Status'])
_, p_value, _, _ = chi2_contingency(contingency_table)
if p_value <= level_of_significance:
    print("Categorical variables have relationships")
else:
    print("Categorical variables have no relationships")


######################
