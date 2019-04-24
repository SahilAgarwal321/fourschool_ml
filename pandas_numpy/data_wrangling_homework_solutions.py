import numpy as np
import pandas as pd

transactions = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/transactions.csv')

#======================================================================================================
#AA. Meta info

# Q1 Show dataframe summary

print(transactions.info())

# Q2 How many observations are there in data?

print(transactions.shape[0])

# Q3 How many features (columns) exist in the dataset?

print(transactions.shape[1])

# Q4 Get feature names

print(transactions.columns)

# Q5 Get the column names as an array

print(transactions.columns.values)

# Q6 Change the name of column "Quantity" to "Quant"

# this creates new dataframe with change
new_transactions = transactions.rename(columns={'Quantity': 'Quant'})
print(new_transactions)
# This changes old dataframe
# transactions.rename(columns={'Quantity': 'Quant'}, inplace=True)
# print(transactions)

# Q7 Change the names of columns ProductID and UserID to PID and UID respectively

new_transactions = transactions.rename(columns={'ProductID': 'PID', 'UserID': 'UID'})
print(new_transactions)


#======================================================================================================
#AB. Ordering the rows of a DataFrame

# Q8 Order the rows of transactions by TransactionID (descending )

print(transactions.sort_values(by='TransactionID', ascending=False))

# Q9 Order the rows of transactions by Quantity ascending and TransactionDate descending

print(transactions.sort_values(by=['Quantity', 'TransactionID'], ascending=[True, False]))

#======================================================================================================
#AC. Ordering the columns of a DataFrame

# Q10 Set the column order of transactions as: ProductID, Quantity, TransactionDate, TransactionID, UserID

print(transactions[['ProductID', 'Quantity', 'TransactionDate', 'TransactionID', 'UserID']])

# Q11 Make UserID the first column of transactions dataset

print(transactions[set(['UserID'] + transactions.columns.values.tolist())])

#======================================================================================================
#AD. Extracting arrays from a DataFrame

# Q12 Extract just the 2nd column

print(transactions[transactions.columns[1]])

# Q13 Get ProductID as an array of values

print(transactions['ProductID'].tolist())

# Q14 Let:
col = "ProductID"
# Use variable 'col' to get all ProductID values of dataset as an array

print(transactions[col].tolist())

#======================================================================================================
#AE. Row subsetting

# Q15 Subset rows 1, 3, and 6

print(transactions.iloc[[0, 2, 5]])

# Q16 Subset rows exlcuding 1, 3, and 6

bad_transactions = transactions.index.isin([0, 2, 5])
print(transactions[~bad_transactions])
# transactions.drop([0,2,5])

# Q17 Subset the first 3 rows

print(transactions.iloc[:3])

# Q18 Subset rows excluding the first 3 rows

print(transactions.iloc[3:])

# Q19 Subset the last 2 rows

print(transactions.iloc[-2:])

# Q20 Subset rows excluding the last 2 rows

print(transactions.iloc[:-2])

# Q21 Subset rows where Quantity > 1

print(transactions[transactions['Quantity'] > 1])

# Q22 Subset rows where UserID = 2

print(transactions[transactions['UserID'] == 2])

# Q23 Subset rows where Quantity > 1 and UserID = 2

print(transactions[(transactions['UserID'] == 2) & (transactions['Quantity'] > 1)])

# Q24 Subset rows where Quantity + UserID is > 3

print(transactions[transactions['UserID'] + transactions['Quantity'] > 3])

# Q25 Subset rows where an external array, foo, is True

foo = np.array([True, False] * 5)
print(transactions[foo == True])

# Q26 Subset rows where an external array, bar, is positive

bar = np.array([2, -3, -2, -2, 0, 4, -4, 5, 0, 7])
print(transactions[bar > 0])

# Q27 Subset rows where foo is TRUE or bar is negative

print(transactions[(foo == True) | (bar < 0)])

# Q28 Subset the rows where foo is not TRUE and bar is not negative

print(transactions[(foo != True) | (bar >= 0)])

#======================================================================================================
#AF. Column subsetting

# Q29 Subset by columns 1 and 3   (SEE ALL ANSWERS BELOW (LINE 240))

print(transactions.iloc[:, [0, 2]])

# Q30 Subset by columns TransactionID and TransactionDate

print(transactions[['TransactionID', 'TransactionDate']])
print(transactions.loc[:, ['TransactionID', 'TransactionDate']])

# Q31 Subset rows where TransactionID > 5 and subset columns by TransactionID and TransactionDate

print(transactions.loc[transactions['TransactionID'] > 5, ['TransactionID', 'TransactionDate']])

# Q32 Subset columns by a variable list of columm names

cols = np.array(["UserID", "Quantity", "TransactionDate"])
print(transactions[cols])
print(transactions.loc[:, cols])

# Q33 Subset columns excluding a variable list of column names

cols = np.array(["UserID", "Quantity", "TransactionDate"])
print(transactions.drop(cols, axis=1))
print(transactions[transactions.columns.difference(cols)])

#======================================================================================================
#AG. Inserting and updating values

# Q34 Convert the TransactionDate column to type Date

# Both methods work -
transactions = transactions.astype({'TransactionDate': np.datetime64})
print(transactions)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
print(transactions)

# Q35 Insert a new column, Foo = UserID + ProductID

transactions['Foo'] = transactions['UserID'] + transactions['ProductID']
print(transactions)

# Q36 Subset rows where TransactionID is even and set Foo = NA

transactions.loc[transactions['TransactionID'] % 2 == 0, 'Foo'] = np.nan
print(transactions)

# Q37  Add 100 to each TransactionID

transactions['TransactionID'] = transactions['TransactionID'] + 100
print(transactions)

# Q38 Insert a column indicating each row number

transactions['RowNumber'] = transactions.index
print(transactions)

# Q39 Insert columns indicating the rank of each Quantity, minimum Quantity and maximum Quantity

transactions['Rank'] = transactions['Quantity'].rank(method='average')
transactions['MinimumQuantity'] = transactions['Quantity'].min()
transactions['MaximumQuantity'] = transactions['Quantity'].max()
print(transactions)

# Q40 Remove column Foo

transactions = transactions.drop('Foo', axis=1)
print(transactions)

# Q41 Remove multiple columns RowIdx, QuantityRk, and RowIdx

# Removed the columns that I had added in Q39.
transactions = transactions.drop(['Rank', 'MinimumQuantity', 'MaximumQuantity'], axis=1)
print(transactions)


#======================================================================================================
#AH. Grouping the rows of a DataFrame

#--------------------------------------------------
# Group By + Aggregate

# Q42 Group the transations per user, measuring the number of transactions per user

transactions.groupby('UserID').size().reset_index(name='counts')
transactions.groupby('UserID')['UserID'].agg(np.size).reset_index(name='counts')

# Q43 Group the transactions per user, measuring the transactions and average quantity per user

# This one is wrong. Have to ask about this.
# transactions.groupby('UserID')['UserID'].agg([np.size, np.mean]).reset_index()


#======================================================================================================
#AI. Joining DataFrames

# Load followinng datasets from CSV files on github and answer questions relating to Basic Joins (below):

users = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/users.csv')
sessions = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/sessions.csv')
products = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/products.csv')
transactions = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/transactions.csv')

# Q44 Convert date columns to Date type

users['Registered'] = pd.to_datetime(users['Registered'])
users['Cancelled'] = pd.to_datetime(users['Cancelled'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

#--------------------------------------------------
#AJ. Basic Joins

# Q45 Join users to transactions, keeping all rows from transactions and only matching rows from users (left join)

pd.merge(transactions, users, how='left', on='UserID')
transactions.merge(users, how='left', on='UserID')

# Q46 Which transactions have a UserID not in users? (anti join)



# Q47 Join users to transactions, keeping only rows from transactions and users that match via UserID (inner join)



# Q48 Join users to transactions, displaying all matching rows AND all non-matching rows (full outer join)



# Q49 Determine which sessions occured on the same day each user registered



# Q50 Build a dataset with every possible (UserID, ProductID) pair (cross join)



# Q51 Determine how much quantity of each product was purchased by each user



# Q52 For each user, get each possible pair of pair transactions (TransactionID1, TransactionID2)



# Q53 Join each user to his/her first occuring transaction in the transactions table


#======================================================================================================
#AK. Reshaping a data.table

# Q54 Read following datasets from CSV

users = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/users.csv')
transactions = pd.read_csv('https://raw.githubusercontent.com/ben519/DataWrangling/master/Data/transactions.csv')

# Q55 Convert date columns to Date type

users['Registered'] = pd.to_datetime(users['Registered'])
users['Cancelled'] = pd.to_datetime(users['Cancelled'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Q56 Add column TransactionWeekday as Categorical type with categories Sunday through Saturday

transactions['TransactionWeekday'] = pd.Categorical(transactions['TransactionDate'].dt.weekday_name, categories=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
