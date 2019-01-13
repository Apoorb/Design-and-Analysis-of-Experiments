#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:10:50 2018

@author: Apoorb
"""


#******************************************************************************
#1 Introduction to Pandas
import pandas as pd
pandas.__version__

# Pandas data structures : "Series", "DataFrame" and "Index"

import numpy as np
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

data.values

data.index
data[1]

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])

data['b']

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data
data[5]

#Series as specialized dict
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)

population['California']
population['California':'Illinois']


pd.Series(5, index=[100, 200, 300])


pd.Series({2:'a', 1:'b', 3:'c'})




pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])


area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)

states = pd.DataFrame({'population': population,
                       'area': area})
states

states.index
states.columns


states['area']


pd.DataFrame(population, columns=['population'])



data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)


pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])


A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A
pd.DataFrame(A)

ind = pd.Index([2, 3, 5, 7, 11])
ind

ind[1]
ind[::2]

print(ind.size, ind.shape, ind.ndim, ind.dtype)

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB  # intersection
indA | indB  # union
indA ^ indB  # symmetric difference

#******************************************************************************
#2 Data Indexing and Selection 

import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data
data['b']

'a' in data

data.keys()

list(data.items())

data['e'] = 1.25
data

# slicing by explicit index
data['a':'c']

# slicing by implicit integer index
data[0:2]

# masking
data[(data > 0.3) & (data < 0.8)]

# fancy indexing
data[['a', 'e']]

#First, the loc attribute allows
#indexing and slicing that always references
#the explicit index:
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
data.loc[1]
data.loc[1:3]

#The iloc attribute allows indexing and slicing
#that always references the implicit Python-style index:
data.iloc[1]
data.iloc[1:3]

### DataFrame as a dictionary
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area']
data.area
data.area is data['area']

data.pop is data['pop']


data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']

data.ix[:3, :'pop']


#******************************************************************************
#3 Operations in Pandas
import pandas as pd
import numpy as np

ser = pd.Series(rng.randint(0, 10, 4))
ser

rng = np.random.RandomState(42)
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
df

np.exp(ser)

np.sin(df * np.pi / 4)

A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
A

B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
B

A + B

fill = A.stack().mean()
A.add(B, fill_value=fill)


A = rng.randint(10, size=(3, 4))
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]

df.subtract(df['R'], axis=0)

#******************************************************************************
#4 Handling Missing Data



#******************************************************************************
#5 Hierarchical Indexing
import pandas as pd
import numpy as np

index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop

#With this indexing scheme, you can straightforwardly index or slice the 
#series based on this multiple index:
pop[('California', 2010):('Texas', 2000)]


#But the convenience ends there. For example, if you need to select all
# values from 2010, you'll need to do some messy (and potentially slow) 
#munging to make it happen:
pop[[i for i in pop.index if i[1] == 2010]]

### The Better Way: Pandas MultiIndex
index = pd.MultiIndex.from_tuples(index)
index
pop = pop.reindex(index)
pop

pop[:, 2010]

pop_df = pop.unstack()
pop_df

pop_df.stack()

pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df

f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()

# Methods of MultiIndex Creation
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df

data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)


pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

pop.index.names = ['state', 'year']
pop

### MultiIndex for columns
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])


# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37


# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data

health_data['Guido']

pop['California', 2000]
pop['California']
pop.loc['California':'New York']
pop[:, 2000]

pop[pop > 22000000]
pop[['California', 'Texas']]

health_data['Guido', 'HR']
health_data.iloc[:2, :2]

data_mean = health_data.mean(level='year')
data_mean

data_mean.mean(axis=1, level='type')

#******************************************************************************
#6 Combining Datasets: Concat and Append

#******************************************************************************
#7 Merge and Join


import pandas as pd
import numpy as np

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# pd.merge
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})


display('df1', 'df2')
df3 = pd.merge(df1, df2)
df3

# Many to one
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
display('df3', 'df4', 'pd.merge(df3, df4)')

# Many to many

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
display('df1', 'df5', "pd.merge(df1, df5)")




display('df1', 'df2', "pd.merge(df1, df2, on='employee')")


df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')


pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)


df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')
display('df1a', 'df2a',
        "pd.merge(df1a, df2a, left_index=True, right_index=True)")

display('df1a', 'df2a', 'df1a.join(df2a)')



display('df1a', 'df3', "pd.merge(df1a, df3, left_index=True, right_on='name')")


#******************************************************************************
#7 Aggregate and Grouping
import numpy as np
import pandas as pd
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
df
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate({'data1': 'min',
                             'data2': 'max'})


import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape

planets.groupby('method')['orbital_period'].median()

for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))


planets.groupby('method')['year'].describe().unstack()

# Filtering

def filter_func(x):
    return x['data2'].std() > 4

display('df', "df.groupby('key').std()", "df.groupby('key').filter(filter_func)")



#Transformation
df.groupby('key').transform(lambda x: x - x.mean())

# The apply() method
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

display('df', "df.groupby('key').apply(norm_by_data2)")

# Using a different key
L = [0, 1, 0, 1, 2, 0]
display('df', 'df.groupby(L).sum()')


df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
display('df2', 'df2.groupby(mapping).sum()')

display('df2', 'df2.groupby(str.lower).mean()')

df2.groupby([str.lower, mapping]).mean()




decade = 10 * (planets['year'] // 10)


decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)


#******************************************************************************
#8 Pivot Tables

#Motivating Pivot Tables
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()
titanic.groupby('sex')[['survived']].mean()

titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
# Better approach
titanic.pivot_table('survived', index='sex', columns='class')


age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')

fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])


titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'})

titanic.pivot_table('survived', index='sex', columns='class', margins=True)

# Example of Birthrate data

#******************************************************************************
#9 Vectorize String Operations

data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]


import pandas as pd
names = pd.Series(data)
names

names.str.capitalize()

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

monte.str.lower()
monte.str.len()
monte.str.startswith('T')
monte.str.split()
monte.str.extract('([A-Za-z]+)', expand=False)
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
monte.str[0:3]
monte.str.split().str.get(-1)


full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte


full_monte['info'].str.get_dummies('|')

#******************************************************************************
#9 Working with Time Series
from datetime import datetime
datetime(year=2015, month=7, day=4)

from dateutil import parser
date = parser.parse("4th of July, 2015")
date

date.strftime('%A')

# Numpy datetime
import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
date

date + np.arange(12)


np.datetime64('2015-07-04')

np.datetime64('2015-07-04 12:00')

# Nanosecond based time
np.datetime64('2015-07-04 12:59:59.50', 'ns')

# Pandas datetime
import pandas as pd
date = pd.to_datetime("4th of July, 2015")
date

date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')

# Pandas Time Series: Indexing by Time
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data

data['2014-07-04':'2015-07-04']
data['2015']

dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
dates
dates.to_period('D')

dates - dates[0]

pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8)
pd.date_range('2015-07-03', periods=8, freq='H')
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')


from pandas_datareader import data

goog = data.DataReader('GOOG', start='2004', end='2016',
                       data_source='google')
goog.head()




























