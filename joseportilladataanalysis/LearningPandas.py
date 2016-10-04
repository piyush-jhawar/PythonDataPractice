import numpy as np
import pandas as pd
from pandas import  Series, DataFrame
import webbrowser
from numpy.random import randn
from IPython.display import YouTubeVideo
from pandas_datareader import data,wb
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import json
from pandas import read_html


# import numpy as np
# import pandas as pd
# from pandas import Series,DataFrame
#
# #Lecture 01
# obj = Series([3,6,9,12])
# print obj
# #To print the values present in Series .values
# print obj.values
# #To print the index of Series .index
# print obj.index
#
# ww2_cas = Series([870000,430000,300000,210000,400000],index=['UUSR','Germany','China','Japan','USA'])
# print ww2_cas
# #to turn Series into Dictionary .to_dict()



# https://en.wikipedia.org/wiki/NFL_win%E2%80%93loss_records

# obj = Series([3,6,9,12])
# # print obj
# # print obj.values
# # print obj.index
#
# ww2_cas = Series([87,43,30,21,40],index=['USSR','Germany','China','Japan','USA'])
# # print ww2_cas
# #
# # print ww2_cas['USA']
#
# #Check which country had cas greater than 40
# print ww2_cas[ww2_cas > 40]
#
# print 'USSR' in ww2_cas
#
# #Convert Series into Dictionary
# ww2_dict = ww2_cas.to_dict()
# print ww2_dict
#
# #Convert back to Series
# ww2_series = Series(ww2_dict)
# print ww2_series
#
# countries = ['China','Germany','Japan','USA','USSR','Argentina']
# obj2 = Series(ww2_dict,index=countries)
# # print obj2
# #
# # print pd.isnull(obj2)
# # print pd.notnull(obj2)
#
# # print ww2_series+obj2
#
# obj2.name="World war 2 Casulalities"
# # print obj2
#
# obj2.index.name = "Countries"
# print obj2

#Lecture 2
#DataFrame

# website = 'https://en.wikipedia.org/wiki/NFL_win%E2%80%93loss_records'
# # webbrowser.open(website)
#
# nfl_frame = pd.read_clipboard()
# print nfl_frame
#
# # print nfl_frame.columns
# # print nfl_frame.Rank
#
# #to grab a column name with space
# # print nfl_frame['First NFL Season']
#
# # print DataFrame(nfl_frame,columns=['Team','Rank','Total Games'])
# # print DataFrame(nfl_frame,columns=['Team','Rank','Total Games','Stadium'])
#
# # How to retrieve rows from DF
# print nfl_frame.head()
# print nfl_frame.tail()
#
#
# print nfl_frame.ix[3]
# nfl_frame['Stadium'] = 'Levis Stadium'
# nfl_frame['Stadium'] = np.arange(6)
# # print nfl_frame.head()
# Stadiums = Series(['Levis','ATNT'],index=[4,0])
#
# nfl_frame['Stadium']=Stadiums
# print nfl_frame
# del nfl_frame['Stadium']
# print nfl_frame

# data = {'City':['SF','LA','NYC'],'Population':[837,388,567]}
# print data
#
# city_frame = DataFrame(data)
# print city_frame
# print nfl_frame.iloc[0]
# print nfl_frame[0:1] #to fetch zeroth position index

#Lecture 3
# my_ser = Series([1,2,3,4],index=['A','B','C','D'])
# print my_ser
#
# my_index = my_ser.index
# print my_index[2:]
#my_index[0] = 'Z'#cannot be performed


#Lecture 4
#Re-indexing

ser1 = Series([1,2,3,4],index=['A','B','C','D'])
# print ser1
ser2 = ser1.reindex(['A','B','C','D','E','F'])

# print "Ser2\n",ser2

# print ser2.reindex(['A','B','C','E','F','G'],fill_value=0)
# print ser3

ser3 = Series(['USA','Mexico','Canada'],index=[0,5,10])
# print "Ser3",ser3.iloc[2]

# ser3.reindex(range(15),method='ffill')
# #fffil will fill usa for 1234,
# # and ffill will fill Canada for 6,7,8,9 same continues.
# print ser3.reindex(range(15),method='ffill')
#
# dframe = DataFrame(randn(25).reshape((5,5)),index=['A','B','D','E','F'],
#                    columns=['Col1','Col2','Col3','Col4','Col5'])
# print dframe

# dframe2 = dframe.reindex(['A','B','C','D','E','F'])
# # print "dframe2",dframe2
#
# new_columns = ['Col1','Col2','Col3','Col4','Col5','Col6']
# dframe2.reindex(columns=new_columns)
# # print dframe2


# print dframe.ix[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],columns]

# #Drop Entry from Series and DF
# ser1 = Series(np.arange(3),index=['a','b','c'])
# print ser1
# #Lecture 5
#
# #to drop a index
# ser1.drop('b')
# print ser1
#
# dframe1 = DataFrame(np.arange(9).reshape(3,3),index=['SF','LA','NY'],columns=['Pop','Size','Year'])
# print dframe1
#
# dframe1.drop('LA')
# print dframe1
# #To drop it permanently move it to another variable else the orginal DF will retains it row
# dframe2 = dframe1.drop('LA')
# print dframe2
# #by default drop user axis 0 that is row, for column you have to specify
# #axis mandatorily.
# # To drop columns
# dframe1.drop('year',axis=1)
#
#Lecture 6
#Selecting Entries
# ser1 = Series(np.arange(3),index=['A','B','C'])
# ser1 = 2 * ser1
#
# print ser1
# print ser1['B']
# print ser1[1]
# print ser1[0:3]

# dframe = DataFrame(np.arange(25).reshape(5,5),index=['NYC','LA','SF','DC','CA'],
#                    columns=['A','B','C','D','E'])
# print dframe.ix['LA']
#Lecture 7
#Data Alignment
ser1 = Series([0,1,2],index=['A','B','C'])
# print ser1
ser2 = Series([3,4,5,6],index=['A','B','C','D'])
# print ser2
# print ser1 + ser2
dframe1 = DataFrame(np.arange(4).reshape(2,2),columns=list('AB'),index=['NYC','LA'])
dframe2 = DataFrame(np.arange(9).reshape(3,3),columns=list('ADC'),index=['NYC','SF','LA'])
# print dframe1 + dframe2
# print dframe1.add(dframe2,fill_value=0)


# print Series([1,2],index=['A','B'],name='Piyush')

#Lecture 8
#Rank and Sort
ser1 = Series(range(3),index=['C','A','B'])
# print ser1
# print ser1.sort_index()
# print ser1.sort_values()#order is deprecated

ser2 = Series(randn(10))
# print ser2.sort_values()
# print ser2.rank()

ser3 = Series(randn(10))
# print ser3
# print ser3.rank()

#Lecure 9
#Summary Stastics
arr = np.array([[1,2,np.nan],[np.nan,3,4]])
dframe1 = DataFrame(arr,index=['A','B'],columns=['One','Two','Three'])
# print dframe1

# print dframe1.sum()
# print dframe1.sum(axis=1)
# print dframe1.min()
# print dframe1.idxmin()
# print dframe1.cumsum()

# print dframe1.describe()

YouTubeVideo('xGbpuFNR1ME')
# prices = data.get_data_yahoo('AAPL')

prices = data.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),
                              end=datetime.datetime(2013,1,1))['Adj Close']
# print prices.head()

volume = data.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),
                              end=datetime.datetime(2013,1,1))['Volume']
# print volume.head()

rets = prices.pct_change()
# print rets.head()
#Corelation of Stock


# corr = rets.corr
# plt.plot(prices)
# plt.draw()
# plt.show()
# sns.set()
# sns.heatmap(rets,annot=False)
# sns.plt.show()

# uniform_data = np.random.rand(10, 12)
# sns.heatmap(uniform_data)
# sns.plt.show()
ser1 = Series(['w','w','x','y','z','w','x','y','x','a'])
# print ser1
# print ser1.unique()
# print ser1.value_counts()
#Lecture 10
#Missing Data

data = Series(['One','Two',np.NaN,'Four'])
# print data
# print data.isnull()
# print data.dropna()
# dframe = DataFrame([[1,2,3],[np.NaN,5,6],[7,np.NaN,9],
#                    [np.NaN,np.NaN,np.NaN]])
# print dframe
# clean_dframe = dframe.dropna()
# print clean_dframe
# print dframe.dropna(how='all')
npn = np.NaN
dframe2 = DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])
# print dframe2
# print dframe2.dropna(thresh=2)
# print dframe2.fillna(1)
# print dframe2.fillna({0:0,1:1,2:2,3:3})
#Lecture 11
#Index Hierarchy

ser = Series(randn(6),index=[[1,1,1,2,2,2],['a','b','c','a','b','c']])
# print ser#.index
# print ser[:,'a']

dframe = ser.unstack()
# print dframe



#Section  5
#Lecture 1
#Reading and Writing Text Files
dframe = pd.read_csv('lec25.csv',header=None)
# print dframe


dframe = pd.read_table('lec25.csv',sep=',',header=None)
# print dframe

dframe = pd.read_csv('lec25.csv',header=None,nrows=2)
# print dframe
dframe.to_csv('mytextdata_out.csv')
# dframe.to_csv(sys.stdout)

# dframe.to_csv(sys.stdout,sep='_')#sep='; or ? or |'

# dframe.to_csv(sys.stdout,columns=[0,1,2])

#Lecture 2
#Reading and writing JSON

json_obj = """
{   "zoo_animal": "Lion",
    "food": ["Meat", "Veggies", "Honey"],
    "fur": "Golden",
    "clothes": null,
    "diet": [{"zoo_animal": "Gazelle", "food":"grass", "fur": "Brown"}]
}
"""

data = json.loads(json_obj)
# print data
# print json.dumps(data)

dframe = DataFrame(data['diet'])
# print dframe

#Lecture 3
#HTML with Pandas

url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'

dframe_list = pd.io.html.read_html(url)
dframe = dframe_list[0]
# print dframe.columns.values

#Lecture 4
#Working with Excel


xlsfile = pd.ExcelFile('lec_28_test.xlsx')
dframe = xlsfile.parse('Sheet1')
print dframe

#Section 6
#Lecture 1
#Merge

