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


# # import numpy as np
# # import pandas as pd
# # from pandas import Series,DataFrame
# #
# # #Lecture 01
# # obj = Series([3,6,9,12])
# # print obj
# # #To print the values present in Series .values
# # print obj.values
# # #To print the index of Series .index
# # print obj.index
# #
# # ww2_cas = Series([870000,430000,300000,210000,400000],index=['UUSR','Germany','China','Japan','USA'])
# # print ww2_cas
# # #to turn Series into Dictionary .to_dict()
#
#
#
# # https://en.wikipedia.org/wiki/NFL_win%E2%80%93loss_records
#
# # obj = Series([3,6,9,12])
# # # print obj
# # # print obj.values
# # # print obj.index
# #
# # ww2_cas = Series([87,43,30,21,40],index=['USSR','Germany','China','Japan','USA'])
# # # print ww2_cas
# # #
# # # print ww2_cas['USA']
# #
# # #Check which country had cas greater than 40
# # print ww2_cas[ww2_cas > 40]
# #
# # print 'USSR' in ww2_cas
# #
# # #Convert Series into Dictionary
# # ww2_dict = ww2_cas.to_dict()
# # print ww2_dict
# #
# # #Convert back to Series
# # ww2_series = Series(ww2_dict)
# # print ww2_series
# #
# # countries = ['China','Germany','Japan','USA','USSR','Argentina']
# # obj2 = Series(ww2_dict,index=countries)
# # # print obj2
# # #
# # # print pd.isnull(obj2)
# # # print pd.notnull(obj2)
# #
# # # print ww2_series+obj2
# #
# # obj2.name="World war 2 Casulalities"
# # # print obj2
# #
# # obj2.index.name = "Countries"
# # print obj2
#
# #Lecture 2
# #DataFrame
#
# # website = 'https://en.wikipedia.org/wiki/NFL_win%E2%80%93loss_records'
# # # webbrowser.open(website)
# #
# # nfl_frame = pd.read_clipboard()
# # print nfl_frame
# #
# # # print nfl_frame.columns
# # # print nfl_frame.Rank
# #
# # #to grab a column name with space
# # # print nfl_frame['First NFL Season']
# #
# # # print DataFrame(nfl_frame,columns=['Team','Rank','Total Games'])
# # # print DataFrame(nfl_frame,columns=['Team','Rank','Total Games','Stadium'])
# #
# # # How to retrieve rows from DF
# # print nfl_frame.head()
# # print nfl_frame.tail()
# #
# #
# # print nfl_frame.ix[3]
# # nfl_frame['Stadium'] = 'Levis Stadium'
# # nfl_frame['Stadium'] = np.arange(6)
# # # print nfl_frame.head()
# # Stadiums = Series(['Levis','ATNT'],index=[4,0])
# #
# # nfl_frame['Stadium']=Stadiums
# # print nfl_frame
# # del nfl_frame['Stadium']
# # print nfl_frame
#
# # data = {'City':['SF','LA','NYC'],'Population':[837,388,567]}
# # print data
# #
# # city_frame = DataFrame(data)
# # print city_frame
# # print nfl_frame.iloc[0]
# # print nfl_frame[0:1] #to fetch zeroth position index
#
# #Lecture 3
# # my_ser = Series([1,2,3,4],index=['A','B','C','D'])
# # print my_ser
# #
# # my_index = my_ser.index
# # print my_index[2:]
# #my_index[0] = 'Z'#cannot be performed
#
#
# #Lecture 4
# #Re-indexing
#
# ser1 = Series([1,2,3,4],index=['A','B','C','D'])
# # print ser1
# ser2 = ser1.reindex(['A','B','C','D','E','F'])
#
# # print "Ser2\n",ser2
#
# # print ser2.reindex(['A','B','C','E','F','G'],fill_value=0)
# # print ser3
#
# ser3 = Series(['USA','Mexico','Canada'],index=[0,5,10])
# # print "Ser3",ser3.iloc[2]
#
# # ser3.reindex(range(15),method='ffill')
# # #fffil will fill usa for 1234,
# # # and ffill will fill Canada for 6,7,8,9 same continues.
# # print ser3.reindex(range(15),method='ffill')
# #
# # dframe = DataFrame(randn(25).reshape((5,5)),index=['A','B','D','E','F'],
# #                    columns=['Col1','Col2','Col3','Col4','Col5'])
# # print dframe
#
# # dframe2 = dframe.reindex(['A','B','C','D','E','F'])
# # # print "dframe2",dframe2
# #
# # new_columns = ['Col1','Col2','Col3','Col4','Col5','Col6']
# # dframe2.reindex(columns=new_columns)
# # # print dframe2
#
#
# # print dframe.ix[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],columns]
#
# # #Drop Entry from Series and DF
# # ser1 = Series(np.arange(3),index=['a','b','c'])
# # print ser1
# # #Lecture 5
# #
# # #to drop a index
# # ser1.drop('b')
# # print ser1
# #
# # dframe1 = DataFrame(np.arange(9).reshape(3,3),index=['SF','LA','NY'],columns=['Pop','Size','Year'])
# # print dframe1
# #
# # dframe1.drop('LA')
# # print dframe1
# # #To drop it permanently move it to another variable else the orginal DF will retains it row
# # dframe2 = dframe1.drop('LA')
# # print dframe2
# # #by default drop user axis 0 that is row, for column you have to specify
# # #axis mandatorily.
# # # To drop columns
# # dframe1.drop('year',axis=1)
# #
# #Lecture 6
# #Selecting Entries
# # ser1 = Series(np.arange(3),index=['A','B','C'])
# # ser1 = 2 * ser1
# #
# # print ser1
# # print ser1['B']
# # print ser1[1]
# # print ser1[0:3]
#
# # dframe = DataFrame(np.arange(25).reshape(5,5),index=['NYC','LA','SF','DC','CA'],
# #                    columns=['A','B','C','D','E'])
# # print dframe.ix['LA']
# #Lecture 7
# #Data Alignment
# ser1 = Series([0,1,2],index=['A','B','C'])
# # print ser1
# ser2 = Series([3,4,5,6],index=['A','B','C','D'])
# # print ser2
# # print ser1 + ser2
# dframe1 = DataFrame(np.arange(4).reshape(2,2),columns=list('AB'),index=['NYC','LA'])
# dframe2 = DataFrame(np.arange(9).reshape(3,3),columns=list('ADC'),index=['NYC','SF','LA'])
# # print dframe1 + dframe2
# # print dframe1.add(dframe2,fill_value=0)
#
#
# # print Series([1,2],index=['A','B'],name='Piyush')
#
# #Lecture 8
# #Rank and Sort
# ser1 = Series(range(3),index=['C','A','B'])
# # print ser1
# # print ser1.sort_index()
# # print ser1.sort_values()#order is deprecated
#
# ser2 = Series(randn(10))
# # print ser2.sort_values()
# # print ser2.rank()
#
# ser3 = Series(randn(10))
# # print ser3
# # print ser3.rank()
#
# #Lecure 9
# #Summary Stastics
# arr = np.array([[1,2,np.nan],[np.nan,3,4]])
# dframe1 = DataFrame(arr,index=['A','B'],columns=['One','Two','Three'])
# # print dframe1
#
# # print dframe1.sum()
# # print dframe1.sum(axis=1)
# # print dframe1.min()
# # print dframe1.idxmin()
# # print dframe1.cumsum()
#
# # print dframe1.describe()
#
# YouTubeVideo('xGbpuFNR1ME')
# # prices = data.get_data_yahoo('AAPL')
#
# prices = data.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),
#                               end=datetime.datetime(2013,1,1))['Adj Close']
# # print prices.head()
#
# volume = data.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),
#                               end=datetime.datetime(2013,1,1))['Volume']
# # print volume.head()
#
# rets = prices.pct_change()
# # print rets.head()
# #Corelation of Stock
#
#
# # corr = rets.corr
# # plt.plot(prices)
# # plt.draw()
# # plt.show()
# # sns.set()
# # sns.heatmap(rets,annot=False)
# # sns.plt.show()
#
# # uniform_data = np.random.rand(10, 12)
# # sns.heatmap(uniform_data)
# # sns.plt.show()
# ser1 = Series(['w','w','x','y','z','w','x','y','x','a'])
# # print ser1
# # print ser1.unique()
# # print ser1.value_counts()
# #Lecture 10
# #Missing Data
#
# data = Series(['One','Two',np.NaN,'Four'])
# # print data
# # print data.isnull()
# # print data.dropna()
# # dframe = DataFrame([[1,2,3],[np.NaN,5,6],[7,np.NaN,9],
# #                    [np.NaN,np.NaN,np.NaN]])
# # print dframe
# # clean_dframe = dframe.dropna()
# # print clean_dframe
# # print dframe.dropna(how='all')
# npn = np.NaN
# dframe2 = DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])
# # print dframe2
# # print dframe2.dropna(thresh=2)
# # print dframe2.fillna(1)
# # print dframe2.fillna({0:0,1:1,2:2,3:3})
# #Lecture 11
# #Index Hierarchy
#
# ser = Series(randn(6),index=[[1,1,1,2,2,2],['a','b','c','a','b','c']])
# # print ser#.index
# # print ser[:,'a']
#
# dframe = ser.unstack()
# # print dframe
#
#
#
# #Section  5
# #Lecture 1
# #Reading and Writing Text Files
# dframe = pd.read_csv('lec25.csv',header=None)
# # print dframe
#
#
# dframe = pd.read_table('lec25.csv',sep=',',header=None)
# # print dframe
#
# dframe = pd.read_csv('lec25.csv',header=None,nrows=2)
# # print dframe
# dframe.to_csv('mytextdata_out.csv')
# # dframe.to_csv(sys.stdout)
#
# # dframe.to_csv(sys.stdout,sep='_')#sep='; or ? or |'
#
# # dframe.to_csv(sys.stdout,columns=[0,1,2])
#
# #Lecture 2
# #Reading and writing JSON
#
# json_obj = """
# {   "zoo_animal": "Lion",
#     "food": ["Meat", "Veggies", "Honey"],
#     "fur": "Golden",
#     "clothes": null,
#     "diet": [{"zoo_animal": "Gazelle", "food":"grass", "fur": "Brown"}]
# }
# """
#
# data = json.loads(json_obj)
# # print data
# # print json.dumps(data)
#
# dframe = DataFrame(data['diet'])
# # print dframe
#
# #Lecture 3
# #HTML with Pandas
#
# url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'
#
# dframe_list = pd.io.html.read_html(url)
# dframe = dframe_list[0]
# # print dframe.columns.values
#
# #Lecture 4
# #Working with Excel
#
#
# xlsfile = pd.ExcelFile('lec_28_test.xlsx')
# dframe = xlsfile.parse('Sheet1')
# # print dframe

#Section 6
#Lecture 1
#Merge
dframe1 = DataFrame({'key':['X','Z','Y','Z','X','X'],'data_set_1':np.arange(6)})
# print dframe1
dframe2 = DataFrame({'key':['Q','Y','Z'],'data_set_2':[1,2,3]})
# print dframe2
# print pd.merge(dframe1,dframe2)
# print pd.merge(dframe1,dframe2,on='key')
# print pd.merge(dframe1,dframe2,on='key',how='left')
# print pd.merge(dframe1,dframe2,on='key',how='right')
# print pd.merge(dframe1,dframe2,on='key',how='outer')#outer means Union of both key
dframe3 = DataFrame({'key':['X','X','X','Y','Z','Z'],'data_set_3':range(6)})
# print dframe3
dframe4 = DataFrame({'key':['Y','Y','X','X','Z'],'data_set_4':range(5)})
# print dframe4
# print pd.merge(dframe3,dframe4)

df_left = DataFrame({'key1':['SF','SF','LA'],
                     'key2':['One','Two','One'],
                     'left_data':[10,20,30]})
# print df_left
df_right = DataFrame({'key1':['SF','SF','LA','LA'],
                     'key2':['One','One','One','Two'],
                     'right_data':[40,50,60,70]})
# print df_right
# print pd.merge(df_left,df_right,on=['key1','key2'],how='outer')
# print pd.merge(df_left,df_right,on='key1')
# print pd.merge(df_left,df_right,on='key1',suffixes=('_lefty','_righty'))

#Lecture 2
#Merge On Index

df_left = DataFrame({'key':['X','Y','Z','X','Y'],
                     'data':range(5)})
# print df_left
df_right = DataFrame({'group_data':[10,20]},
                     index=['X','Y'])
# print df_right
# print pd.merge(df_left,df_right,left_on='key',right_index=True)

df_left_hr = DataFrame({'key1':['SF','SF','SF','LA','LA'],
                        'key2':[10,20,30,20,30],
                        'data_set':np.arange(5)})
# print df_left_hr

df_right_hr = DataFrame(np.arange(10).reshape(5,2),
                       index=[['LA','LA','SF','SF','SF'],
                              [20,10,10,10,20]],
                       columns=['col_1','col_2'])
# print df_right_hr

# print pd.merge(df_left_hr,df_right_hr,left_on=['key1','key2'],right_index=True)
# print df_left.join(df_right)

#Lecture 3
#Concetanate

arr1 = np.arange(9).reshape(3,3)
# print arr1
# print np.concatenate([arr1,arr1],axis=1)

ser1 = Series([0,1,2],index=['T','U','V'])
ser2 = Series([3,4],index=['X','Y'])
# print ser1
# print ser2
#
# print pd.concat([ser1,ser2])
# print pd.concat([ser1,ser2],axis=1)
# print pd.concat([ser1,ser2],keys=['cat1','cat2'])

dframe1 = DataFrame(np.random.randn(4,3),columns=['X','Y','Z'])
# print dframe1
dframe2 = DataFrame(np.random.randn(3,3),columns=['Y','Q','X'])
# print dframe2
#
# print pd.concat([dframe1,dframe2],ignore_index=True)


#Lecture 4
#Combine DataFrames

# ser1 = Series([2,np.nan,4,np.nan,6,np.nan],
#            index=['Q','R','S','T','U','V'])
# print ser1
#
# ser2 = Series(np.arange(len(ser1)),
#            index=['Q','R','S','T','U','V'])#, dtype=np.float64
# print ser2

# print Series(np.where(pd.isnull(ser1),ser2,ser1),index=ser1.index)

# print ser1.combine_first(ser2)
nan = np.nan
dframe_odds = DataFrame({'X':[1.,nan,3.,nan],
                         'Y':[nan,5.,nan,7.],
                         'Z':[nan,9,nan,11]})
# print dframe_odds

dframe_evens = DataFrame({'X': [2., 4., nan, 6., 8.],
                     'Y': [nan, 10., 12., 14., 16.]})
# print dframe_evens
#
# print dframe_odds.combine_first(dframe_evens)

#Lecture 5
#Reshaping DataFrames

dframe1 = DataFrame(np.arange(8).reshape((2, 4)),
                 index=pd.Index(['LA', 'SF'], name='city'),
                 columns=pd.Index(['A', 'B', 'C','D'], name='letter'))
print dframe1

dframe_st = dframe1.stack()
# print dframe_st
#
# print dframe_st.unstack()
# print dframe_st.unstack('letter')
# print dframe_st.unstack('city')

ser1 = Series([0,1,2], index=['Q','X','Y'])
ser2 = Series([4,5,6], index=['X','Y','Z'])

dframe = pd.concat([ser1,ser2],keys=['Alpha','Beta'])
print dframe

print dframe.unstack()
dframe = dframe.unstack()
print dframe
print dframe.stack(dropna=False)




