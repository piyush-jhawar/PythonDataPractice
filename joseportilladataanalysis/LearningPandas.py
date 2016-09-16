import numpy as np
import pandas as pd
from pandas import  Series, DataFrame
import webbrowser
from numpy.random import rand


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

# ser1 = Series([1,2,3,4],index=['A','B','C','D'])
# print ser1
# ser2 = ser1.reindex(['A','B','C','D','E','F'])
#
# # print ser2
#
# ser2.reindex(['A','B','C','E','F','G'],fill_value=0)
# print ser2.index

# ser3 = Series(['USA','Mexico','Canada'],index=[0,5,10])
# print ser3
#
# ser3.reindex(range(15),method='ffill')#fffil will fill usa for 1234,
# # and ffill will fill Canada for 6,7,8,9 same continues.
# print ser3
#
# dframe = DataFrame(rand(25).reshape((5,5)),index=['A','B','D','E','F'],
#                    columns=['Col1','Col2','Col3','Col4','Col5'])
# print dframe
#
# dframe2 = dframe.reindex(['A','B','C','D','E','F'])
# print dframe2
#
# columns = ['Col1','Col2','Col3','Col4','Col5','Col6']
# dframe2.reindex(columns=new_columns)
# print dframe2

# print dframe.ix[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],columns]

#Lecture 5
#Drop Entry from Series and DF
ser1 = Series(np.arange(3),index=['a','b','c'])
print ser1

#to drop a index
ser1.drop('b')
print ser1

dframe1 = DataFrame(np.arange(9).reshape(3,3),index=['SF','LA','NY'],columns=['Pop','Size','Year'])
print dframe1

dframe1.drop('LA')
print dframe1
#To drop it permanently move it to another variable else the orginal DF will retains it row
dframe2 = dframe1.drop('LA')
print dframe2
#by default drop user axis 0 that is row, for column you have to specify
#axis mandatorily.
# To drop columns
dframe1.drop('year',axis=1)

#Lecture 6