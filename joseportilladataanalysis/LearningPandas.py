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


import numpy as np
import pandas as pd
from pandas import  Series, DataFrame
import webbrowser
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
#nfl_frame = pd.read_clipboard()
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