# from __future__ import division
# import numpy as np
# import matplotlib.pyplot as plt
#
# #Lecture 01
# # my_list1 = [1,2,3,4]
# # my_array = np.array(my_list1)
# # print my_array
# # my_list2 = [11,55,33,44]
# # my_lists = [my_list1,my_list2]
# # my_array2 = np.array(my_lists)
# #
# # #To check the dimension we use .shape
# # print my_array2.shape
# # #To get the postion of the maximum vaule we use .argmax
# # print my_array2.argmax()
# # print my_array2.max()
# #
# # # to find the data type of the values .dtype
# # print my_array2.dtype
# # # special case arrays
# # # print np.zeros(5)
# # # print np.ones([5,5])
# # # print np.empty(5)
# # # print np.eye(5)
# #
# # print np.arange(5,50,2)
# #
# # #Lecture 02
# # #Arrays and Scalars
# # print 5/2
# # arr1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
# # print arr1 * arr1
# #
# # #Lecture 03
# #
# # #Indexing Array
# # arr = np.arange(0,11)
# # print arr
# # print arr[8]
# # print arr[1:5]
# # print arr[0:5]
# #
# # arr[0:5] = 100
# # print arr
# # arr = np.arange(0,11)
# #
# # # slice_arr = arr[0:6]
# # # print slice_arr
# # #
# # # arr_copy = arr.copy()
# # # arr_copy[0:4] = 100
# # # print arr_copy
# # # print arr
# #
# # arr_2d = np.array([[5,10,15,20],[25,30,35,40],[45,50,55,60]])
# # # print arr_2d
# # # print arr_2d[1]
# #
# # print arr_2d[1][0]
# # print arr_2d[:2,1:]
# #
# # #fancy indexing
# # arr2d = np.zeros((10,10))
# # print arr2d
# # print arr2d.shape[1]
# # print arr2d[[2,4,6,8]]
#
# #Lecture 04
# # arr = np.arange(50).reshape((10,5))
# # # print arr
# # # print arr.T
# # # print np.dot(arr.T,arr)
# # arr3d = np.arange(50).reshape((5,5,2))
# # # print arr3d.transpose((1,0,2))
# # arr = np.array([[1,2,3]])
# # print arr.swapaxes(0,1)
#
# #Lecture 05
# # arr = np.arange(11)
# # print np.sqrt(arr)
# #
# # print np.exp(arr)
# #
# # A = np.random.randn(10)
# # B = np.random.randn(10)
# # #Binary Functions
# # np.add(A,B)
# # np.maximum(A,B)
#
# #Lecture 06
# #No Notes taken, refer python notebook
#
# #Lecture 07
# # points = np.arange(-5,5,0.01)
# # dx,dy = np.meshgrid(points,points)
# # z = (np.sin(dx)+np.sin(dy))
# # plt.imshow(z)
# # plt.show()
#
# A = np.array([1,2,3,4])
# B = np.array([100,200,300,400])
# condition = np.array([True,True,False,False])
#
# #Lecture 08
# #Array Input and Output
#
# #How to save array and access them
#
# arr = np.arange(5)
# print arr
#
# np.save('myarray',arr)
# print np.load('myarray.npy')
#
# arr1 = np.load('myarray.npy')
# arr2 = np.arange(10)
#
# np.savez('ziparray.npz',x=arr1,y=arr2)
# archive_array = np.load('ziparray.npz')
#
# arr = np.array([[1,2,3],[4,5,6]])
# print arr
#
# np.savetxt('mytextarray.txt',arr,delimiter=',')
# arr = np.loadtxt('mytextarray.txt',delimiter=',')
# print arr