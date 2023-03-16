# import ast
# import matplotlib.pyplot as plt
# import numpy
# import  numpy as np
# from matplotlib.colors import ListedColormap
#
#
#
#
# print(np.char.add(['sb'],['吧']))
# print('\n-----\n')
# print(np.char.add(['sb','uu'],['ba','men']))
#
#
# print(np.char.multiply(('sb'),2))
#
# print(np.char.center('sb 吧,uu们',20,fillchar='-'))
# print('\n-----\n')
# print (np.char.capitalize('runoob'))
#
# print (np.char.split ('sb 吧,uu们', sep = ','))
# print('\r')
#
# print('1'
#       '2')
# print('1\r2')
#
# print (np.char.strip('aassbaba','a'))
#
# print (np.char.replace ('i like uu', 'uu', 'sb'))
#
import random

# x=np.arange(1,5).reshape(2,2)
# print(x)
#
# print(x.shape)
# print(x.ndim)
# print('-------')
#
# y=np.expand_dims(x,axis=2)
# print(y)
# print(y.shape)
# print(y.ndim)
# print('-------')
#
# y=np.expand_dims(x,axis=0)
# print(y)
# print(y.shape)
# print(y.ndim)
# print('-------')
#
# x=np.arange(1,10).reshape(3,3,1)
# print(x)
# print('-----------')
# y=np.squeeze(x)
# print(y)
# print(y.shape)
# print('-----------')
# y=np.expand_dims(y,axis=2)
# print(y)
# print(y.shape)
# print(y.ndim)
#
#
#
#
#
# a=np.arange(12)
# b=np.split(a,[5,7])
# print(b)
# print('\n')
# print('-----------')
#
#
# x=np.arange(24).reshape(6,4)
# print(x)
# print('\n')
# print('-----------')
# # a=np.split(x,2,axis=0)
# # print(a)
#
# print(np.vsplit(x,3))


#
# x=np.arange(6).reshape(3,2)
# print(x)
# print('\n')
# print('-----------')
# a=np.insert(x,1,[11],axis=0)
# b=np.insert(x,1,[[1],[2]],axis=0)
# print(a)
# print(b)



# b=np.append(x,[[5,5],[4,4]],axis=1)
# print(b)

# print('-----------')
# b=[[[1,2,3],[5,6,7]]]
# b=np.asarray(b)
# print(b.shape)
# print('\n')
# print('-----------')
# a=np.append(x,b,axis=1)
# print(a)
# print(a.shape)
# print('\n')


# y=np.resize(x,(2,3))
# print(y)
# print('\n')
# z=np.resize(x,(3,3))
# print(z)

















#
#
# a=np.arange(12).reshape(3,4)
# print(a)
# print('-------------')
# print('\n')
#
# b=a.ravel(order='f')
# print(b)
# b[1]=100
# print(b)
# print(a)
#



import numpy as np
# x=np.array([[1],[2],[3]])
# print(x)
#
#
# y=np.array([4,5,6])
# print(y)
#
#
#
# b=np.broadcast(x,y)
# print(b.shape)
# r,c=b.iters
#
# k=np.empty(b.shape)
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# c=np.empty(b.shape)
# print(c)
# print('--------')
# print(x+y)

#
# a=np.arange(1,8,1)
# print(a)
# a=np.arange(1,5).reshape(1,4)
# print(a)
# print('----------')
# b=np.broadcast_to(a,(5,4))
# print(b)

# import numpy as np
# a=np.arange(12).reshape(4,3,1)
# print(a)
# print(a.shape)
# print('\n')
# print('----------')
#
# b=np.rollaxis(a,2)
# print(b)
# print(b.shape)
# print('\n')
# print('----------')
#
# b=np.swapaxes(a,0,2)
# print(b)
# print(b.shape)
#
#
#





# a=np.arange(6).reshape(2,3)
# print(a)
# print('\n')
#
# for x in np.nditer(a):
#     print(x,end=', ')
#
# print('\n')
#
#
# b=a.T
# print(b)
# for x in np.nditer(b):
#     print(x,end=', ')
#
# print('\n--------------\n')
# for x in np.nditer(a.T.copy(order='F')):
#     print(x,end=', ')
#
# print('\n--------------\n')
# for x in np.nditer(a.T.copy(order='C')):
#     print(x,end=', ')
#
#
#

#
# x=np.arange(12).reshape(3,4)
# print(x)
# print('\n--------------\n')
#
# # for x in np.nditer(x,order='C'):
# #     print(x,end=',')
# # print('\n--------------\n')
# # for x in np.nditer(x,order='F'):
# #     print(x,end=',')
# # print('\n--------------\n')
# #
# #
# c=x.copy(order='F')
# print(c)
# print('\n--------------\n')
# for x in np.nditer(c):
#     print(x,end=',')
# print('\n--------------\n')
#
#
#


# a=np.arange(12).reshape(3,4,)
# print(a)
# print('---------------')
# # # for x in np.nditer(a,order='c',op_flags=['readwrite']):
# # #     x[...]=x*5
# # # print('\n')
# # # print(a)
# # for x in np.nditer(a,order='f'):
# #     print(x,end=',')
# # print('')
#
#
# b=np.arange(1,5)
# print(b)
# print([a,b])




# b=x.T
# print(b)
# print('\n--------------\n')
# for x in np.nditer(b):
#     print(x,end=',')
# print('\n--------------\n')
#


# c=x.copy(order='C')
# print(c)
#
# print('\n--------------\n')
#
#
#
#
#




# # a=np.arange(10)
# # s=slice(2,9,2)
# # print(a[s])
# #
# #
# #
# # b=a[5:7]
# # print(b)
# #
# # x=[(1,2,3),(4,5,6)]
# # a=np.asarray(x)
# # print(a)
# x=np.arange(24)
# print(x)
# b=x.reshape(2,4,3)
# print(b)
# print('数组维度为：',b.shape)
# print('维度为',b.ndim)
# print(b.size)
#
#
# x=np.array([1])
# print(x[x>1])
# print(x.shape)
# print(x.ndim)
#
#
# a = np.array([[ 0, 0, 0],
#            [10,10,10],
#            [20,20,20],
#            [30,30,30]])
# print(a.shape)
# print(a.ndim)



# a=np.arange(12).reshape(2,2,3)
# b=np.arange(12).reshape(3,1,4)
# c=np.arange(12).reshape(1,6,2)
# print(a)
# print('\n')
# print(b)
# print('\n')
# print(c)
#











#
# # arr=np.empty([3,2],'f')
# # print(arr)
# #
# #
# # arr=np.zeros([3,3],dtype=[('x','f'),('y','i4')])
# #
# # print(arr)
# #
# # arr=np.ones(3,dtype='i4')
# # print(arr)
# #
# #
# #
# # arr=np.full([3,2],fill_value=3)
# # print(arr)
# # arr=np.eye(3,dtype='i4')
# # print(arr)
# #
# #
# # arr=np.arange(1,10,2)
# # print(arr)
#
#
# s=b'abcd'
# arr=numpy.frombuffer(s,dtype='S1',count=2,offset=2)
# print(arr)
#
#
# x=[1,2,3,4,5]
#
# z=iter(x)
# print(type(z))
# arr=np.fromiter(z,dtype='f')
# print(arr)
# arr=np.logspace(1,2,5,dtype='f',base=2)
# print(arr)
#
#
# s= b'hello\ world'
# arr=np.frombuffer(s,dtype='S1')
# print(arr)
# import numpy as np

# list=range(10)
# it=iter(list)
# x=np.fromiter(it,dtype='i4')
# print(x)

# import numpy as np
# x=np.arange(1,10,2,'f')
# print('长度为：',len(x))
# print(x)
#
# x=np.linspace(1,5,5,endpoint=False,retstep=True)
# print(x)






















#
# xp=np.dtype([('kiss','bool'),('doi','bool')])
# arr=np.array([(0,1)],dtype=xp)
# print(arr)
#
# arr=np.array([1,2,3],dtype='f')
# print(arr)
#
# x=([1,2,3],[4,5,6])
# arr=np.asarray(x,dtype='f',order='C')
# print(arr)
# #
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.style.use('_mpl-gallery')
#
# # make the data
# np.random.seed(3)
# x = 4 + np.random.normal(0, 2, 24)
# y = 4 + np.random.normal(0, 2, len(x))
# # size and color:
# sizes = np.random.uniform(15, 80, len(x))
# colors = np.random.uniform(15, 80, len(x))
#
# # plot
# fig, ax = plt.subplots()
#
# ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
#
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
#
# plt.show()
#
#
#
#
#
#
#







# import numpy as np
# import matplotlib.pyplot as plt
#
# # setup some generic data
# N = 37
# x, y = np.mgrid[:N, :N]
# Z = (np.cos(x*0.2) + np.sin(y*0.3))
#
# # mask out the negative and positive values, respectively
# Zpos = np.ma.masked_less(Z, 0)
# Zneg = np.ma.masked_greater(Z, 0)
#
# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
#
# # plot just the positive data and save the
# # color "mappable" object returned by ax1.imshow
# pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
#
# # add the colorbar using the figure's method,
# # telling which mappable we're talking about and
# # which axes object it should be near
# fig.colorbar(pos, ax=ax1)
#
# # repeat everything above for the negative data
# # you can specify location, anchor and shrink the colorbar
# neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
# fig.colorbar(neg, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.7)
#
# # Plot both positive and negative values between +/- 1.2
# pos_neg_clipped = ax3.imshow(Z, cmap='RdBu', vmin=-1.2, vmax=1.2,
#                              interpolation='none')
# # Add minorticks on the colorbar to make it easy to read the
# # values off the colorbar.
# cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
# cbar.minorticks_on()
# plt.show()
#









# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.arange(0.0, 2.0, 0.01)
# s = np.sin(2 * np.pi * t)
#
# # upper = 0.5
# # lower = -0.5
# #
# # supper = np.ma.masked_where(s < upper, s)
# # slower = np.ma.masked_where(s > lower, s)
# # smiddle = np.ma.masked_where((s < lower) | (s > upper), s)
# #
# # fig, ax = plt.subplots()
# # ax.plot(t, smiddle, t, slower, t, supper)
# # plt.show()
# # x=np.arange(0,10,0.01)
# # t=np.sin(x)
# #
# # upper=0.50
# # lower=-0.77
# #
# # tupper=np.ma.masked_where(t<upper,t)
# # tlower=np.ma.masked_where(t>lower,t)
# # tmiddle=np.ma.masked_where((t<lower)|(t>upper),t)
# # fig,ax=plt.subplots()
# # ax.plot(x,tmiddle,x,tlower,x,tupper)
# # plt.show()
# a=np.diag(range(100))
# print(a)
# fig=plt.figure(num=3)
# plt.matshow(a,fig)
# plt.show()
#
#
#
#
#
# '''
# x=np.linspace(-3,3,50)
# y1=2*x-1
# y2=x**2
# plt.figure(num=1)
# l1=plt.plot(x,y1,label='y1')
# l2=plt.plot(x,y2,color='blue',label='y2')
# plt.xlim((-1,2))
# plt.ylim((-3,3))
# plt.xlabel('x')
# plt.ylabel('y')
#
# ax=plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# ax.spines['bottom'].set_position(('data',0))
# ax.spines['left'].set_position(('data',0))
#
#
# plt.legend(loc='best')
#
#
# x0=1
# y0=2*x0-1
# plt.scatter(x0,y0)
# plt.plot([1,2],[2,3],'k--',lw=3)
#
#
#
#
# plt.figure()
# ax1=plt.subplot2grid((4,4),(0,0),colspan=4,rowspan=1)
# ax1.plot(x,y1)
# ax1.set_title('ax1')
# ax1.set_xlabel(r'$x$',color='g')
# ax1.set_xticks((-1,1,10))
# ax1=plt.gca()
# ax1.spines['right'].set_color=['b']
#
# ax2=plt.subplot2grid((4,4),(2,2),rowspan=2)
# ax2.plot(x,y2)
#
# fig=plt.figure(num=10)
# left=0.1
# bottom=0.2
# height=0.6
# width=0.6
# ax1=fig.add_axes([left,bottom,width,height])
# ax1.plot(x,y1,'r--')
# ax1.legend(labels='l1',loc='best')
# left=0.2
# bottom=0.7
# height=0.1
# width=0.1
# x3=[1,2,3,4,5]
# y3=[6,4,3,2,1]
# ax2=fig.add_axes([left,bottom,width,height])
# ax2.plot(x3,y3,'b-')
# plt.show()
#
# '''
#

import matplotlib.pyplot as plt
import  numpy as np


# fig,ax=plt.subplots()
# counts=[10,20,30,40]
# bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'red']
# name=['wxy','ljl','ldl','czm']
# bar_labels=['red','blue','green','pink']
# ax.bar(name,counts,label=bar_labels,color=bar_colors)
# ax.legend(title='uu',loc='upper left')
# #ax.legend(title='Fruit color')
# plt.show()
#


# species=('a','b',"c")
# width=1
# fig,ax=plt.subplots()
# bottom=np.zeros(3)
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('_mpl-gallery')
#
# # make data
# x = np.linspace(0, 10, 100)
# y = 4 + 2 * np.sin(2 * x)
#
# # plot
# fig, ax = plt.subplots()
#
# ax.plot(x, y, linewidth=2.0)
#
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
# mu, sigma = 0, 0.1 # mean and standard deviation
# s = np.random.normal(mu, sigma, 1000)
# k=np.asarray(s)
# print(k)
# plt.plot(k)
# plt.show()
# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()

#
# a=random.random()
# print(a)
# print('\n-------------')
# b=random.uniform(1,2)
# print(b)
# print('\n-------------')
# c=random.randint(3,19)
# print(c)
# print('\n-------------')
# d=random.randrange(4,100,5)
# print(d)
#
# k=[1,2,3]
# a=random.choice('学习sb')
# print(a)
# p=['sha','gou','ba']
# random.shuffle(p)
# print(p)
#
#
# slice=random.sample(p,2)
# print(slice)
#

import random

# x=np.random.rand()
# print(x)
# print('\n-----------')
# y=np.random.rand(4)
# print(y)
#
# plot Gaussian Function
# 注：正态分布也叫高斯分布
# import matplotlib.pyplot as plt
# import numpy as np
#
# u1 = 0  # 第一个高斯分布的均值
# sigma1 = 1  # 第一个高斯分布的标准差
#
# u2 = 1  # 第二个高斯分布的均值
# sigma2 = 2  # 第二个高斯分布的标准差
# x = np.arange(-5, 5, 0.1)
# # 表示第一个高斯分布函数
# y1 = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))
# # 表示第二个高斯分布函数
# plt.plot(x,y1)
# plt.show()
#
# !/usr/bin/python
# -*- coding: UTF-8 -*-

# list = ['runoob', 786, 2.23, 'john', 70.2]
# tinylist = [123, 'john']
#
#
# # print(list[1:3])
# tuple = ('runoob', 786, 2.23, 'john', 70.2)
# tinytuple = (123, 'john')
#
# print(tuple)
#
#
# a=str(input())
#
# b=str(input())
# print(a+b)
# import pandas as pd
# pd.set_option( 'display.max_columns', None)
# pd.set_option( 'display.max_rows', None)





import numpy as np
# M=np.random.randint(0,2,(32,32))
# print(M)
# for i in range(32):
#     M[i][i] = 0
#     for j in range(32):
#
#
# print(M)
#
# import  matplotlib.pyplot as plt
# a=np.random.randint(0,2,(32,32))
# print(a)
# for i in range(32):
#     a[i][i]=0
#
# for i in range(32):
#     for j in range(32):
#         a[i][j]=a[j][i]
# print(a)
#
# b=np.cov(a)
# print(b)
#
# plt.matshow(b,cmap=plt.cm.gray)
#
#
#
#




















fig,axes=plt.subplots(2,2,figsize=(5,6))

data_set=np.random.randint(0,2,32)
print(data_set)
m=np.random.randint(0,2,(32,32))
for i in range(32): m[i][i]=0

for i in range(32):
    for j in range(32):
        if(data_set[j]==1 and i!=j and m[i][j]==1):
           m[i][j]=m[j][i]
M=np.cov(m)

axes[0,0].matshow(m,cmap=plt.cm.gray)
axes[0,0].set_title('$Random:Adj Matrix$')
axes[1,0].matshow(M,cmap=plt.cm.gray)
axes[1,0].set_title('$Random:Cov Matrix$')

a=np.zeros((32,32),dtype='int')
#16-24的点->下表为15-23 23不显示所以24;1->16:0->15  0>16
for j in range(15,24):
    for k in range(15,24):
        if(j!=k):a[j][k]=a[k][j]=1

#index=0->15 表示前16个点
for i in range(15):
    for j in range(15,24):
        a[i][j]=1
for i in range(24,32):
    for j in range(15,24):
        a[i][j]=1
b=np.cov(a)
axes[0,1].matshow(a,cmap=plt.cm.gray)

axes[0,1].set_title('$Intergaration:Adj Matrix$')
#a0=
axes[1,1].matshow(b,cmap=plt.cm.gray)
axes[1,1].set_title('$Intergaration:Cov Matrix$')
#cbar1=fig.colorbar(a0)
plt.show()
