import numpy as np
x=[]
yi=[]
arr=[]
y=[]
# enter the length of the array
n=int(input("enter the array size "))
#entering the elements in column x
print("enter element in x ")
for i in range(n):
    q=int(input())
    arr.append(q)
#entering the elements in column y
print("enter element in y ")
for i in range(n):
    r=int(input())
    y.append(r)
#finding mean of the elements in array x
arrbar=np.mean(arr)
#finding mean of the elements in array y
ybar=np.mean(y)
#making an array of elements (xi-xbar)
for i in range(n):
    x.append(arr[i]-arrbar)
#making an array of elements (yi-ybar)
for j in range(n):
    yi.append(y[j]-ybar)
#squaring the elements (xi-xbar)
xx=np.square(x)
#finding the sum of all the elements (xi-xbar)square
xis=sum(xx)
#multiplying (xi-xbar) and (yi-ybar)
xy=np.multiply(x,yi)
#finding the summation of (xi-xbar)*(yi-ybar)
xys=sum(xy)
#calculating the slope
ans=xys/xis
#calculating the y intercept
www=ybar-ans*arrbar
print("The slope is")
print(ans)
print("The intercept is")
print(www)
#printing the equation
print('Y'+ '='+ str(ans)+'X' +'+'+ str(www))
#printing the predicted value of y for a x
zz=int(input("enter 1 if you want to predict a value for Y "))
if(zz==1):
    zzz=int(input("enter value for X "))
    print(ans*zzz + www)
###################################################################################################################################################
###################################################################################################################################################




######################################################################################################################################################
#####################################################################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[3]:


# get_ipython().system('pip install -U scikit-learn')


# In[7]:


import numpy as np

from sklearn.linear_model import LinearRegression
import pandas as pd


# In[9]:


x=np.array([5,15,25,35,45,55])
y=np.array([5,20,14,32,22,38])
print(x)
print(y)


# In[12]:


x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
print(x)


# In[13]:


print(y)


# In[14]:


model=LinearRegression()
model.fit(x,y)


# In[15]:


r_sq=model.score(x,y)
print('coefficient of determination:',r_sq)


# In[17]:


print('intercept:',model.intercept_)


# In[18]:


print('slope',model.coef_)


# In[19]:


new_model=LinearRegression().fit(x,y.reshape((-1,1)))
print('intercept:',new_model.intercept_)
print('slope:',new_model.coef_)


# In[20]:


y_pred=model.predict(x)
print('predicted response:',y_pred, sep='\n')
print(x)


# In[21]:


y_pred=model.intercept_ + model.coef_ *x
print('predicted response:',y_pred,sep='\n')


# In[28]:


x_new= np.arange(5).reshape((-1,1))
print(x_new)
y_new=model.predict(x_new)
print(y_new)


# In[ ]:




