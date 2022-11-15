#!/usr/bin/env python
# coding: utf-8

# In[336]:


#import standard libraries 
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


# In[337]:


import requests as r


# In[338]:


# create the link
wiki_page = r.get('https://en.wikipedia.org/wiki/Production_car_speed_record')
wiki_text = wiki_page.text


# In[339]:


# get BeautifulSoup
from bs4 import BeautifulSoup


# In[340]:


# parse the html with BeautifulSoup
wiki_soup = BeautifulSoup(wiki_text, 'html.parser')
type(wiki_soup)


# In[341]:


# create the table from the html reader
wiksoup_tab = wiki_soup.find_all('table')
soup = pd.read_html(str(wiksoup_tab))[0]


# In[342]:


# check that it was read in properly
soup.head()


# In[343]:


#split the data on the code for the space coding, then only take the first instance of the split
split = soup['Top speed'].str.split("\xa0", n = 1, expand = False)

for i in range(0, len(split)):
    split[i] = split[i][0]
    pass
# row 15 didn't read properly so had to hard code that one
split[15] = 412.22
soup['Top speed'] = split


# In[344]:


# check the dataframe
soup


# In[345]:


for i in range(0, len(soup)):
    soup['Top speed'][i] = int(float(soup['Top speed'][i]))
    pass
print(soup)


# In[346]:


# drop engine and comment
soup = soup.drop(columns=['Engine', 'Comment'])
print(soup)


# In[356]:


soup.head(5)


# In[347]:


#import linear modeling
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[348]:


X_train, X_test, y_train, y_test = train_test_split(soup['Year'], soup['Top speed'])


# In[349]:


# create the model
lr = LinearRegression()
lr.fit(X_train.values.reshape(-1,1), y_train)


# In[350]:


predict = lr.predict(X_test.values.reshape(-1,1))
X = soup['Year']
y = soup['Top speed']

# In[351]:


# graph the plot
plt.scatter(X,y)
plt.plot(X_test, predict, label = 'Linear Regression', color = 'r')
plt.xlabel('Year')
plt.ylabel('Top Speed')
plt.title('Top Speed vs. Year')


# In[352]:


r2 = lr.score(X_test.values.reshape(-1,1), y_test.values)


# In[353]:


print(f'The coefficient of determination is {r2}, meaning, {r2 * 100} percent of the variation in Y can be attributed to X.')


# In[354]:


coeff = lr.coef_
intercept = lr.intercept_


# In[355]:


print(f'The equation for the line of best fit is: Y = {coeff}(X) + {intercept}.')


# In[ ]:




