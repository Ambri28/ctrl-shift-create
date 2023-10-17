#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[9]:


pwd


# In[12]:


df=pd.read_csv("OSHack_BankCustomers.csv")


# In[13]:


df.head


# In[14]:


df.describe()


# In[15]:


df.columns


# In[16]:


import seaborn as sns
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

sns.histplot(df['age'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

sns.countplot(data=df, x='job', ax=axes[0, 1])
axes[0, 1].set_title('Job Distribution')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='marital', ax=axes[0, 2])
axes[0, 2].set_title('Marital Status Distribution')

sns.countplot(data=df, x='education', ax=axes[1, 0])
axes[1, 0].set_title('Education Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='default', ax=axes[1, 1])
axes[1, 1].set_title('Default Distribution')

sns.histplot(df['balance'], kde=True, ax=axes[1, 2])
axes[1, 2].set_title('Balance Distribution')

sns.countplot(data=df, x='housing', ax=axes[2, 0])
axes[2, 0].set_title('Housing Loan Distribution')

sns.countplot(data=df, x='loan', ax=axes[2, 1])
axes[2, 1].set_title('Personal Loan Distribution')

sns.countplot(data=df, x='contact', ax=axes[2, 2])
axes[2, 2].set_title('Contact Method Distribution')
axes[2, 2].tick_params(axis='x', rotation=45)


plt.tight_layout()

plt.show()


# In[17]:


# Set up a single subplot for the scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create the scatter plot with labels and regression line
sns.scatterplot(data=df, x='age', y='balance', ax=ax, alpha=0.5)
sns.regplot(data=df, x='age', y='balance', ax=ax, scatter=False, color='red')

# Define the age intervals
start_age = 10
interval_width = 5
max_age = df['age'].max()
age_intervals = np.arange(start_age, max_age + interval_width, interval_width)

# Set x-axis labels to age intervals
ax.set_xticks(age_intervals)
ax.set_xticklabels([f"{age}-{age+interval_width}" for age in age_intervals])

# Customize the plot appearance
ax.set_title('Age vs. Balance Scatter Plot')
ax.set_xlabel('Age Intervals')
ax.set_ylabel('Balance')
ax.legend(['Regression Line'])

# Improve plot aesthetics
sns.set(style="whitegrid")  # Use a white grid background

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




