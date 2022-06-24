# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:39 2022

@author: KTong
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Visualisation():
    def __init__(self):
        pass
    
    def single_cont_plot(self,x):
        '''
        Generates plot for a single numeric attribute. 

        Parameters
        ----------
        x : array like
            An array of numeric feature.

        Returns
        -------
        seaborn.displot().

        '''
        plt.figure()
        sns.distplot(x)
        plt.show()

    
    def cont_plot(self,df,num_features):
        '''
        Creates plots for numerical data.

        Parameters
        ----------
        df : pandas Dataframe
            Dataset.
        num_features : list
            Column names of numerical features.

        Returns
        -------
        seaborn.distplot().

        '''
        for i in num_features:
            plt.figure()
            sns.distplot(df[i])
            plt.show()

    def cat_plot(self,df,cat_features):
        '''
        Creates plots for categorical data.

        Parameters
        ----------
        df : pandas Dataframe
            Dataset.
        cat_features : list
            Column names of categorical features.

        Returns
        -------
        seaborn.countplot().

        '''
        for i in cat_features:
            plt.figure(figsize=(12,10))
            sns.countplot(df[i])
            plt.show()

    def cat_group_plot(self,df,cat_features,target):
        '''
        Creates plots for categorical data base on group by target feature.

        Parameters
        ----------
        df : pandas Dataframe
            Dataset.
        cat_features : list
            Column names of categorical features.
        target : string
            Target feature.

        Returns
        -------
        seaborn.countplot(,hue=).

        '''
        for i in cat_features:
            plt.figure(figsize=(12,10))
            sns.countplot(df[i],hue=(df[target]))
            plt.show()

    def line_plot(self,df,xlen,column,linewidth):
        '''
        Generates a line plot.

        Parameters
        ----------
        df : pandas Dataframe
            Dataframe.
        xlen : int
            Length of x-axis.
        column : str
            Column name.
        linewidth : int
            Width of line plot.

        Returns
        -------
        matplotlib.pyplot.plot().

        '''
        plt.figure()
        plt.plot(np.arange(0,xlen,1),df[column],linewidth=1)
        plt.show()
