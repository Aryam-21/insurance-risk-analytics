import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class Plot:
    def __init__(self):
        pass
    def univariate_numerical(self,data, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=50, kde=True, log_scale=True) # Log scale helps with highly skewed data
        plt.title(f'Distribution of {column} (Lognormal-like)')
        plt.xlabel(f'{column} (Log Scale)')
        plt.show()
    def univariate_categorical(self, data, column):
        plt.figure(figsize=(10, 5))
        data[column].value_counts().head(20).plot(kind='bar')
        plt.title(f"Category Counts: {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
    def scatter_plot(self, mdata, X, Y):
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=mdata, x=X, y=Y, alpha=0.6)
        plt.title("{X} Change vs {Y} Change")
        plt.show()
    def heat_plot(self, d_corr):
       
        mask = np.zeros_like(d_corr)
        up_tri = np.triu_indices_from(mask)
        mask[up_tri] = True
        sns.heatmap(data=d_corr, mask=mask, annot=True)
        plt.show()
    def box_plot(self, numeric_cols, data):
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(1, len(numeric_cols), i)
            sns.boxplot(y=data[col])
            plt.title(col)
        plt.tight_layout()
        plt.show()