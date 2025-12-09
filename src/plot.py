import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
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
        plt.title(f"{X} Change vs {Y} Change")
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
    def time_serice_plot(self, M_data, ClaimFrequency,ClaimSeverity, month):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{ClaimFrequency}', color=color)
        ax1.plot(M_data[month].astype(str), M_data[ClaimFrequency], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(M_data[month].astype(str), rotation=45)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel(f'{ClaimSeverity} (Avg. Claim Amt)', color=color)
        ax2.plot(M_data[month].astype(str), M_data[ClaimSeverity], color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Temporal Trend: {ClaimFrequency} vs. {ClaimSeverity}')
        fig.tight_layout() 
        plt.show()
    def plot_metrics(self,models, mae_scores, mse_scores, r2_scores):
        """
        Visualizes model performance using bar charts.
        Helps compare:
        - MAE accuracy
        - MSE error
        - R² fit score
        """

        import matplotlib.pyplot as plt

        # ---------- Plot MAE ----------
        plt.figure(figsize=(6, 4))
        plt.bar(models, mae_scores, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Comparison of MAE Scores')
        plt.xticks(rotation=45)
        plt.show()

        # ---------- Plot MSE ----------
        plt.figure(figsize=(6, 4))
        plt.bar(models, mse_scores, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Comparison of MSE Scores')
        plt.xticks(rotation=45)
        plt.show()

        # ---------- Plot R² ----------
        plt.figure(figsize=(6, 4))
        plt.bar(models, r2_scores, color='salmon')
        plt.xlabel('Models')
        plt.ylabel('R-squared Score')
        plt.title('Comparison of R-squared Scores')
        plt.xticks(rotation=45)
        plt.show()
    def tree_plot(self,model, feature):
        plt.figure(figsize=(20, 10))
        plot_tree(decision_tree=model, feature_names=feature, filled=True, rounded=True)
        plt.show()
    def plot_feature_importance(self,model, feature_names, model_name):
        n = len(feature_names)
        feature_importance = pd.DataFrame(model.feature_importances_[:n], index=feature_names, columns=["Importance"])
        feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar', legend=False, color='skyblue')
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.show()
