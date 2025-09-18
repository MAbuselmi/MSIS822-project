# visualization.py
# Functions for generating plots and visualizations

import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    """Bar chart of Human vs AI distribution."""
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution (Human=0, AI=1)")
    plt.show()
