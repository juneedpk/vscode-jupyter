from flask import Flask, render_template
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load example dataset from seaborn
    iris = sns.load_dataset('iris')

    # Drop non-numeric columns
    numeric_columns = iris.select_dtypes(include=[float, int]).columns
    iris_numeric = iris[numeric_columns]

    # Perform basic EDA
    eda_plots = []

    # Pairplot without using 'species' as hue
    pairplot = sns.pairplot(iris_numeric, diag_kind='kde')
    plt.title('Pairplot')
    plt.tight_layout()
    pairplot_img = io.BytesIO()
    plt.savefig(pairplot_img, format='png')
    pairplot_img.seek(0)
    pairplot_base64 = base64.b64encode(pairplot_img.getvalue()).decode()
    eda_plots.append(pairplot_base64)
    plt.close()

    # Correlation heatmap
    corr = iris_numeric.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    corr_img = io.BytesIO()
    plt.savefig(corr_img, format='png')
    corr_img.seek(0)
    corr_base64 = base64.b64encode(corr_img.getvalue()).decode()
    eda_plots.append(corr_base64)
    plt.close()

    return render_template('index.html', eda_plots=eda_plots)

if __name__ == "__main__":
    app.run(debug=True)




