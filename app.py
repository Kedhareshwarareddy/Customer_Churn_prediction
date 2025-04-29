from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from werkzeug.utils import secure_filename
import os
import numpy as np  # Make sure this import is at the top
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import missingno as msno

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def predict_churn(data):
    # Preprocess the data similar to your notebook
    if 'customerID' in data.columns:
        data = data.drop(['customerID'], axis=1)
    
    # Handle categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Convert data to numpy array for model
    X = np.array(data)  # Properly convert to numpy array
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create dummy target variable (all zeros for demonstration)
    y = np.zeros(len(data))
    
    # Fit and predict
    predictions = model.fit(scaled_features, y).predict(scaled_features)
    
    return predictions

def create_plots(data):
    plots = []
    
    # Cell 1: Import Libraries and Load Data
    plt.figure(figsize=(12, 4))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    cell_text = """# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('customer_churn.csv')"""
    ax.text(0.1, 0.5, cell_text, fontsize=10, family='monospace')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 1: Import Libraries", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 2: Display first few rows
    plt.figure(figsize=(15, 8))
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data.head().values,
                    colLabels=data.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('data.head()')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 2: Data Head", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 3: Data Info
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    info_text = f"data.info()\n\n{data.info(buf=io.StringIO())}\n\ndata.shape: {data.shape}"
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 3: Data Info", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 4: Missing Values Analysis
    plt.figure(figsize=(12, 6))
    msno.matrix(data)
    plt.title('Missing Values Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 4: Missing Values Analysis", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 5: Statistical Summary
    plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    stats_text = f"data.describe()\n\n{data.describe().to_string()}"
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 5: Statistical Summary", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 6: Churn Distribution
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.countplot(data=data, x='Churn', ax=ax1)
    ax1.set_title('Churn Distribution')
    
    churn_pct = data['Churn'].value_counts(normalize=True) * 100
    ax2.pie(churn_pct, labels=[f'No Churn ({churn_pct[0]:.1f}%)', 
                              f'Churn ({churn_pct[1]:.1f}%)'],
            autopct='%1.1f%%')
    ax2.set_title('Churn Percentage')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 6: Churn Analysis", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 7: Correlation Analysis
    plt.figure(figsize=(15, 12))
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots.append(("Cell 7: Correlation Analysis", base64.b64encode(buf.read()).decode('utf-8')))
    plt.close()

    # Cell 8: Numerical Features Analysis
    for col in numeric_cols:
        plt.figure(figsize=(15, 6))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Distribution plot
        sns.histplot(data=data, x=col, ax=ax1)
        ax1.set_title(f'{col} Distribution')
        
        # Box plot
        sns.boxplot(data=data, y=col, ax=ax2)
        ax2.set_title(f'{col} Box Plot')
        
        # Box plot by churn
        sns.boxplot(data=data, x='Churn', y=col, ax=ax3)
        ax3.set_title(f'{col} by Churn Status')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots.append((f"Cell 8: {col} Analysis", base64.b64encode(buf.read()).decode('utf-8')))
        plt.close()

    # Cell 9: Categorical Features Analysis
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'customerID':
            plt.figure(figsize=(15, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # Count plot
            sns.countplot(data=data, x=col, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_title(f'{col} Distribution')
            
            # Count plot by churn
            sns.countplot(data=data, x=col, hue='Churn', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_title(f'{col} by Churn Status')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots.append((f"Cell 9: {col} Analysis", base64.b64encode(buf.read()).decode('utf-8')))
            plt.close()
    
    return plots

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and process the data
        data = pd.read_csv(filepath)
        
        # Generate data info summary
        buffer = io.StringIO()
        data.info(buf=buffer)
        data_info = buffer.getvalue()
        
        # Generate detailed summary
        summary = {
            "total_customers": len(data),
            "churn_rate": f"{(data['Churn'].value_counts(normalize=True)[1] * 100):.2f}%",
            "avg_monthly_charges": f"${data['MonthlyCharges'].mean():.2f}",
            "total_revenue": f"${data['MonthlyCharges'].sum():.2f}",
            "data_info": data_info,
            "missing_values": data.isnull().sum().to_dict(),
            "numerical_summary": data.describe().to_dict(),
            "categorical_summary": {col: data[col].value_counts().to_dict() 
                                  for col in data.select_dtypes(include=['object']).columns}
        }
        
        plots = create_plots(data)
        predictions = predict_churn(data)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'summary': summary
        })
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port to 5001