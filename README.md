# Laptop-Price-Prediction

Laptop Price Prediction 💻
This project uses machine learning to predict the price of laptops based on their various hardware and software specifications. The entire data analysis, feature engineering, and modeling process is contained within the Jupyter Notebook.

The final model is a Stacking Regressor that achieves an R2 Score of 0.888 and a Mean Absolute Error (MAE) of 0.160 on the test set, demonstrating high accuracy.

## 🚀 Project Workflow
This project follows a complete data science pipeline:

#### Data Loading & Cleaning: 
Loaded the laptop_data.csv dataset and performed initial cleaning by removing units (GB, kg) and converting columns to their correct numerical types.

#### Feature Engineering:
This was the most critical step. Raw string-based data was transformed into valuable features for the model:

-- Screen: Extracted Touchscreen and IPS as binary features.

-- PPI (Pixels Per Inch): Engineered a new feature for screen density using the formula: ppi=
fracsqrtX_res 
2
 +Y_res 
2
 Inches.

-- CPU: Simplified processor information into key brands (e.g., 'Intel Core i7', 'AMD Processor').

-- Memory: Split the Memory column into four distinct columns: HDD, SSD, Hybrid, and Flash_Storage.

-- OS: Grouped operating systems into three categories: Windows, Mac, and Others/No OS/Linux.

### Exploratory Data Analysis (EDA):
Visualized the data to find patterns. Key insights include a strong right-skew in the Price (fixed with a log transform) and strong positive correlations between price and RAM, SSD, and PPI.

### Modeling: 
Trained and evaluated eight different regression models to find the best performer.

### Tuning & Stacking:
Fine-tuned the best model (XGBoost) using GridSearchCV and then combined the strongest models into a StackingRegressor to achieve the best possible performance.

## 🛠️ Tech Stack & Libraries
Data Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (for preprocessing, pipelines, and all models), XGBoost (for the XGBoost Regressor)

### 📈 Model Performance Comparison
Multiple regression models were trained using a scikit-learn pipeline for preprocessing. The Price column was log-transformed to normalize its distribution, so the MAE is relative to the log value.

| Model               | R² Score | Mean Absolute Error (MAE) |
|----------------------|----------|----------------------------|
| Linear Regression    | 0.807    | 0.210                      |
| K-Nearest Neighbors  | 0.812    | 0.188                      |
| Decision Tree        | 0.835    | 0.184                      |
| SVM                  | 0.811    | 0.201                      |
| Random Forest        | 0.887    | 0.157                      |
| Gradient Boosting    | 0.879    | 0.160                      |
| Extra Trees          | 0.884    | 0.157                      |
| XGBoost (Tuned)      | 0.888    | 0.156                      |

Final Model: Stacking Regressor
By combining the top-performing ensemble models (Random Forest, GBDT, XGBoost, and Extra Trees), the final stacked model achieved the highest accuracy.

#### Final R2 Score: 0.888

#### Final MAE: 0.160

## ⚙️ How to Run this Project
1. Clone this repository:


     git clone https://github.com/adityagoswami004/Laptop-Price-Prediction.git
     cd Laptop-Price-Prediction


2. Install dependencies:

     pip install -r requirements.txt

`
3. Open the Jupyter Notebook:

    jupyter notebook Laptop_Price_prediction.ipynb

## 📊 Future Work

Deploy the model using Streamlit/Flask for real-time predictions.

Expand dataset with newer laptop models for better accuracy.
