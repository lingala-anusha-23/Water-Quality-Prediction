# Water Quality Prediction Project 📊💧

## Overview 🌍
This project focuses on predicting water quality parameters using machine learning techniques. The dataset contains various water quality measurements such as NH4, BSK5, Suspended particles, O2, NO3, NO2, SO4, PO4, and CL, collected over multiple years from different locations. The goal is to build a predictive model to forecast these parameters, aiding in water quality monitoring and management.

---

## Dataset 📂
The dataset includes the following columns:
- **id**: Unique identifier for the location 📍
- **date**: Date of the measurement 📅
- **NH4**: Ammonium concentration 🧪
- **BSK5**: Biochemical oxygen demand in 5 days 🔬
- **Suspended**: Suspended particles in water 💧
- **O2**: Oxygen concentration 🌬️
- **NO3**: Nitrate concentration 🧪
- **NO2**: Nitrite concentration 🧪
- **SO4**: Sulfate concentration 🧪
- **PO4**: Phosphate concentration 🧪
- **CL**: Chloride concentration 🧪
- **year**: Year of measurement (derived from `date`) 📅
- **month**: Month of measurement (derived from `date`) 📅

### Dataset Statistics 📊
- **Total Rows**: 2861
- **Columns**: 13 (including derived features like `year` and `month`)
- **Missing Values**: Handled by dropping rows with missing target values.

---

## Project Steps 🛠️

### 1. Installation of Libraries 📦
The necessary libraries for data manipulation and machine learning are installed:
```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Data Loading and Exploration 🔍
- The dataset is loaded using `pandas`.
- Basic exploration includes checking the structure, missing values, and statistical summaries.
- **Key Observations**:
  - The dataset spans from 2000 to 2021.
  - Some columns like `SO4` and `CL` have higher missing value counts.

### 3. Data Preprocessing 🧹
- **Date Conversion**: The `date` column is converted to a datetime format.
- **Feature Engineering**: New features `year` and `month` are extracted from the `date` column.
- **Handling Missing Values**: Rows with missing target values (`O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL`) are dropped.

### 4. Model Building 🤖
- **Multi-output Regression**: Used to predict multiple water quality parameters simultaneously.
- **Random Forest Regressor**: A robust model for handling non-linear relationships in the data.
- **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets for model evaluation.
- **Feature Encoding**: The `id` column is one-hot encoded to handle categorical data.

### 5. Model Evaluation 📉
- **Metrics**: Mean Squared Error (MSE) and R-squared (R²) are used to evaluate model performance.
- **Performance Summary**:
  - **O2**: MSE = 22.22, R² = -0.02
  - **NO3**: MSE = 18.15, R² = 0.52
  - **NO2**: MSE = 10.61, R² = -78.42
  - **SO4**: MSE = 2412.14, R² = 0.41
  - **PO4**: MSE = 0.38, R² = 0.32
  - **CL**: MSE = 34882.81, R² = 0.74

### 6. Prediction Example 🔮
The model predicts pollutant levels for a given station and year. For example:
```python
Predicted pollutant levels for station '22' in 2024:
  O2: 12.60
  NO3: 6.90
  NO2: 0.13
  SO4: 143.08
  PO4: 0.50
  CL: 67.33
```

### 7. Model Saving 💾
The trained model and feature columns are saved for future use:
```python
joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")
```

---

## Key Features ✨
- **Multi-output Prediction**: Predicts multiple water quality parameters at once.
- **Time-based Features**: Utilizes `year` to capture temporal trends.
- **Robust Model**: Random Forest handles non-linearity effectively.
- **Scalability**: The model can be extended to include more features or stations.

---

## Future Improvements 🚀
- **Feature Selection**: Identify the most important features for prediction.
- **Hyperparameter Tuning**: Optimize model parameters for better performance.
- **Error Analysis**: Investigate poor performance for certain pollutants (e.g., `NO2`).
- **Real-time Prediction**: Deploy the model for real-time water quality monitoring.

---

## How to Use 🏁
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
3. Run the Jupyter notebook `Water_Quality_Prediction.ipynb` to preprocess the data, train the model, and evaluate performance.
4. Use the saved model (`pollution_model.pkl`) for predictions on new data.

---

## Dependencies 🧩
- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

--- 
