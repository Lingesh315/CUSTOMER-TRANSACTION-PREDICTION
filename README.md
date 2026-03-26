Bank Customer Transaction Prediction
Project Overview:

This project aims to predict whether a bank customer will make a transaction in the future.
The dataset contains 200 anonymized numerical features, along with:

ID_code → Unique customer ID
target :
0 → No transaction
1 → Will make a transaction

Objective:

Build a machine learning model that can accurately identify customers who are likely to make future transactions.


Project Workflow (Step-by-Step)
Data Collection:

  * Loaded the dataset using Pandas.
  * Verified that the data is correctly imported.


Basic Data Checks:

  * Checked dataset shape (rows & columns)
  * Viewed sample data using .head()
  * Checked data types of columns
  * Looked for missing values
  * Checked duplicate records
  * Analyzed target variable distribution

Problem Understanding:

  * This is a binary classification problem
  * Goal: Predict whether a customer will transact (0 or 1)
  * Since features are anonymized, domain-based analysis is limited


Feature Engineering & Preprocessing:

  * Removed unnecessary column (ID_code)
  * Separated features (X) and target (y)
  * Handled missing values using SimpleImputer
  * Scaled features using StandardScaler
  * Reduced dimensionality using SelectKBest (Mutual Information)


Train-Test Split:

  * Split data into training and testing sets (70:30)
  * Used stratified sampling to maintain class balance

Model Building:

Trained multiple machine learning models:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * K-Nearest Neighbors (KNN)


Model Evaluation:

Evaluated models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix
Used cross-validation for reliable comparison.


Model Selection:

  * Compared all models based on performance metrics
  * Selected the best-performing model (based on ROC-AUC and F1-score)


Final Model Training:

  * Trained the best model on full training data
  * Tested performance on unseen test data


Model Saving:

  * Saved the final model using Joblib
  * This allows reuse without retraining


Results:

  * Built a robust model to predict customer transactions
  * Achieved good performance using evaluation metrics like ROC-AUC and F1-score
  * Successfully identified potential customers for future transactions


Challenges Faced:

  * Features were anonymized (no domain meaning)
  * High number of features (200) → risk of overfitting
  * Possible class imbalance


Solutions:

  * Used feature selection to reduce dimensions
  * Used cross-validation for reliable evaluation
  * Focused on ROC-AUC and F1-score instead of only accuracy


Future Improvements:

  * Hyperparameter tuning (GridSearch / RandomSearch)
  * Use advanced models like XGBoost / LightGBM
  * Handle imbalance using SMOTE or class weights
  * Deploy model using Flask or Streamlit


Tools & Technologies Used:

  * Python – Core programming language
  * Pandas – Data loading and manipulation
  * NumPy – Numerical computations
  * Scikit-learn – Machine learning, preprocessing, and evaluation
  * Jupyter Notebook – Development and experimentation environment
  * Joblib – Model saving and loading


Conclusion:

This project successfully demonstrates the end-to-end development of a machine learning model for predicting customer transaction behavior using anonymized data. Despite the absence of domain-specific feature information, a structured approach involving data preprocessing, feature selection, and model comparison enabled the creation of a robust predictive system.

By leveraging multiple machine learning algorithms and evaluating them using appropriate metrics such as ROC-AUC and F1-score, the best-performing model was identified and prepared for deployment. The solution highlights the importance of proper preprocessing, model validation, and metric selection in real-world business problems.

Overall, the project provides a scalable and practical framework that can assist banks in making data-driven decisions, improving customer targeting, and optimizing business strategies.


How to Run the Project:

Step 1: Clone the repository
   * ( git clone <your-repo-link> )

Step 2: Navigate to the project folder
   * ( cd <project-folder> )
    
Step 3: Install required libraries
   * ( pip install -r requirements.txt )
 
Step 4: Run Jupyter Notebook
   * ( jupyter notebook )
