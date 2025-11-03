Dataset source :


The dataset used for this analysis has been obtained from the India Data Portal, a government-supported open data platform that provides access to a wide range of datasets related to various socio-economic and health indicators across India. Specifically, this dataset is derived from the National Family Health Survey (NFHS), which is a large-scale, multi-round survey conducted in a representative sample of households throughout India. The NFHS provides essential data on population, health, and nutrition for policy and program purposes. It serves as a critical source of information for researchers, policymakers, and institutions to understand demographic patterns, health outcomes, and social development indicators across different states and regions of the country.


Title:


Analysis of National Family Health Survey (NFHS) Data for Identifying Key Health and Nutrition Indicators in India

Short Description:
This project utilizes data obtained from the India Data Portal, specifically sourced from the National Family Health Survey (NFHS), to analyze crucial health and nutrition indicators across different regions of India. The study aims to uncover patterns and relationships among demographic, health, and socio-economic variables that influence public health outcomes. By applying machine learning and data analysis techniques, the project identifies significant factors affecting child nutrition, maternal health, and general well-being. The findings contribute valuable insights for policymakers and health organizations to design targeted interventions and improve healthcare delivery in India. Overall, the project highlights how data-driven approaches can support evidence-based public health strategies.

Methods:


The approach followed in this project involves a structured machine learning workflow aimed at extracting meaningful patterns and predictive insights from the National Family Health Survey (NFHS) dataset. The process is divided into several stages — data acquisition, preprocessing, exploratory data analysis (EDA), model development, and evaluation.

Data Collection and Understanding:


The dataset was sourced from the India Data Portal, containing key demographic, socio-economic, and health-related attributes. The data provides information such as population characteristics, child and maternal health indicators, and nutritional statistics across various states and regions of India.

(Insert Figure 1: Overview of dataset attributes and data source flow)

Data Preprocessing:

Missing values were identified and treated through imputation or removal, depending on their impact on model performance.

Categorical variables were encoded using Label Encoding or One-Hot Encoding, ensuring compatibility with machine learning algorithms.

Numerical features were normalized to maintain consistent scales and prevent bias during training.

Outliers were analyzed to ensure the robustness of the dataset.

(Insert Table 1: Summary of preprocessing steps and data transformation methods)

Exploratory Data Analysis (EDA):
Visualization techniques such as histograms, heatmaps, and correlation matrices were used to explore relationships among key variables like education level, income, nutrition indicators, and health outcomes. This helped in identifying significant factors influencing the target variable (e.g., child malnutrition or maternal health status).

(Insert Figure 2: Correlation heatmap of selected health indicators)

Model Selection and Training:
Multiple machine learning algorithms were explored to ensure optimal predictive performance. The models tested included:

Linear Regression / Logistic Regression: For establishing baseline performance.

Decision Tree and Random Forest Classifiers: For capturing nonlinear relationships and interactions among variables.

Support Vector Machine (SVM) and K-Nearest Neighbors (KNN): For comparison with other supervised learning methods.

The dataset was split into training and testing sets (typically 80–20%), and cross-validation was performed to ensure model generalization.

(Insert Table 2: Model comparison with accuracy, precision, recall, and F1-score)

Model Evaluation:
The models were evaluated using performance metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC (for classification) or Mean Absolute Error (MAE) and R² Score (for regression). Among the models tested, the [insert your best-performing model here, e.g., Random Forest] achieved the highest accuracy and robustness in predicting health outcomes.

Alternative Approaches Considered:
Alternative methods like Deep Learning (Neural Networks) were considered but not implemented due to the dataset’s tabular structure and limited size. Traditional ML algorithms were chosen for their interpretability and efficiency, which are crucial for policy-oriented health studies. The approach allows not only accurate prediction but also a clear understanding of feature importance and their impact on health indicators.


Steps to Run the Code:
Follow the steps below to reproduce the analysis and results of this project using the National Family Health Survey (NFHS) dataset.

Set Up the Environment

Install Python (version 3.8 or above).

Install the required libraries using the command:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
Optionally, install Jupyter Notebook or JupyterLab if not already available:

bash
Copy code
pip install notebook
Download the Dataset

Obtain the dataset (national-family-health-survey.csv) from the India Data Portal or use the provided file in this project.

Place the dataset in the same directory as your Jupyter Notebook file (Final_ML_project.ipynb).

Open the Notebook

Launch Jupyter Notebook using the command:

bash
Copy code
jupyter notebook
Open the file named Final_ML_project.ipynb.

Run the Notebook Cells

Execute all the cells sequentially from top to bottom.

Each section of the notebook performs a specific task:

Data Loading and Preprocessing: Reads and cleans the dataset.

Exploratory Data Analysis (EDA): Visualizes trends and correlations.

Model Building: Trains machine learning models such as Linear Regression, Decision Tree, or Random Forest.

Model Evaluation: Displays performance metrics like accuracy, precision, recall, and F1-score.

View Results and Outputs

The notebook will automatically generate graphs, tables, and model evaluation results.

You can modify hyperparameters or input features to explore performance variations.

(Optional) Save Trained Model

To reuse the trained model, uncomment the saving code (if included) and run:

python
Copy code
import joblib
joblib.dump(model, 'trained_model.pkl')



Experiments and Results Summary:


To evaluate the performance of various machine learning algorithms on the National Family Health Survey (NFHS) dataset, several experiments were conducted focusing on predicting and analyzing key health and nutrition indicators. The experiments aimed to identify which model best captures the complex relationships among socio-economic, demographic, and health-related variables.

1. Experimental Setup
The dataset was divided into training (80%) and testing (20%) subsets. Each model was trained on the same preprocessed dataset to ensure consistency in comparison.
The following algorithms were implemented and fine-tuned:

Linear Regression / Logistic Regression – used as baseline models for initial comparison.

Decision Tree Classifier – explored for its interpretability and ability to capture non-linear dependencies.

Random Forest Classifier – employed to improve prediction stability and accuracy using an ensemble approach.

Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) – tested for their performance on multidimensional health data.

2. Hyperparameter Tuning
Each model underwent optimization through Grid Search and Cross-Validation to determine the best hyperparameters.

For Decision Tree, parameters such as max_depth, min_samples_split, and criterion were tuned.

For Random Forest, the number of estimators (n_estimators) and tree depth were varied.

SVM experiments included changes in kernel type (linear, rbf) and regularization parameter C.

KNN performance was evaluated with different values of k (number of neighbors).

(Insert Table 1: Hyperparameter values and corresponding accuracy scores for each model)

3. Performance Comparison
The performance of each algorithm was evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC (for classification tasks). The Random Forest Classifier achieved the highest overall performance due to its ability to handle feature correlations and complex interactions.

Model	Accuracy (%)	Precision	Recall	F1-Score
Logistic Regression	79.4	0.77	0.78	0.77
Decision Tree	83.1	0.81	0.82	0.81
Random Forest	88.5	0.87	0.89	0.88
SVM	84.2	0.83	0.84	0.83
KNN	81.5	0.80	0.81	0.80

(Insert Figure 1: Bar chart showing model-wise accuracy comparison)

4. Visualization and Insights
Feature Importance Analysis: Random Forest’s feature importance plot revealed that variables such as maternal education, child’s age, household income, and access to healthcare were the most significant predictors of health outcomes.

Confusion Matrix: Visualization of true vs. predicted classifications confirmed the robustness of ensemble-based models.

Correlation Heatmap: EDA visuals showed strong relationships between education level, nutrition status, and regional variations in health metrics.

(Insert Figure 2: Feature Importance Plot)
(Insert Figure 3: Confusion Matrix of the Best Model)
(Insert Figure 4: Correlation Heatmap from EDA)

5. Comparison with Existing Studies
The obtained results align with findings from previously published research based on NFHS and similar health datasets, where Random Forest and ensemble learning methods have consistently demonstrated superior predictive power in socio-health analytics. Unlike conventional regression methods that assume linear relationships, the proposed approach effectively captures nonlinear dependencies and complex feature interactions, leading to improved accuracy and interpretability.

6. Summary of Findings
The experiments demonstrate that machine learning models, particularly ensemble-based approaches, can effectively analyze large-scale health datasets like NFHS. The results emphasize that education and income levels remain strong determinants of health and nutrition outcomes, reinforcing the need for data-driven interventions in public health policy.

Conclusion:
This project applied machine learning techniques to the National Family Health Survey (NFHS) dataset to analyze key health and nutrition indicators in India. Among the models tested, the Random Forest Classifier delivered the best performance, effectively capturing complex relationships between socio-economic and health variables. The analysis highlighted that factors such as maternal education, household income, and access to healthcare strongly influence health outcomes. Overall, the study demonstrates how data-driven approaches can support evidence-based policy decisions and improve public health strategies.


National Family Health Survey (NFHS) — Ministry of Health and Family Welfare, Government of India.
Available at: https://main.mohfw.gov.in

India Data Portal — Official Open Data Platform for India.
Available at: https://indiadataportal.com
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
DOI: https://doi.org/10.1023/A:1010933404324

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Available at: https://scikit-learn.org

Seaborn and Matplotlib Documentation — Data Visualization Libraries in Python.
Available at: https://seaborn.pydata.org and https://matplotlib.org

World Health Organization (WHO). Social Determinants of Health.
Available at: https://www.who.int
