Tasks I have to complete :

- Diabets risk prediction
- Feature importance analysis
- Correlation analysis
- Data visualization
- Dataset handling
- Age-related analysis
- Pedigree function analysis
- Model performance

/--------------------------------\
Imports and Timer:

The necessary libraries for data analysis, machine learning, and visualization are imported.

A timer is started to measure the execution time of the script.
Read and Explore Data:

A dataset related to healthcare and diabetes is read into a DataFrame.

Basic information about the dataset (data types, non-null counts) and descriptive statistics is displayed.
Data Preparation:

The dataset is separated into features (X) and the target variable (y).

The data is split into training and testing sets using train_test_split.
Data Standardization:

Features are standardized using StandardScaler to bring them to zero mean and unit variance.
Building and Training the Logistic Regression Model:

A logistic regression model is built and trained on the training data.
Making Predictions and Evaluating the Model:

Predictions are made on the test set.

The model's performance is evaluated using accuracy, a classification report, and a confusion matrix.
Visualization and Interpretation:

Seaborn and Matplotlib are used to create visualizations:

A histogram for the age distribution.
A scatter plot for glucose vs BMI with markers for the 'Outcome' variable.
A bar plot for the distribution of the 'Outcome' variable.
A heatmap for the correlation matrix between different features.
Measuring Execution Time:
The total execution time of the script is measured and displayed.
