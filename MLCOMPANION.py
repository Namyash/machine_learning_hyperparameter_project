# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVC, SVR
# from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression, Ridge, Lasso
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
# import io
# import time
# from sklearn.metrics import confusion_matrix
# import plotly.figure_factory as ff

# # Custom styling for a sexy UI
# st.markdown("""
#     <style>
#     .main {
#         background-color: #ac11d6;
#         padding: 20px;
#         border-radius: 10px;
#     }
#     .stButton>button {
#         background-color: #ac11d6;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#     }
#     .stSelectbox>select, .stRadio>label, .stCheckbox>label {
#         font-size: 16px;
#         color: #333;
#     }
#     .stProgress>div>div {
#         background-color: #4CAF50;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title with logo (simulated with text)
# st.markdown("<h1 style='text-align: center; color: #ac11d6;'>ML Companion: Generic Data Insights and Model Tuning</h1>", unsafe_allow_html=True)

# # Step 1: Upload a CSV file with sampling option for large datasets
# uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

# if uploaded_file is not None:
#     try:
#         # Load dataset and optionally sample for large files
#         df = pd.read_csv(uploaded_file)
#         total_rows = len(df)
#         if total_rows > 10000:
#             sample_size = st.slider("Sample size for large datasets (rows)", 1000, total_rows, 5000, step=1000)
#             st.warning(f"Dataset is large ({total_rows} rows). Sampling {sample_size} rows for performance.")
#             df = df.sample(n=sample_size, random_state=42)
#         else:
#             st.write(f"Dataset size: {total_rows} rows")

#         # Step 2: Show dataset preview and stats
#         st.write("### Dataset Preview")
#         st.dataframe(df.head())
#         st.write(f"Total rows containing null values: {df.isnull().any(axis=1).sum()}/{len(df)}")

#         # Dataset statistics for guidance
#         st.write("### Dataset Statistics")
#         st.write("Numerical Columns Skewness:")
#         st.write(df.select_dtypes(include=['int64', 'float64']).skew().round(2))
#         st.write("Missing Values by Column:")
#         st.write(df.isnull().sum())

#         # Step 3: Check column types
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#         numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#         st.write(f"**Categorical Columns**: {categorical_cols}")
#         st.write(f"**Numerical Columns**: {numerical_cols}")

#         # Step 4: Select target column with help text
#         target_column = st.selectbox(
#             "Select the target column", 
#             df.columns, 
#             help="Choose the column you want to predict (e.g., a class for classification or a number for regression)"
#         )
#         st.write(f"Selected target column: {target_column}")

#         if target_column not in df.columns:
#             st.error("Please select a valid target column.")
#         else:
#             # Determine task type with user override
#             unique_values = df[target_column].nunique()
#             is_categorical = target_column in categorical_cols
#             default_classification = is_categorical or (unique_values <= 10 and df[target_column].dtype in ['int64', 'float64'])
#             task_type = st.radio(
#                 "Detected Task Type (override if needed):", 
#                 ["Classification", "Regression"], 
#                 index=0 if default_classification else 1,
#                 help="Auto-detected based on unique values and data type. Adjust if incorrect."
#             )
#             is_classification = task_type == "Classification"
#             st.write(f"Task type set to: {task_type} (based on {unique_values} unique values in '{target_column}')")

#             # Encode categorical columns if required
#             if st.checkbox("Encode Categorical Columns", help="Convert categorical columns to numerical using Label Encoding"):
#                 label_encoders = {}
#                 for col in categorical_cols:
#                     le = LabelEncoder()
#                     df[col] = le.fit_transform(df[col].astype(str))
#                     label_encoders[col] = le
#                 st.write("Categorical columns have been label encoded.")

#             # Encode the target column if selected (only for classification)
#             if is_classification and st.checkbox("Encode Target Column", help="Convert target column to numerical for classification"):
#                 if target_column in categorical_cols:
#                     le_target = LabelEncoder()
#                     df[target_column] = le_target.fit_transform(df[target_column].astype(str))
#                     st.write(f"Target column '{target_column}' has been label encoded.")

#             # Step 5: Handle missing values with user choice and suggestions
#             if st.checkbox("Handle Missing Values", help="Address missing data in your dataset"):
#                 st.write("### Missing Values Handling")
#                 missing_summary = df.isnull().sum()
#                 st.write("Missing values per column:")
#                 st.write(missing_summary[missing_summary > 0])
#                 handle_option = st.radio(
#                     "Select a method:",
#                     ["Remove rows with null values", "Fill numerical columns with Mean", 
#                      "Fill numerical columns with Median", "Fill numerical columns with Mode", 
#                      "Drop columns with > 50% missing values"],
#                     help="Choose how to handle missing data based on your dataset's needs."
#                 )

#                 if handle_option == "Remove rows with null values":
#                     df = df.dropna()
#                     st.write("Rows with null values removed.")
#                 elif handle_option == "Fill numerical columns with Mean":
#                     df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#                     for col in categorical_cols:
#                         df[col].fillna(df[col].mode()[0], inplace=True)
#                     st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")
#                 elif handle_option == "Fill numerical columns with Median":
#                     df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                     for col in categorical_cols:
#                         df[col].fillna(df[col].mode()[0], inplace=True)
#                     st.write("Missing values filled with median for numerical columns and mode for categorical columns.")
#                 elif handle_option == "Fill numerical columns with Mode":
#                     df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
#                     for col in categorical_cols:
#                         df[col].fillna(df[col].mode()[0], inplace=True)
#                     st.write("Missing values filled with mode for both numerical and categorical columns.")
#                 elif handle_option == "Drop columns with > 50% missing values":
#                     df = df.loc[:, df.isnull().mean() < 0.5]
#                     st.write("Columns with more than 50% missing values dropped.")
#                 numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#                 categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#             # Step 5.1: Handle Outliers with suggestions
#             if st.checkbox("Handle Outliers", help="Manage extreme values in numerical columns"):
#                 st.write("### Outlier Handling")
#                 outlier_summary = {}
#                 for col in numerical_cols:
#                     Q1, Q3 = df[col].quantile([0.25, 0.75])
#                     IQR = Q3 - Q1
#                     lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
#                     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
#                     if outliers > 0:
#                         outlier_summary[col] = outliers
#                 if outlier_summary:
#                     st.write("Columns with outliers detected:")
#                     st.write(pd.Series(outlier_summary))
#                 outlier_option = st.radio(
#                     "Select a method:",
#                     ["Remove rows with outliers", "Cap outliers to 5th and 95th percentiles"],
#                     help="Choose how to handle outliers based on your dataset's distribution."
#                 )

#                 def remove_outliers(df, numerical_cols):
#                     for col in numerical_cols:
#                         Q1 = df[col].quantile(0.25)
#                         Q3 = df[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#                         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                     return df

#                 def cap_outliers(df, numerical_cols):
#                     for col in numerical_cols:
#                         lower_bound = df[col].quantile(0.05)
#                         upper_bound = df[col].quantile(0.95)
#                         df[col] = np.clip(df[col], lower_bound, upper_bound)
#                     return df

#                 if outlier_option == "Remove rows with outliers":
#                     df = remove_outliers(df, numerical_cols)
#                     st.write("Rows with outliers removed.")
#                 elif outlier_option == "Cap outliers to 5th and 95th percentiles":
#                     df = cap_outliers(df, numerical_cols)
#                     st.write("Outliers capped to 5th and 95th percentiles.")
#                 numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             # Optional log transformation for target and skewed features (regression only)
#             if task_type == "Regression":
#                 if st.checkbox("Apply Log Transformation to Target", help="Log-transform the target for regression if skewed"):
#                     df[target_column] = np.log1p(df[target_column])
#                     st.write(f"Log transformation applied to target column '{target_column}'")
#                 if st.checkbox("Apply Log Transformation to Skewed Features", help="Log-transform numerical features with high skewness"):
#                     skewed_cols = [col for col in numerical_cols if df[col].skew() > 1 and col != target_column]
#                     if skewed_cols:
#                         for col in skewed_cols:
#                             df[col] = np.log1p(df[col])  # log1p to handle zeros
#                         st.write(f"Log transformation applied to: {skewed_cols}")
#                     else:
#                         st.write("No highly skewed features detected for log transformation.")
#                 numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

#             # Feature engineering for regression (optional, now generic)
#             if task_type == "Regression" and st.checkbox("Perform Feature Engineering", help="Create interactions and polynomial terms for numerical features"):
#                 if len(numerical_cols) > 1:
#                     for i in range(len(numerical_cols)):
#                         for j in range(i + 1, len(numerical_cols)):
#                             if numerical_cols[i] != target_column and numerical_cols[j] != target_column:
#                                 new_col = f"{numerical_cols[i]}_{numerical_cols[j]}_Interaction"
#                                 df[new_col] = df[numerical_cols[i]] * df[numerical_cols[j]]
#                                 st.write(f"Added interaction term: {new_col}")
#                     for col in numerical_cols:
#                         if col != target_column:
#                             new_col = f"{col}_Squared"
#                             df[new_col] = df[col] ** 2
#                             st.write(f"Added polynomial term: {new_col}")
#                 else:
#                     st.write("Insufficient numerical columns for feature engineering.")
#                 numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#                 numerical_cols_without_target = [col for col in numerical_cols if col != target_column]

#             # Step 6: Standardize all numerical data except target column
#             st.write("### Standardizing Data")
#             scaler = StandardScaler()
#             numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
#             if numerical_cols_without_target:
#                 df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
#                 st.write(f"Standardized columns: {numerical_cols_without_target}")
#             else:
#                 st.warning("No numerical columns to standardize except the target.")

#             # Show the dataset after preprocessing
#             st.write("### Preprocessed Dataset")
#             st.dataframe(df)

#             # Step 7: Feature Selection
#             selected_features = numerical_cols_without_target
#             if st.checkbox("Perform Feature Selection", help="Select features based on their correlation with the target"):
#                 st.write("### Feature Selection")
#                 correlation_threshold = st.slider("Set correlation threshold", -1.0, 1.0, 0.2, 
#                                                 help="Higher threshold excludes weaker correlations; lower includes more features.")
#                 correlations = df[numerical_cols_without_target].corrwith(df[target_column])
#                 selected_features = correlations[abs(correlations) >= correlation_threshold].index.tolist()
#                 st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

#             if not selected_features:
#                 st.warning("No features selected. Using all numerical features except target.")
#                 selected_features = numerical_cols_without_target

#             # Quick Start mode for beginners
#             quick_start = st.checkbox("Quick Start Mode (Basic Settings for Beginners)", 
#                                      help="Use default preprocessing and models without tuning for fast results.")
#             if quick_start:
#                 st.write("### Quick Start Mode Enabled")
#                 tune_params = False
#                 handle_option = "Fill numerical columns with Median"
#                 outlier_option = "Cap outliers to 5th and 95th percentiles"
#                 if st.checkbox("Apply Quick Start Preprocessing"):
#                     if df.isnull().sum().any():
#                         df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
#                         for col in categorical_cols:
#                             df[col].fillna(df[col].mode()[0], inplace=True)
#                         st.write("Applied quick start missing value handling (Median for numerical, Mode for categorical).")
#                     df = cap_outliers(df, numerical_cols)
#                     st.write("Applied quick start outlier handling (capped at 5th/95th percentiles).")
#                 selected_features = numerical_cols_without_target  # Use all features

#             # Step 8: Model Training with progress
#             if st.checkbox("Train Model", help="Train machine learning models on your preprocessed data"):
#                 st.write(f"### Training {task_type} Models")
#                 X = df[selected_features]
#                 y = df[target_column]
                
#                 # Train/Test Split with progress spinner
#                 with st.spinner("Splitting data into training and test sets..."):
#                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                     time.sleep(1)  # Simulate processing time
#                 st.success("Data split completed.")

#                 if is_classification:
#                     st.write("#### Class Distribution Before SMOTE:")
#                     st.write(y_train.value_counts())
#                     unique_counts = y_train.value_counts()
#                     min_count = unique_counts.min()
#                     max_count = unique_counts.max()

#                     if min_count / max_count < 0.8:
#                         st.write(f"Imbalance detected (minority/majority ratio: {min_count/max_count:.2f})")
#                         smote_option = st.radio(
#                             "Handle class imbalance with SMOTE?",
#                             ["Fully balance to majority class", "Partially balance (custom ratio)", "Skip SMOTE"],
#                             help="Oversample minority classes to balance the dataset."
#                         )

#                         if smote_option == "Fully balance to majority class":
#                             with st.spinner("Applying SMOTE to balance classes..."):
#                                 smote = SMOTE(random_state=42)
#                                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                                 time.sleep(1)
#                             st.write("SMOTE applied: Classes fully balanced to match majority class.")
#                         elif smote_option == "Partially balance (custom ratio)":
#                             ratio = st.slider("Select minority-to-majority ratio", 0.1, 1.0, 0.3, step=0.1, 
#                                              help="Set the proportion of minority to majority class after SMOTE.")
#                             with st.spinner("Applying SMOTE with custom ratio..."):
#                                 smote = SMOTE(sampling_strategy=ratio, random_state=42)
#                                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                                 time.sleep(1)
#                             st.write(f"SMOTE applied: Minority class oversampled to {ratio*100}% of majority class.")
#                         else:
#                             st.write("SMOTE skipped: Proceeding with original class distribution.")
#                     else:
#                         st.write("No significant imbalance detected. Proceeding without SMOTE.")

#                     st.write("#### Class Distribution After SMOTE (if applied):")
#                     st.write(y_train.value_counts())

#                     if smote_option in ["Fully balance to majority class", "Partially balance (custom ratio)"]:
#                         with st.spinner("Preparing SMOTE-applied dataset for download..."):
#                             smote_df = pd.DataFrame(X_train, columns=selected_features)
#                             smote_df[target_column] = y_train
#                             csv_buffer = io.StringIO()
#                             smote_df.to_csv(csv_buffer, index=False)
#                             time.sleep(1)
#                         st.download_button(
#                             label="Download SMOTE-Applied Training Dataset",
#                             data=csv_buffer.getvalue(),
#                             file_name="smote_applied_dataset.csv",
#                             mime="text/csv"
#                         )

#                 # Step 9: Visualize Data with interactive Plotly
#                 if st.checkbox("Visualize Data Distribution", help="Explore your data with interactive plots"):
#                     st.write("### Data Visualization")
#                     train_df = pd.DataFrame(X_train, columns=selected_features)
#                     train_df[target_column] = y_train

#                     viz_column = st.selectbox("Select a column to visualize", train_df.columns, 
#                                              help="Choose a feature or target to visualize.")
#                     viz_type = st.radio("Choose visualization type", ["Histogram", "Box Plot", "Scatter Plot", 
#                                                                    "Heatmap", "Count Plot", "Violin Plot"],
#                                         help="Select the type of plot for data exploration.")

#                     if viz_type == "Histogram" and viz_column in numerical_cols:
#                         fig = px.histogram(train_df, x=viz_column, nbins=30, title=f"Histogram of {viz_column}")
#                         st.plotly_chart(fig)
#                     elif viz_type == "Count Plot" and viz_column in categorical_cols:
#                         fig = px.histogram(train_df, x=viz_column, title=f"Count Plot of {viz_column}")
#                         st.plotly_chart(fig)
#                     elif viz_type == "Box Plot" and viz_column in numerical_cols:
#                         fig = px.box(train_df, y=viz_column, title=f"Box Plot of {viz_column}")
#                         st.plotly_chart(fig)
#                     elif viz_type == "Scatter Plot" and len(selected_features) > 1:
#                         x_column = st.selectbox("Select x-axis column", selected_features, 
#                                                help="Choose the x-axis feature for the scatter plot.")
#                         y_column = st.selectbox("Select y-axis column", selected_features, 
#                                                help="Choose the y-axis feature for the scatter plot.")
#                         fig = px.scatter(train_df, x=x_column, y=y_column, color=target_column, 
#                                         title=f"Scatter Plot: {x_column} vs {y_column}")
#                         st.plotly_chart(fig)
#                     elif viz_type == "Heatmap" and len(selected_features) > 1:
#                         corr_matrix = train_df[selected_features].corr()
#                         fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
#                                        title="Correlation Heatmap of Features")
#                         st.plotly_chart(fig)
#                     elif viz_type == "Violin Plot" and viz_column in numerical_cols:
#                         fig = px.violin(train_df, y=viz_column, box=True, title=f"Violin Plot of {viz_column}")
#                         st.plotly_chart(fig)
#                     else:
#                         st.write("Invalid combination of column and visualization type or insufficient features.")

#                 # Define models based on task type
#                 classification_models = {
#                     "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10, 100]}),
#                     "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}),
#                     "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
#                     "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]}),
#                     "Perceptron": (Perceptron(), {"alpha": [0.0001, 0.001, 0.01]}),
#                     "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]})
#                 }

#                 regression_models = {
#                     "Linear Regression": (LinearRegression(), {"fit_intercept": [True, False]}),
#                     "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
#                     "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
#                     "SVR": (SVR(), {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto", 0.1, 0.01]}),
#                     "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]}),
#                     "Gradient Boosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]})
#                 }

#                 models = classification_models if is_classification else regression_models
#                 model_options = list(models.keys()) + ["Compare All Models"]

#                 # Quick model recommendation
#                 if st.checkbox("Show Model Recommendations", help="Get suggested models based on dataset size and task"):
#                     st.write(f"### Recommended Models for {task_type}")
#                     if task_type == "Classification":
#                         st.write("- Start with Random Forest or Gradient Boosting for balanced accuracy and robustness.")
#                     else:
#                         st.write("- Start with Random Forest or Gradient Boosting for high R² on numerical predictions.")
#                     if len(df) > 5000:
#                         st.write("- Consider using Quick Start Mode or sampling for large datasets.")

#                 # Model Selection with tuning option and recommendation
#                 model_option = st.selectbox(
#                     f"Select a {task_type.lower()} model or compare all", 
#                     model_options, 
#                     help="Choose a single model or compare all to find the best performer."
#                 )
#                 tune_params = st.checkbox("Perform Hyperparameter Tuning (faster with Randomized Search)", value=not quick_start, 
#                                         help="Optimize model performance, but may take longer. Disable for quick results.")
#                 if tune_params:
#                     n_iter = st.slider("Number of iterations for Randomized Search", 5, 30, 20, 
#                                       help="More iterations improve tuning but increase runtime.")
#                     cv_folds = st.slider("Number of cross-validation folds", 3, 10, 5, 
#                                         help="More folds ensure robust results but increase computation time.")

#                 # Function to evaluate a model with performance monitoring and predictions
#                 def evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, tune=True, is_regression=False):
#                     start_time = time.time()
#                     if tune and param_grid:
#                         search = RandomizedSearchCV(model, param_grid, n_iter=n_iter if tune_params else 5, 
#                                                   cv=cv_folds if tune_params else 3, random_state=42, n_jobs=-1)
#                         with st.spinner(f"Tuning {model.__class__.__name__}..."):
#                             search.fit(X_train, y_train)
#                         best_model = search.best_estimator_
#                         best_params = search.best_params_
#                     else:
#                         best_model = model
#                         with st.spinner(f"Training {model.__class__.__name__} with default parameters..."):
#                             best_model.fit(X_train, y_train)
#                         best_params = "Default Parameters"
                    
#                     training_time = time.time() - start_time
#                     st.write(f"Training time for {model.__class__.__name__}: {training_time:.2f} seconds")

#                     y_pred = best_model.predict(X_test)
#                     if is_regression:
#                         mse = mean_squared_error(y_test, y_pred)
#                         r2 = r2_score(y_test, y_pred)
#                         mae = mean_absolute_error(y_test, y_pred)
#                         baseline_pred = np.full_like(y_test, y_train.mean())
#                         baseline_r2 = r2_score(y_test, baseline_pred)
#                         return {
#                             "Best Parameters": best_params,
#                             "Predictions": y_pred,  # Store predictions here for regression
#                             "Mean Squared Error": mse,
#                             "R² Score": r2,
#                             "Mean Absolute Error": mae,
#                             "Baseline R² (Mean Predictor)": baseline_r2,
#                             "Training Time (s)": training_time
#                         }
#                     else:
#                         accuracy = accuracy_score(y_test, y_pred)
#                         precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#                         recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#                         f1 = f1_score(y_test, y_pred, average='weighted')
#                         cm = confusion_matrix(y_test, y_pred)
#                         return {
#                             "Best Parameters": best_params,
#                             "Predictions": y_pred,  # Store predictions here for classification
#                             "Accuracy": accuracy,
#                             "Precision": precision,
#                             "Recall": recall,
#                             "F1 Score": f1,
#                             "Confusion Matrix": cm,
#                             "Training Time (s)": training_time
#                         }

#                 # Train and evaluate based on user selection with progress
#                 if model_option == "Compare All Models":
#                     st.write(f"### Training and Evaluating All {task_type.lower()} Models")
#                     results = {}
#                     progress_bar = st.progress(0)
#                     total_models = len(models)
#                     for i, (name, (model, param_grid)) in enumerate(models.items()):
#                         with st.spinner(f"Training {name}..."):
#                             results[name] = evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, 
#                                                          tune=tune_params, is_regression=not is_classification)
#                         progress_bar.progress((i + 1) / total_models)

#                     # Display results in an interactive table
#                     comparison_df = pd.DataFrame(results).T
#                     st.write(f"### Comparison of All {task_type} Models")
#                     st.dataframe(comparison_df.style.format({"R² Score": "{:.3f}", "Mean Squared Error": "{:.2f}", 
#                                                            "Mean Absolute Error": "{:.2f}", "Accuracy": "{:.3f}", 
#                                                            "Training Time (s)": "{:.2f}"}))

#                     # Download comparison results
#                     buffer = io.StringIO()
#                     comparison_df.to_csv(buffer, index=True)
#                     st.download_button(
#                         label=f"Download {task_type} Model Comparison Results",
#                         data=buffer.getvalue(),
#                         file_name=f"{task_type.lower()}_model_comparison_results.csv",
#                         mime="text/csv"
#                     )

#                     # Visualize results
#                     if task_type == "Regression":
#                         st.write("### Predicted vs. Actual Plot")
#                         first_model = list(models.keys())[0]
#                         fig = px.scatter(x=y_test, y=results[first_model]["Predictions"], 
#                                         trendline="ols", title="Predicted vs. Actual Values")
#                         st.plotly_chart(fig)
#                     else:
#                         st.write("### Confusion Matrix (First Model)")
#                         cm = results[list(models.keys())[0]]["Confusion Matrix"]
#                         fig = ff.create_annotated_heatmap(cm, x=list(range(len(cm))), y=list(range(len(cm))), 
#                                                         colorscale='Viridis')
#                         st.plotly_chart(fig)

#                 else:
#                     # Train a single model with progress
#                     with st.spinner(f"Training {model_option}..."):
#                         model, param_grid = models[model_option]
#                         result = evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, 
#                                               tune=tune_params, is_regression=not is_classification)
#                     st.success(f"Training {model_option} completed.")

#                     # Display results
#                     st.write(f"### Results for {model_option}:")
#                     for metric, value in result.items():
#                         if metric != "Confusion Matrix" and metric != "Predictions":
#                             st.write(f"{metric}: {value}")
#                         elif metric == "Confusion Matrix":
#                             st.write("Confusion Matrix:")
#                             fig = ff.create_annotated_heatmap(value, x=list(range(len(value))), y=list(range(len(value))), 
#                                                             colorscale='Viridis')
#                             st.plotly_chart(fig)

#                     # Download model details and predictions
#                     buffer = io.StringIO()
#                     buffer.write(f"Model: {model_option}\n")
#                     for metric, value in result.items():
#                         if metric != "Confusion Matrix" and metric != "Predictions":
#                             buffer.write(f"{metric}: {value}\n")
#                     st.download_button(
#                         label="Download Model Details",
#                         data=buffer.getvalue(),
#                         file_name=f"{model_option}_details.txt",
#                         mime="text/plain"
#                     )

#                     # Download predictions
#                     if task_type == "Regression":
#                         predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": result["Predictions"]})
#                         csv_buffer = io.StringIO()
#                         predictions_df.to_csv(csv_buffer, index=False)
#                         st.download_button(
#                             label="Download Predictions",
#                             data=csv_buffer.getvalue(),
#                             file_name=f"{model_option}_predictions.csv",
#                             mime="text/csv"
#                         )

#             # Save/Load State
#             if st.checkbox("Save Current Settings", help="Save your preprocessing and model settings for later use"):
#                 settings = {
#                     "target_column": target_column,
#                     "task_type": task_type,
#                     "encode_categorical": st.session_state.get("Encode Categorical Columns", False),
#                     "encode_target": st.session_state.get("Encode Target Column", False),
#                     "handle_missing": st.session_state.get("Handle Missing Values", False),
#                     "missing_method": handle_option if "handle_missing" in locals() else None,
#                     "handle_outliers": st.session_state.get("Handle Outliers", False),
#                     "outlier_method": outlier_option if "handle_outliers" in locals() else None,
#                     "log_target": st.session_state.get("Apply Log Transformation to Target", False),
#                     "log_features": st.session_state.get("Apply Log Transformation to Skewed Features", False),
#                     "feature_engineering": st.session_state.get("Perform Feature Engineering", False),
#                     "feature_selection": st.session_state.get("Perform Feature Selection", False),
#                     "correlation_threshold": correlation_threshold if "feature_selection" in locals() else 0.2,
#                     "tune_params": tune_params,
#                     "n_iter": n_iter if tune_params else 5,
#                     "cv_folds": cv_folds if tune_params else 3,
#                     "model_option": model_option,
#                     "quick_start": quick_start
#                 }
#                 import json
#                 settings_json = json.dumps(settings)
#                 st.download_button(
#                     label="Download Settings",
#                     data=settings_json,
#                     file_name="ml_settings.json",
#                     mime="application/json"
#                 )

#             if st.checkbox("Load Previous Settings", help="Load saved preprocessing and model settings"):
#                 uploaded_settings = st.file_uploader("Upload settings file (JSON)", type=["json"])
#                 if uploaded_settings:
#                     settings = json.load(uploaded_settings)
#                     st.session_state["target_column"] = settings["target_column"]
#                     st.session_state["task_type"] = settings["task_type"]
#                     st.session_state["Encode Categorical Columns"] = settings["encode_categorical"]
#                     st.session_state["Encode Target Column"] = settings["encode_target"]
#                     st.session_state["Handle Missing Values"] = settings["handle_missing"]
#                     if settings["handle_missing"]:
#                         handle_option = settings["missing_method"]
#                     st.session_state["Handle Outliers"] = settings["handle_outliers"]
#                     if settings["handle_outliers"]:
#                         outlier_option = settings["outlier_method"]
#                     st.session_state["Apply Log Transformation to Target"] = settings["log_target"]
#                     st.session_state["Apply Log Transformation to Skewed Features"] = settings["log_features"]
#                     st.session_state["Perform Feature Engineering"] = settings["feature_engineering"]
#                     st.session_state["Perform Feature Selection"] = settings["feature_selection"]
#                     if settings["feature_selection"]:
#                         correlation_threshold = settings["correlation_threshold"]
#                     st.session_state["tune_params"] = settings["tune_params"]
#                     if settings["tune_params"]:
#                         n_iter = settings["n_iter"]
#                         cv_folds = settings["cv_folds"]
#                     st.session_state["model_option"] = settings["model_option"]
#                     st.session_state["quick_start"] = settings["quick_start"]
#                     st.success("Settings loaded successfully. Refresh the page to apply.")

#             # About/Help section
#             with st.expander("About & Help", expanded=False):
#                 st.write("""
#                 ### ML Companion: Generic Data Insights and Model Tuning
#                 This app helps you preprocess, visualize, and train machine learning models on any CSV dataset.
                
#                 **Features:**
#                 - Handles classification and regression tasks automatically.
#                 - Preprocesses data (missing values, outliers, encoding, standardization, log transformations).
#                 - Offers feature engineering, selection, and interactive visualizations.
#                 - Trains multiple models with optional hyperparameter tuning.
#                 - Saves/loads settings for reproducibility.
                
#                 **Quick Tips:**
#                 - Use "Quick Start Mode" for beginners or large datasets.
#                 - Check visualizations to understand data distribution.
#                 - Enable feature engineering for complex relationships.
#                 - Adjust tuning parameters for better performance.
                
#                 **Sample Datasets:**
#                 - Classification: Any dataset with a categorical or low-cardinality numeric target.
#                 - Regression: Any dataset with a continuous numeric target.
                
                
#                 """)
#             with st.expander("About This Website and Author", expanded=False):
#                 st.write("""
#                 ### About This Website
#                 ML Companion is a user-friendly web application designed to simplify machine learning workflows. Built with Streamlit, it enables users to upload CSV datasets, preprocess data, visualize insights, and train models for both classification and regression tasks. With features like automated preprocessing, interactive visualizations, and hyperparameter tuning, it caters to beginners and advanced users alike, making data science accessible and efficient.

#                 ### About the Author
#                 I am a passionate data scientist and developer with a keen interest in creating tools that democratize machine learning. With experience in building end-to-end ML pipelines, I developed ML Companion to empower users to explore and model their data effortlessly. Connect with me at namya.vishal.shah.campus@gmail.com for feedback or collaboration opportunities!
#                 """)
#                 st.image("profile.jpg", caption="Author: Namya Vishal Shah", width=200)
            

#     except Exception as e:
#         st.error(f"Error: {e}. Please check your dataset or settings and try again.")
#         if "uploaded_file" in locals():
#             st.write("Dataset Preview (for debugging):")
#             st.dataframe(df.head() if 'df' in locals() else "Data not loaded due to error.")





import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import io
import time
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from ydata_profiling import ProfileReport  # Added for ydata-profiling

# Custom styling for a sexy UI
st.markdown("""
    <style>
    .main {
        background-color: #ac11d6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ac11d6;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stSelectbox>select, .stRadio>label, .stCheckbox>label {
        font-size: 16px;
        color: #333;
    }
    .stProgress>div>div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with logo (simulated with text)
st.markdown("<h1 style='text-align: center; color: #ac11d6;'>ML Companion: Generic Data Insights and Model Tuning</h1>", unsafe_allow_html=True)

# Step 1: Upload a CSV file with sampling option for large datasets
uploaded_file = st.file_uploader("Upload your dataset (CSV file only)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load dataset and optionally sample for large files
        df = pd.read_csv(uploaded_file)
        total_rows = len(df)
        if total_rows > 10000:
            sample_size = st.slider("Sample size for large datasets (rows)", 1000, total_rows, 5000, step=1000)
            st.warning(f"Dataset is large ({total_rows} rows). Sampling {sample_size} rows for performance.")
            df = df.sample(n=sample_size, random_state=42)
        else:
            st.write(f"Dataset size: {total_rows} rows")

        # Step 2: Show dataset preview and stats
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        st.write(f"Total rows containing null values: {df.isnull().any(axis=1).sum()}/{len(df)}")

        # Dataset statistics for guidance
        st.write("### Dataset Statistics")
        st.write("Numerical Columns Skewness:")
        st.write(df.select_dtypes(include=['int64', 'float64']).skew().round(2))
        st.write("Missing Values by Column:")
        st.write(df.isnull().sum())

        # Step 3: Check column types
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        st.write(f"**Categorical Columns**: {categorical_cols}")
        st.write(f"**Numerical Columns**: {numerical_cols}")

        # Step 4: Select target column with help text
        target_column = st.selectbox(
            "Select the target column", 
            df.columns, 
            help="Choose the column you want to predict (e.g., a class for classification or a number for regression)"
        )
        st.write(f"Selected target column: {target_column}")

        if target_column not in df.columns:
            st.error("Please select a valid target column.")
        else:
            # Determine task type with user override
            unique_values = df[target_column].nunique()
            is_categorical = target_column in categorical_cols
            default_classification = is_categorical or (unique_values <= 10 and df[target_column].dtype in ['int64', 'float64'])
            task_type = st.radio(
                "Detected Task Type (override if needed):", 
                ["Classification", "Regression"], 
                index=0 if default_classification else 1,
                help="Auto-detected based on unique values and data type. Adjust if incorrect."
            )
            is_classification = task_type == "Classification"
            st.write(f"Task type set to: {task_type} (based on {unique_values} unique values in '{target_column}')")

            # Encode categorical columns if required
            if st.checkbox("Encode Categorical Columns", help="Convert categorical columns to numerical using Label Encoding"):
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                st.write("Categorical columns have been label encoded.")

            # Encode the target column if selected (only for classification)
            if is_classification and st.checkbox("Encode Target Column", help="Convert target column to numerical for classification"):
                if target_column in categorical_cols:
                    le_target = LabelEncoder()
                    df[target_column] = le_target.fit_transform(df[target_column].astype(str))
                    st.write(f"Target column '{target_column}' has been label encoded.")

            # Step 5: Handle missing values with user choice and suggestions
            if st.checkbox("Handle Missing Values", help="Address missing data in your dataset"):
                st.write("### Missing Values Handling")
                missing_summary = df.isnull().sum()
                st.write("Missing values per column:")
                st.write(missing_summary[missing_summary > 0])
                handle_option = st.radio(
                    "Select a method:",
                    ["Remove rows with null values", "Fill numerical columns with Mean", 
                     "Fill numerical columns with Median", "Fill numerical columns with Mode", 
                     "Drop columns with > 50% missing values"],
                    help="Choose how to handle missing data based on your dataset's needs."
                )

                if handle_option == "Remove rows with null values":
                    df = df.dropna()
                    st.write("Rows with null values removed.")
                elif handle_option == "Fill numerical columns with Mean":
                    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
                    for col in categorical_cols:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    st.write("Missing values filled with mean for numerical columns and mode for categorical columns.")
                elif handle_option == "Fill numerical columns with Median":
                    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                    for col in categorical_cols:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    st.write("Missing values filled with median for numerical columns and mode for categorical columns.")
                elif handle_option == "Fill numerical columns with Mode":
                    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mode().iloc[0])
                    for col in categorical_cols:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    st.write("Missing values filled with mode for both numerical and categorical columns.")
                elif handle_option == "Drop columns with > 50% missing values":
                    df = df.loc[:, df.isnull().mean() < 0.5]
                    st.write("Columns with more than 50% missing values dropped.")
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Step 5.1: Handle Outliers with suggestions
            if st.checkbox("Handle Outliers", help="Manage extreme values in numerical columns"):
                st.write("### Outlier Handling")
                outlier_summary = {}
                for col in numerical_cols:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                    if outliers > 0:
                        outlier_summary[col] = outliers
                if outlier_summary:
                    st.write("Columns with outliers detected:")
                    st.write(pd.Series(outlier_summary))
                outlier_option = st.radio(
                    "Select a method:",
                    ["Remove rows with outliers", "Cap outliers to 5th and 95th percentiles"],
                    help="Choose how to handle outliers based on your dataset's distribution."
                )

                def remove_outliers(df, numerical_cols):
                    for col in numerical_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    return df

                def cap_outliers(df, numerical_cols):
                    for col in numerical_cols:
                        lower_bound = df[col].quantile(0.05)
                        upper_bound = df[col].quantile(0.95)
                        df[col] = np.clip(df[col], lower_bound, upper_bound)
                    return df

                if outlier_option == "Remove rows with outliers":
                    df = remove_outliers(df, numerical_cols)
                    st.write("Rows with outliers removed.")
                elif outlier_option == "Cap outliers to 5th and 95th percentiles":
                    df = cap_outliers(df, numerical_cols)
                    st.write("Outliers capped to 5th and 95th percentiles.")
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Optional log transformation for target and skewed features (regression only)
            if task_type == "Regression":
                if st.checkbox("Apply Log Transformation to Target", help="Log-transform the target for regression if skewed"):
                    df[target_column] = np.log1p(df[target_column])
                    st.write(f"Log transformation applied to target column '{target_column}'")
                if st.checkbox("Apply Log Transformation to Skewed Features", help="Log-transform numerical features with high skewness"):
                    skewed_cols = [col for col in numerical_cols if df[col].skew() > 1 and col != target_column]
                    if skewed_cols:
                        for col in skewed_cols:
                            df[col] = np.log1p(df[col])  # log1p to handle zeros
                        st.write(f"Log transformation applied to: {skewed_cols}")
                    else:
                        st.write("No highly skewed features detected for log transformation.")
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Feature engineering for regression (optional, now generic)
            if task_type == "Regression" and st.checkbox("Perform Feature Engineering", help="Create interactions and polynomial terms for numerical features"):
                if len(numerical_cols) > 1:
                    for i in range(len(numerical_cols)):
                        for j in range(i + 1, len(numerical_cols)):
                            if numerical_cols[i] != target_column and numerical_cols[j] != target_column:
                                new_col = f"{numerical_cols[i]}_{numerical_cols[j]}_Interaction"
                                df[new_col] = df[numerical_cols[i]] * df[numerical_cols[j]]
                                st.write(f"Added interaction term: {new_col}")
                    for col in numerical_cols:
                        if col != target_column:
                            new_col = f"{col}_Squared"
                            df[new_col] = df[col] ** 2
                            st.write(f"Added polynomial term: {new_col}")
                else:
                    st.write("Insufficient numerical columns for feature engineering.")
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                numerical_cols_without_target = [col for col in numerical_cols if col != target_column]

            # Step 6: Standardize all numerical data except target column
            st.write("### Standardizing Data")
            scaler = StandardScaler()
            numerical_cols_without_target = [col for col in numerical_cols if col != target_column]
            if numerical_cols_without_target:
                df[numerical_cols_without_target] = scaler.fit_transform(df[numerical_cols_without_target])
                st.write(f"Standardized columns: {numerical_cols_without_target}")
            else:
                st.warning("No numerical columns to standardize except the target.")

            # Show the dataset after preprocessing
            st.write("### Preprocessed Dataset")
            st.dataframe(df)

            # Step 7: Feature Selection
            selected_features = numerical_cols_without_target
            if st.checkbox("Perform Feature Selection", help="Select features based on their correlation with the target"):
                st.write("### Feature Selection")
                correlation_threshold = st.slider("Set correlation threshold", -1.0, 1.0, 0.2, 
                                                help="Higher threshold excludes weaker correlations; lower includes more features.")
                correlations = df[numerical_cols_without_target].corrwith(df[target_column])
                selected_features = correlations[abs(correlations) >= correlation_threshold].index.tolist()
                st.write(f"Features selected based on correlation threshold ({correlation_threshold}): {selected_features}")

            if not selected_features:
                st.warning("No features selected. Using all numerical features except target.")
                selected_features = numerical_cols_without_target

            # Quick Start mode for beginners
            quick_start = st.checkbox("Quick Start Mode (Basic Settings for Beginners)", 
                                     help="Use default preprocessing and models without tuning for fast results.")
            if quick_start:
                st.write("### Quick Start Mode Enabled")
                tune_params = False
                handle_option = "Fill numerical columns with Median"
                outlier_option = "Cap outliers to 5th and 95th percentiles"
                if st.checkbox("Apply Quick Start Preprocessing"):
                    if df.isnull().sum().any():
                        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                        for col in categorical_cols:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        st.write("Applied quick start missing value handling (Median for numerical, Mode for categorical).")
                    df = cap_outliers(df, numerical_cols)
                    st.write("Applied quick start outlier handling (capped at 5th/95th percentiles).")
                selected_features = numerical_cols_without_target  # Use all features

            # Step 8: Model Training with progress
            if st.checkbox("Train Model", help="Train machine learning models on your preprocessed data"):
                st.write(f"### Training {task_type} Models")
                X = df[selected_features]
                y = df[target_column]
                
                # Train/Test Split with progress spinner
                with st.spinner("Splitting data into training and test sets..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    time.sleep(1)  # Simulate processing time
                st.success("Data split completed.")

                if is_classification:
                    st.write("#### Class Distribution Before SMOTE:")
                    st.write(y_train.value_counts())
                    unique_counts = y_train.value_counts()
                    min_count = unique_counts.min()
                    max_count = unique_counts.max()

                    if min_count / max_count < 0.8:
                        st.write(f"Imbalance detected (minority/majority ratio: {min_count/max_count:.2f})")
                        smote_option = st.radio(
                            "Handle class imbalance with SMOTE?",
                            ["Fully balance to majority class", "Partially balance (custom ratio)", "Skip SMOTE"],
                            help="Oversample minority classes to balance the dataset."
                        )

                        if smote_option == "Fully balance to majority class":
                            with st.spinner("Applying SMOTE to balance classes..."):
                                smote = SMOTE(random_state=42)
                                X_train, y_train = smote.fit_resample(X_train, y_train)
                                time.sleep(1)
                            st.write("SMOTE applied: Classes fully balanced to match majority class.")
                        elif smote_option == "Partially balance (custom ratio)":
                            ratio = st.slider("Select minority-to-majority ratio", 0.1, 1.0, 0.3, step=0.1, 
                                             help="Set the proportion of minority to majority class after SMOTE.")
                            with st.spinner("Applying SMOTE with custom ratio..."):
                                smote = SMOTE(sampling_strategy=ratio, random_state=42)
                                X_train, y_train = smote.fit_resample(X_train, y_train)
                                time.sleep(1)
                            st.write(f"SMOTE applied: Minority class oversampled to {ratio*100}% of majority class.")
                        else:
                            st.write("SMOTE skipped: Proceeding with original class distribution.")
                    else:
                        st.write("No significant imbalance detected. Proceeding without SMOTE.")

                    st.write("#### Class Distribution After SMOTE (if applied):")
                    st.write(y_train.value_counts())

                    if smote_option in ["Fully balance to majority class", "Partially balance (custom ratio)"]:
                        with st.spinner("Preparing SMOTE-applied dataset for download..."):
                            smote_df = pd.DataFrame(X_train, columns=selected_features)
                            smote_df[target_column] = y_train
                            csv_buffer = io.StringIO()
                            smote_df.to_csv(csv_buffer, index=False)
                            time.sleep(1)
                        st.download_button(
                            label="Download SMOTE-Applied Training Dataset",
                            data=csv_buffer.getvalue(),
                            file_name="smote_applied_dataset.csv",
                            mime="text/csv"
                        )

                # Step 9: Visualize Data with interactive Plotly
                # if st.checkbox("Visualize Data Distribution", help="Explore your data with interactive plots"):
                #     st.write("### Data Visualization")
                #     train_df = pd.DataFrame(X_train, columns=selected_features)
                #     train_df[target_column] = y_train

                #     viz_column = st.selectbox("Select a column to visualize", train_df.columns, 
                #                              help="Choose a feature or target to visualize.")
                #     viz_type = st.radio("Choose visualization type", ["Histogram", "Box Plot", "Scatter Plot", 
                #                                                    "Heatmap", "Count Plot", "Violin Plot"],
                #                         help="Select the type of plot for data exploration.")

                #     if viz_type == "Histogram" and viz_column in numerical_cols:
                #         fig = px.histogram(train_df, x=viz_column, nbins=30, title=f"Histogram of {viz_column}")
                #         st.plotly_chart(fig)
                #     elif viz_type == "Count Plot" and viz_column in categorical_cols:
                #         fig = px.histogram(train_df, x=viz_column, title=f"Count Plot of {viz_column}")
                #         st.plotly_chart(fig)
                #     elif viz_type == "Box Plot" and viz_column in numerical_cols:
                #         fig = px.box(train_df, y=viz_column, title=f"Box Plot of {viz_column}")
                #         st.plotly_chart(fig)
                #     elif viz_type == "Scatter Plot" and len(selected_features) > 1:
                #         x_column = st.selectbox("Select x-axis column", selected_features, 
                #                                help="Choose the x-axis feature for the scatter plot.")
                #         y_column = st.selectbox("Select y-axis column", selected_features, 
                #                                help="Choose the y-axis feature for the scatter plot.")
                #         fig = px.scatter(train_df, x=x_column, y=y_column, color=target_column, 
                #                         title=f"Scatter Plot: {x_column} vs {y_column}")
                #         st.plotly_chart(fig)
                #     elif viz_type == "Heatmap" and len(selected_features) > 1:
                #         corr_matrix = train_df[selected_features].corr()
                #         fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                #                        title="Correlation Heatmap of Features")
                #         st.plotly_chart(fig)
                #     elif viz_type == "Violin Plot" and viz_column in numerical_cols:
                #         fig = px.violin(train_df, y=viz_column, box=True, title=f"Violin Plot of {viz_column}")
                #         st.plotly_chart(fig)
                #     else:
                #         st.write("Invalid combination of column and visualization type or insufficient features.")

                # Step 9: Visualize Data with interactive Plotly and ydata-profiling
                if st.checkbox("Visualize Data Distribution", help="Explore your data with interactive plots or a detailed profiling report"):
                    st.write("### Data Visualization")
                    train_df = pd.DataFrame(X_train, columns=selected_features)
                    train_df[target_column] = y_train

                    # Offer a choice between Plotly visualizations and ydata-profiling
                    viz_option = st.radio("Choose visualization method", 
                                        ["Interactive Plotly Charts", "YData Profiling Report"],
                                        help="Select Plotly for custom interactive charts or YData Profiling for a comprehensive EDA report.")

                    if viz_option == "Interactive Plotly Charts":
                        viz_column = st.selectbox("Select a column to visualize", train_df.columns, 
                                                help="Choose a feature or target to visualize.")
                        viz_type = st.radio("Choose visualization type", ["Histogram", "Box Plot", "Scatter Plot", 
                                                                        "Heatmap", "Count Plot", "Violin Plot"],
                                            help="Select the type of plot for data exploration.")

                        if viz_type == "Histogram" and viz_column in numerical_cols:
                            fig = px.histogram(train_df, x=viz_column, nbins=30, title=f"Histogram of {viz_column}")
                            st.plotly_chart(fig)
                        elif viz_type == "Count Plot" and viz_column in categorical_cols:
                            fig = px.histogram(train_df, x=viz_column, title=f"Count Plot of {viz_column}")
                            st.plotly_chart(fig)
                        elif viz_type == "Box Plot" and viz_column in numerical_cols:
                            fig = px.box(train_df, y=viz_column, title=f"Box Plot of {viz_column}")
                            st.plotly_chart(fig)
                        elif viz_type == "Scatter Plot" and len(selected_features) > 1:
                            x_column = st.selectbox("Select x-axis column", selected_features, 
                                                help="Choose the x-axis feature for the scatter plot.")
                            y_column = st.selectbox("Select y-axis column", selected_features, 
                                                help="Choose the y-axis feature for the scatter plot.")
                            fig = px.scatter(train_df, x=x_column, y=y_column, color=target_column, 
                                            title=f"Scatter Plot: {x_column} vs {y_column}")
                            st.plotly_chart(fig)
                        elif viz_type == "Heatmap" and len(selected_features) > 1:
                            corr_matrix = train_df[selected_features].corr()
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                        title="Correlation Heatmap of Features")
                            st.plotly_chart(fig)
                        elif viz_type == "Violin Plot" and viz_column in numerical_cols:
                            fig = px.violin(train_df, y=viz_column, box=True, title=f"Violin Plot of {viz_column}")
                            st.plotly_chart(fig)
                        else:
                            st.write("Invalid combination of column and visualization type or insufficient features.")

                    elif viz_option == "YData Profiling Report":
                        st.write("### Generating YData Profiling Report")
                        with st.spinner("Generating comprehensive EDA report..."):
                            # Generate the profiling report
                            profile = ProfileReport(train_df, title="EDA Report for Training Data", 
                                                    explorative=True, minimal=False)
                            # Save the report to a temporary HTML file
                            temp_file = "eda_report.html"
                            profile.to_file(temp_file)
                            # Display the report in Streamlit using components.html
                            with open(temp_file, "r", encoding="utf-8") as f:
                                report_html = f.read()
                            st.components.v1.html(report_html, height=1000, scrolling=True)
                            # Provide a download button for the report
                            st.download_button(
                                label="Download YData Profiling Report",
                                data=report_html,
                                file_name="eda_report.html",
                                mime="text/html"
                            )
                        st.success("YData Profiling Report generated successfully!")
                
                # Define models based on task type
                classification_models = {
                    "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10, 100]}),
                    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}),
                    "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
                    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]}),
                    "Perceptron": (Perceptron(), {"alpha": [0.0001, 0.001, 0.01]}),
                    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]})
                }

                regression_models = {
                    "Linear Regression": (LinearRegression(), {"fit_intercept": [True, False]}),
                    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
                    "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
                    "SVR": (SVR(), {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto", 0.1, 0.01]}),
                    "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]}),
                    "Gradient Boosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]})
                }

                models = classification_models if is_classification else regression_models
                model_options = list(models.keys()) + ["Compare All Models"]

                # Quick model recommendation
                if st.checkbox("Show Model Recommendations", help="Get suggested models based on dataset size and task"):
                    st.write(f"### Recommended Models for {task_type}")
                    if task_type == "Classification":
                        st.write("- Start with Random Forest or Gradient Boosting for balanced accuracy and robustness.")
                    else:
                        st.write("- Start with Random Forest or Gradient Boosting for high R² on numerical predictions.")
                    if len(df) > 5000:
                        st.write("- Consider using Quick Start Mode or sampling for large datasets.")

                # Model Selection with tuning option and recommendation
                model_option = st.selectbox(
                    f"Select a {task_type.lower()} model or compare all", 
                    model_options, 
                    help="Choose a single model or compare all to find the best performer."
                )
                tune_params = st.checkbox("Perform Hyperparameter Tuning (faster with Randomized Search)", value=not quick_start, 
                                        help="Optimize model performance, but may take longer. Disable for quick results.")
                if tune_params:
                    n_iter = st.slider("Number of iterations for Randomized Search", 5, 30, 20, 
                                      help="More iterations improve tuning but increase runtime.")
                    cv_folds = st.slider("Number of cross-validation folds", 3, 10, 5, 
                                        help="More folds ensure robust results but increase computation time.")

                # Function to evaluate a model with performance monitoring and predictions
                def evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, tune=True, is_regression=False):
                    start_time = time.time()
                    if tune and param_grid:
                        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter if tune_params else 5, 
                                                  cv=cv_folds if tune_params else 3, random_state=42, n_jobs=-1)
                        with st.spinner(f"Tuning {model.__class__.__name__}..."):
                            search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                    else:
                        best_model = model
                        with st.spinner(f"Training {model.__class__.__name__} with default parameters..."):
                            best_model.fit(X_train, y_train)
                        best_params = "Default Parameters"
                    
                    training_time = time.time() - start_time
                    st.write(f"Training time for {model.__class__.__name__}: {training_time:.2f} seconds")

                    y_pred = best_model.predict(X_test)
                    if is_regression:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        baseline_pred = np.full_like(y_test, y_train.mean())
                        baseline_r2 = r2_score(y_test, baseline_pred)
                        return {
                            "Best Parameters": best_params,
                            "Predictions": y_pred,  # Store predictions here for regression
                            "Mean Squared Error": mse,
                            "R² Score": r2,
                            "Mean Absolute Error": mae,
                            "Baseline R² (Mean Predictor)": baseline_r2,
                            "Training Time (s)": training_time
                        }
                    else:
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        cm = confusion_matrix(y_test, y_pred)
                        return {
                            "Best Parameters": best_params,
                            "Predictions": y_pred,  # Store predictions here for classification
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 Score": f1,
                            "Confusion Matrix": cm,
                            "Training Time (s)": training_time
                        }

                # Train and evaluate based on user selection with progress
                if model_option == "Compare All Models":
                    st.write(f"### Training and Evaluating All {task_type.lower()} Models")
                    results = {}
                    progress_bar = st.progress(0)
                    total_models = len(models)
                    for i, (name, (model, param_grid)) in enumerate(models.items()):
                        with st.spinner(f"Training {name}..."):
                            results[name] = evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, 
                                                         tune=tune_params, is_regression=not is_classification)
                        progress_bar.progress((i + 1) / total_models)

                    # Display results in an interactive table
                    comparison_df = pd.DataFrame(results).T
                    st.write(f"### Comparison of All {task_type} Models")
                    st.dataframe(comparison_df.style.format({"R² Score": "{:.3f}", "Mean Squared Error": "{:.2f}", 
                                                           "Mean Absolute Error": "{:.2f}", "Accuracy": "{:.3f}", 
                                                           "Training Time (s)": "{:.2f}"}))

                    # Download comparison results
                    buffer = io.StringIO()
                    comparison_df.to_csv(buffer, index=True)
                    st.download_button(
                        label=f"Download {task_type} Model Comparison Results",
                        data=buffer.getvalue(),
                        file_name=f"{task_type.lower()}_model_comparison_results.csv",
                        mime="text/csv"
                    )

                    # Visualize results
                    if task_type == "Regression":
                        st.write("### Predicted vs. Actual Plot")
                        first_model = list(models.keys())[0]
                        fig = px.scatter(x=y_test, y=results[first_model]["Predictions"], 
                                        trendline="ols", title="Predicted vs. Actual Values")
                        st.plotly_chart(fig)
                    else:
                        st.write("### Confusion Matrix (First Model)")
                        cm = results[list(models.keys())[0]]["Confusion Matrix"]
                        fig = ff.create_annotated_heatmap(cm, x=list(range(len(cm))), y=list(range(len(cm))), 
                                                        colorscale='Viridis')
                        st.plotly_chart(fig)

                else:
                    # Train a single model with progress
                    with st.spinner(f"Training {model_option}..."):
                        model, param_grid = models[model_option]
                        result = evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, 
                                              tune=tune_params, is_regression=not is_classification)
                    st.success(f"Training {model_option} completed.")

                    # Display results
                    st.write(f"### Results for {model_option}:")
                    for metric, value in result.items():
                        if metric != "Confusion Matrix" and metric != "Predictions":
                            st.write(f"{metric}: {value}")
                        elif metric == "Confusion Matrix":
                            st.write("Confusion Matrix:")
                            fig = ff.create_annotated_heatmap(value, x=list(range(len(value))), y=list(range(len(value))), 
                                                            colorscale='Viridis')
                            st.plotly_chart(fig)

                    # Download model details and predictions
                    buffer = io.StringIO()
                    buffer.write(f"Model: {model_option}\n")
                    for metric, value in result.items():
                        if metric != "Confusion Matrix" and metric != "Predictions":
                            buffer.write(f"{metric}: {value}\n")
                    st.download_button(
                        label="Download Model Details",
                        data=buffer.getvalue(),
                        file_name=f"{model_option}_details.txt",
                        mime="text/plain"
                    )

                    # Download predictions
                    if task_type == "Regression":
                        predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": result["Predictions"]})
                        csv_buffer = io.StringIO()
                        predictions_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv_buffer.getvalue(),
                            file_name=f"{model_option}_predictions.csv",
                            mime="text/csv"
                        )

            # Save/Load State
            if st.checkbox("Save Current Settings", help="Save your preprocessing and model settings for later use"):
                settings = {
                    "target_column": target_column,
                    "task_type": task_type,
                    "encode_categorical": st.session_state.get("Encode Categorical Columns", False),
                    "encode_target": st.session_state.get("Encode Target Column", False),
                    "handle_missing": st.session_state.get("Handle Missing Values", False),
                    "missing_method": handle_option if "handle_missing" in locals() else None,
                    "handle_outliers": st.session_state.get("Handle Outliers", False),
                    "outlier_method": outlier_option if "handle_outliers" in locals() else None,
                    "log_target": st.session_state.get("Apply Log Transformation to Target", False),
                    "log_features": st.session_state.get("Apply Log Transformation to Skewed Features", False),
                    "feature_engineering": st.session_state.get("Perform Feature Engineering", False),
                    "feature_selection": st.session_state.get("Perform Feature Selection", False),
                    "correlation_threshold": correlation_threshold if "feature_selection" in locals() else 0.2,
                    "tune_params": tune_params,
                    "n_iter": n_iter if tune_params else 5,
                    "cv_folds": cv_folds if tune_params else 3,
                    "model_option": model_option,
                    "quick_start": quick_start
                }
                import json
                settings_json = json.dumps(settings)
                st.download_button(
                    label="Download Settings",
                    data=settings_json,
                    file_name="ml_settings.json",
                    mime="application/json"
                )

            if st.checkbox("Load Previous Settings", help="Load saved preprocessing and model settings"):
                uploaded_settings = st.file_uploader("Upload settings file (JSON)", type=["json"])
                if uploaded_settings:
                    settings = json.load(uploaded_settings)
                    st.session_state["target_column"] = settings["target_column"]
                    st.session_state["task_type"] = settings["task_type"]
                    st.session_state["Encode Categorical Columns"] = settings["encode_categorical"]
                    st.session_state["Encode Target Column"] = settings["encode_target"]
                    st.session_state["Handle Missing Values"] = settings["handle_missing"]
                    if settings["handle_missing"]:
                        handle_option = settings["missing_method"]
                    st.session_state["Handle Outliers"] = settings["handle_outliers"]
                    if settings["handle_outliers"]:
                        outlier_option = settings["outlier_method"]
                    st.session_state["Apply Log Transformation to Target"] = settings["log_target"]
                    st.session_state["Apply Log Transformation to Skewed Features"] = settings["log_features"]
                    st.session_state["Perform Feature Engineering"] = settings["feature_engineering"]
                    st.session_state["Perform Feature Selection"] = settings["feature_selection"]
                    if settings["feature_selection"]:
                        correlation_threshold = settings["correlation_threshold"]
                    st.session_state["tune_params"] = settings["tune_params"]
                    if settings["tune_params"]:
                        n_iter = settings["n_iter"]
                        cv_folds = settings["cv_folds"]
                    st.session_state["model_option"] = settings["model_option"]
                    st.session_state["quick_start"] = settings["quick_start"]
                    st.success("Settings loaded successfully. Refresh the page to apply.")

            # About/Help section
            with st.expander("About & Help", expanded=False):
                st.write("""
                ### ML Companion: Generic Data Insights and Model Tuning
                This app helps you preprocess, visualize, and train machine learning models on any CSV dataset.
                
                **Features:**
                - Handles classification and regression tasks automatically.
                - Preprocesses data (missing values, outliers, encoding, standardization, log transformations).
                - Offers feature engineering, selection, and interactive visualizations.
                - Trains multiple models with optional hyperparameter tuning.
                - Saves/loads settings for reproducibility.
                
                **Quick Tips:**
                - Use "Quick Start Mode" for beginners or large datasets.
                - Check visualizations to understand data distribution.
                - Enable feature engineering for complex relationships.
                - Adjust tuning parameters for better performance.
                
                **Sample Datasets:**
                - Classification: Any dataset with a categorical or low-cardinality numeric target.
                - Regression: Any dataset with a continuous numeric target.
                
                
                """)
            with st.expander("About This Website and Author", expanded=False):
                st.write("""
                ### About This Website
                ML Companion is a user-friendly web application designed to simplify machine learning workflows. Built with Streamlit, it enables users to upload CSV datasets, preprocess data, visualize insights, and train models for both classification and regression tasks. With features like automated preprocessing, interactive visualizations, and hyperparameter tuning, it caters to beginners and advanced users alike, making data science accessible and efficient.

                ### About the Author
                I am a passionate data scientist and developer with a keen interest in creating tools that democratize machine learning. With experience in building end-to-end ML pipelines, I developed ML Companion to empower users to explore and model their data effortlessly. Connect with me at namya.vishal.shah.campus@gmail.com for feedback or collaboration opportunities!
                """)
                st.image("profile.jpg", caption="Author: Namya Vishal Shah", width=200)
            

    except Exception as e:
        st.error(f"Error: {e}. Please check your dataset or settings and try again.")
        if "uploaded_file" in locals():
            st.write("Dataset Preview (for debugging):")
            st.dataframe(df.head() if 'df' in locals() else "Data not loaded due to error.")

