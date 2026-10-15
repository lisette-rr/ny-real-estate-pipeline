import data_cleaning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def encode_data(X_train:pd.DataFrame, X_test:pd.DataFrame):
    """
    Serves to encode the DataFrame of data, more specifically the
    columns type and city. OneHot Encoding is used.

    Parameters:
    - X_train: the training data
    - X_test: the training data

    Return: 
    - X_train_ohenc: one hot encoded X_train data
    - X_test_ohenc: one hot encoded X_test data
    """

    # categorical columns to oh encode
    categorical_cols = ['type', 'city']

    # one hot encoding the data
    oh_encoder = OneHotEncoder(
        drop='first',
        sparse_output=False
    )

    transformer_ohenc = ColumnTransformer(
        transformers=[
            ('oh_enc', oh_encoder, categorical_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    transformer_ohenc.fit(X = X_train)

    X_train_ohenc = \
    pd.DataFrame(data= transformer_ohenc.transform(X= X_train), 
                columns= transformer_ohenc.get_feature_names_out(),
                index= X_train.index)

    X_test_ohenc = \
    pd.DataFrame(data= transformer_ohenc.transform(X= X_test), 
                columns= transformer_ohenc.get_feature_names_out(),
                index= X_test.index)
    
    return X_train_ohenc, X_test_ohenc


def train_and_test(X_train_ohenc, y_train, X_test_ohenc, y_test):
    """
    Method serves to contain the training and testing of both models.
    Also prints performance metrics for comparison of models.

    Parameters:
    - X_train_ohenc: the X training data that is one hot encoded
    - y_train: the y training data
    - X_test_ohenc: the X testing data that is one hot encoded
    - y_test: the y testing data

    Return: None
    """

    print("Training models...")

    # linear regressor 
    lr_model = LinearRegression()
    lr_model.fit(X_train_ohenc, y_train)
    
    lr_preds = lr_model.predict(X_test_ohenc)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)

    # random forest regressor 
    rf_model = RandomForestRegressor(n_estimators=100, 
                                     random_state=123)
    rf_model.fit(X_train_ohenc, y_train)
    
    rf_preds = rf_model.predict(X_test_ohenc)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    # performance metrics
    print("FINAL MODEL RESULTS")
    print("--- Linear Regression ---")
    print(f"MAE: ${lr_mae:,.2f}")
    print(f"R2 Score: {lr_r2:.4f}\n")

    print("--- Random Forest Regressor ---")
    print(f"MAE: ${rf_mae:,.2f}")
    print(f"R2 Score: {rf_r2:.4f}")


def optimal_lin_reg_hyperparameters(X_train_ohenc, y_train):
    """
    Tunes alpha for Linear Regression using Ridge Regressor
    Prints the results and returns the best estimator

    Parameters:
    - X_train_ohenc: the X training data that is one hot encoded
    - y_train: the y training data

    Return:
    - the best Ridge estimator 
    """

    print("\nTuning Ridge Regressor...")

    # Define a grid of alpha values (the penalty strength)
    ridge_param_grid = {'alpha': np.linspace(0.01, 5, 100)}

    # ridge regressor initialization
    ridge_base = Ridge(max_iter=100000)

    # grid search with 5-cross validation
    ridge_search = GridSearchCV(
        estimator=ridge_base, 
        param_grid=ridge_param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1
    )

    # fits ridge regressor
    ridge_search.fit(X_train_ohenc, y_train)

    # prints results
    print(f"Optimal Ridge Alpha: {ridge_search.best_params_}")
    print(f"Optimized Ridge R2: {ridge_search.best_score_:.4f}")

    return ridge_search.best_estimator_


def optimal_rand_tree_hyperparameters(X_train_ohenc, y_train):
    """
    Tunes hyperparameters for Random Forest.
    Prints the results and returns the best estimator

    Parameters:
    - X_train_ohenc: the X training data that is one hot encoded
    - y_train: the y training data

    Return:
    - the best RandomForestRegressor estimator
    """

    print("\nTuning Random Forest hyperparameters...")

    # parameter grid to explore
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # clean random forest base
    rf_base = RandomForestRegressor(random_state=123)

    # grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf_base, 
        param_grid=param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1
    )

    # fit the grid search on training data
    grid_search.fit(X_train_ohenc, y_train)

    # prints results and score
    print(f"\nOptimal Parameters Found: {grid_search.best_params_}")
    print(f"Grid Search CV Best R2: {grid_search.best_score_:.4f}\n")

    return grid_search.best_estimator_

def evaluate_with_cross_validation(model, X, y, model_name):
    """
    Computes 5-fold cross-validation scores on the training set.

    Parameters:
    - model: the model being evaluated
    - X: the X training data
    - y: the y training data
    - model_name: the name of the model passed

    Return: None
    """
    # cv=5 splits the training data 5 times to get an array of 5 R2 scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    print(f"--- {model_name} 5-Fold Cross-Validation ---")
    print(f"All Fold R2 Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean Cross-Validated R2 Score: {cv_scores.mean():.4f}\n")



def main():
    
    # using the data cleaning function from the data cleaning file
    cleaned_df = data_cleaning.load_and_clean_data()

    # splitting the data
    y = cleaned_df['lastSoldPrice']
    X = cleaned_df.drop(columns=['lastSoldPrice'])

    # training and testing split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


    # encoding the X train and X test data
    X_train_ohenc, X_test_ohenc = encode_data(X_train, X_test)
    
    # trains and tests the data with default hyperparameters
    train_and_test(X_train_ohenc, y_train, X_test_ohenc, y_test)

    # optimal hyperparameters for linear regression and rand tree. gets the best estimator
    best_ridge = optimal_lin_reg_hyperparameters(X_train_ohenc, y_train)
    best_rf = optimal_rand_tree_hyperparameters(X_train_ohenc, y_train)
    
    
    print("\n\nFINAL PERFORMANCE SCORES")
    

    # evaluate final tuned models across all folds
    evaluate_with_cross_validation(best_ridge, X_train_ohenc, y_train, "Optimized Ridge Regression")
    evaluate_with_cross_validation(best_rf, X_train_ohenc, y_train, "Optimized Random Forest")

    # results of optimal models of each
    print("--- Final Holdout Test Set Performance ---")
    
    print("Ridge Regressor")
    ridge_final_preds = best_ridge.predict(X_test_ohenc)
    print(f"Tuned Ridge Regressor Test MAE: ${mean_absolute_error(y_test, ridge_final_preds):,.2f}")
    print(f"Tuned Ridge Regressor Test R2 Score: {r2_score(y_test, ridge_final_preds):.4f}")
    
    print("\nRandom Forest")
    rf_final_preds = best_rf.predict(X_test_ohenc)
    print(f"Tuned Random Forest Test MAE: ${mean_absolute_error(y_test, rf_final_preds):,.2f}")
    print(f"Tuned Random Forest Test R2 Score: {r2_score(y_test, rf_final_preds):.4f}")

if __name__ == '__main__':
    main()