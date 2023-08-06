import pickle
from random import sample

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from scipy import linalg
from scipy.stats.mstats import pearsonr, spearmanr
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.svm import SVC
from sklearn.utils import resample


def train_svm_with_best_param(params, X_train, y_train):
    clf_svm_best = SVC(
        random_state=42, C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    clf_svm_best.fit(X_train, y_train)
    return clf_svm_best


def find_svm_best_param(X_train, y_train):
    """A function to find the best parametrs of SVM classifer
    """
    param_grid = [
        {
            'C': [0.05, 0.1, 0.2, 0.3, 0.5],
            'gamma':[0.01, 0.05, 0.08],
            'kernel': ['rbf'],


        },
    ]
    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=20,
        scoring='f1',
        verbose=1,
        n_jobs=-1,

    )
    optimal_params.fit(X_train, y_train)
    return optimal_params.best_params_


def train_svm(X_train_transformed, y_train):
    """ A function to train SVM
    """
    # Find the best parameters
    params = find_svm_best_param(X_train_transformed, y_train)

    svm_clf = SVC(
        random_state=42, C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    svm_clf.fit(X_train_transformed, y_train)

    filename = 'svm_clf.sav'
    pickle.dump(svm_clf, open(filename, 'wb'))


def find_xgboost_best_param(X_train_transformed, y_train):
    """A function to find the best parameter of XGBoost
    """
    param_grid = {
        'max_depth': [1, 2, 3, 4],
        'learning_rate': [0.1, 0.01, 0.05],
        'gamma': [0, 0.1, 0.15, 0.25, 1, 1.1],
        'reg_lambda': [1, 2, 3, 5, 10],
        'scale_pos_weight': [0.05],
    }

    optimal_params = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic',
                                    seed=42),
        param_grid=param_grid,
        scoring='roc_auc',
        verbose=0,
        n_jobs=10,
        cv=3)

    optimal_params.fit(X_train_transformed, y_train, verbose=0)
    return optimal_params


def train_xgboost(X_train_transformed, y_train):
    """A function to train xgboost classifer
    """
    # Find the best parameter
    optimal_params = find_xgboost_best_param(X_train_transformed, y_train)
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42,
                                gamma=optimal_params.best_params_['gamma'],
                                learning_rate=optimal_params.best_params_[
                                    'learning_rate'],
                                max_depth=optimal_params.best_params_[
                                    'max_depth'],
                                reg_lambda=optimal_params.best_params_[
                                    'reg_lambda'],
                                scale_pos_weight=optimal_params.best_params_['scale_pos_weight'])

    # Train XGBoost
    clf_xgb.fit(X_train_transformed, y_train, verbose=False, eval_metric='auc')

    # Save the trained xgboost model
    filename = 'clf_xgboost_model.sav'
    pickle.dump(clf_xgb, open(filename, 'wb'))


def random_over_sampler(X_train_transformed, y_train):
    """A function to increase the training data
    """
    ROS = RandomOverSampler()
    X_train_transformed, y_train = ROS.fit_resample(
        X_train_transformed, y_train)
    return X_train_transformed, y_train


def normalize_features(X_train):
    """A function to normalize the input featuers
    """
    sc = MinMaxScaler()
    X_train_transformed = pd.DataFrame(sc.fit_transform(X_train))
    return X_train_transformed


def main():

    # Read data
    X_train, y_train = pd.read_csv(
        './src/dataset/JPEGAI-quality-metrics-on-IQA.csv', header=0, sep=",")

    # Normalize features
    X_train_transformed = normalize_features(X_train)

    # Increase training examples
    X_train_transformed, y_train = random_over_sampler(
        X_train_transformed, y_train)

    # Train SVM
    train_svm(X_train_transformed, y_train)

    # Train xgboost
    train_xgboost(X_train_transformed, y_train)


if __name__ == '__main__':
    main()
