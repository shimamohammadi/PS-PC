import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from scipy.stats.mstats import pearsonr
from collections import Counter
import imblearn.combine


def train_svm(X_train, y_train):
    clf_svm = SVC(random_state=42)
    clf_svm.fit(X_train, y_train)
    return clf_svm


def train_svm_with_best_param(params, X_train, y_train):
    clf_svm_best = SVC(
        random_state=42, C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    clf_svm_best.fit(X_train, y_train)
    return clf_svm_best


def prediction_accuracy(clf_svm, X_test, y_test):
    prediction = clf_svm.predict(X_test)
#     print(f" accuracy score is: {metrics.accuracy_score(y_test, y_pred=prediction)}")
#     print(f" precision score is: {metrics.precision_score(y_test, y_pred=prediction)}")
#     print(f" recall score is: {metrics.recall_score(y_test, y_pred=prediction)}")
    print(
        f" classification score is:\n {metrics.classification_report(y_test, y_pred=prediction)}")
#     print(f" score is : {clf_svm.score(X_test, y_test)}")


def plot_confusion_mtx(clf_svm, X_test, y_test):
    plot_confusion_matrix(clf_svm,
                          X_test,
                          y_test,
                          values_format='d',
                          )


def optimize_param(X_train, y_train):
    param_grid = [
        {
            'C': [0.5, 1, 10, 100],
            'gamma':['scale', 1, 0.1, 0.01, 0.001, 0.001],
            'kernel': ['rbf']
        },
    ]
    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=0
    )
    optimal_params.fit(X_train, y_train)
    return optimal_params.best_params_


def merging(quality_metrics, kld_labels):
    first_df = pd.merge(kld_labels, quality_metrics)
    quality_metrics.rename({'Pair_a': 'Pair_b'}, axis='columns', inplace=True)
    second_df = pd.merge(kld_labels, quality_metrics)
    df = pd.merge(first_df, second_df, on=[
                  'Pair_a', 'Pair_b'], suffixes=['_a', '_b'])
    df.drop(['Defer_b', 'ms_ssim1_a', 'ms_ssim1_b',
             'psnr_hvs[1]_a', 'psnr_hvs[1]_b',
             'lpips_yuv_a', 'lpips_yuv_b', 'lpips_y_a', 'lpips_y_b',
             'dists_yuv_a', 'dists_yuv_b', 'dists_y_a', 'dists_y_b'], axis=1, inplace=True)
    df.rename({'Defer_a': 'Defer'}, axis='columns', inplace=True)
    X = df.drop(['Defer', 'Pair_a', 'Pair_b'], axis=1).copy()
    y = df['Defer'].copy()
    return X, y


def main():
    quality_metrics = pd.read_csv(
        'src/dataset/JPEGAI-quality-metrics-on-IQA.csv', header=0, sep=",")
    kld_labels = pd.read_csv(
        'src/dataset/KLD_labeling_on_IQA.csv', header=0, sep=",")
    X, y = merging(quality_metrics, kld_labels)
    SMOTEENN = imblearn.combine.SMOTEENN()

    print('Original dataset shape %s' % Counter(y))

    X_res, y_res = SMOTEENN.fit_resample(X, y)

    print('After undersample dataset shape %s' % Counter(y_res))

    y_res.value_counts()
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=15, stratify=y_res)
    print("SVM under SMOTEENN .....")
    clf_svm = train_svm(X_train, y_train)
    prediction_accuracy(clf_svm, X_test, y_test)

    print("\n\n\nSVM under SMOTEENN and with best params.....")
    params = optimize_param(X_train, y_train)
    clf_svm_best = train_svm_with_best_param(params, X_train, y_train)
    prediction_accuracy(clf_svm_best, X_test, y_test)
    plot_confusion_mtx(clf_svm_best, X_test, y_test)

    prediction = clf_svm_best.predict(X_test)
    pearson_correlations, _ = pearsonr(prediction, y_test)
    pearson_correlations


if __name__ == '__main__':
    main()
