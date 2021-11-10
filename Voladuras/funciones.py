import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve, ShuffleSplit

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_SBR=True):
        self.add_SBR = add_SBR 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.add_SBR:
            X = X.copy()
            X["SBR"] = X.B / X.S
            return X.drop(["B", "S"], axis=1)
        else:
            return X
        
def remove_outliers(df, columns, stds=3, n=1):
    '''Remueve outliers de un DataFrame tomando como criterio el número de desviaciones estándar del valor a partir de la media
    df: DataFrame del que se requiere remover outliers, pandas.DataFrame
    columns: lista que contiene los nombres de las columnas a analizar, list
    stds: número de desviaciones estándar a partir de la media que se considera como outlier, int o float
    n: número de iteraciones, int'''
    
    df_clean = df.copy()
    for _ in range(n):
        df_temp = df_clean.copy()
        for col in columns:
            rows = df_temp[abs(df_temp[col] - df_temp[col].mean()) > stds * df_temp[col].std()].index
            df_clean.drop(rows, inplace=True, errors="ignore")
    print(f"Se elimminaron {len(df) - len(df_clean)} filas, consideras como outliers")
    return df_clean

def graph_distrib(df, rows, cols, figsize):
    plt.figure()
    fig, ax = plt.subplots(rows, cols, figsize = figsize)
    ax = ax.reshape(-1)
    for index, col in enumerate(df):
        if pd.api.types.is_float_dtype(df[col]):
            sns.histplot(data = df, x = col, kde = True, ax = ax[index])
        elif pd.api.types.is_integer_dtype(df[col]):
            sns.histplot(data = df, x = col, discrete = True, ax = ax[index])
        elif pd.api.types.is_string_dtype(df[col]):
            sns.countplot(data = df, y = col, ax = ax[index], order=sorted(df[col].unique()))
        ax[index].set_title(col, fontsize = 15)
    plt.tight_layout()
    
def plot_predicted_vs_measured(estimator, X, y, cv=None, alpha=0.5):
    predicted = cross_val_predict(estimator, X, y, cv=cv)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, alpha=alpha)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title(str(estimator.__class__).replace("'>", '').split('.')[-1])
    plt.show()
    
def display_validation_scores(estimator, X, y, cv=10):
    scores = np.sqrt(-cross_val_score(estimator, X, y, scoring="neg_mean_squared_error", cv=cv))
    print("Scores:", scores)
    print(f"Mean: {scores.mean():.3}")
    print(f"Standar deviation: {scores.std():.3}")
    

def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None, scoring="neg_mean_squared_error",
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(f'''Learning Curves: {str(estimator.__class__).replace("'>", "").split(".")[-1]}''')
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring=scoring,
                       return_times=True)
    
    if scoring == "neg_mean_squared_error":
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def display_scores(estimator, X, y, validation=False, cv=None):
    print("Train".center(64, "-"))
    print(f"Score: {np.sqrt(mean_squared_error(y, estimator.predict(X))):.3}")
    if validation:
        print("Validation".center(64, "-"))
        display_validation_scores(estimator, X, y, cv=cv)
        plot_predicted_vs_measured(estimator, X, y, cv=cv)
