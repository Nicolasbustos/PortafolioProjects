"""
Soft Voting/Majority Rule classifier and Voting regressor.

This module contains:
 - A Soft Voting/Majority Rule classifier for classification estimators.
 - A Voting regressor for regression estimators.
"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>,
#          Ramil Nugmanov <stsouko@live.ru>
#          Mohamed Ali Jamaoui <m.ali.jamaoui@gmail.com>
#
# License: BSD 3 clause

from abc import abstractmethod

import numpy as np

from joblib import Parallel

from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils.fixes import delayed


class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(%d of %d) Processing %s' % (idx, total, name)

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights)
                if est[1] != 'drop']

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X) for est in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(
                        clone(clf), X, y,
                        sample_weight=sample_weight,
                        message_clsname='Voting',
                        message=self._log_message(names[idx],
                                                  idx + 1, len(clfs))
                )
                for idx, clf in enumerate(clfs) if clf != 'drop'
            )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == 'drop' else next(est_iter)
            self.named_estimators_[name] = current_est

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return class labels or probabilities for each estimator.

        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            Input samples

        y : ndarray of shape (n_samples,), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return super().fit_transform(X, y, **fit_params)

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute."
                .format(self.__class__.__name__)
            ) from nfe

        return self.estimators_[0].n_features_in_

    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        return _VisualBlock('parallel', estimators, names=names)

    def _more_tags(self):
        return {"preserves_dtype": []}


class VotingRegressor(RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : Bunch
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2)])
    >>> print(er.fit(X, y).predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]
    """
    @_deprecate_positional_args
    def __init__(self, estimators, *, weights=None, n_jobs=None,
                 verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            if len(y.shape) > 2 and y.shape[2] > 1:
                raise ValueError('y argument should be a 1 or 2D array-like,'
                                 f'got array with shape {y.shape} instead.')
            else:
                self.multioutput_ = True
        else:
            self.multioutput_ = False
            y = column_or_1d(y, warn=True)

        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        axis = 2 if self.multioutput_ else 1
        return np.average(self._predict(X), axis=axis,
                          weights=self._weights_not_none).T

    def transform(self, X):
        """Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions: ndarray of shape (n_samples, n_classifiers)
            Values predicted by each regressor.
        """
        check_is_fitted(self)
        return self._predict(X)