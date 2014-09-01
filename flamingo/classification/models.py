import cPickle as pickle
import numpy as np
import pandas
import sklearn.linear_model
import pystruct.models, pystruct.learners

from features import relativelocation

from utils import *

class LogisticRegression(sklearn.linear_model.LogisticRegression):

    def fit(self, X, Y):
        X, Y = linearize_data(X, Y)
        return super(LogisticRegression, self).fit(X, Y)

    def predict(self, X):
        X = linearize_data(X)
        return super(LogisticRegression, self).predict(X)

    def score(self, X, Y):
        Y = linearize_data(Y=Y)
        return super(LogisticRegression, self).score(X, Y)

class LogisticRegressionRLP(LogisticRegression):

    rlp_maps = None
    rlp_stats = None
    rlp_init = False
    n = 1

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, rlp_maps=None, rlp_stats=None):

        super(LogisticRegressionRLP, self).__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                                    class_weight=class_weight, random_state=random_state)

        self.rlp_maps = rlp_maps
        self.rlp_stats = rlp_stats

    def fit(self, X, X_rlp, Y):
        X = [np.concatenate((Xi, Xi_rlp), axis=-1) for Xi, Xi_rlp in zip(X, X_rlp)]
        return super(LogisticRegressionRLP, self).fit(X, Y)

    def predict(self, X):
        shp = X.shape[:2]
        X = self.compute_rlp_features(X)
        Y = super(LogisticRegressionRLP, self).predict(X)
        with open('/flamingo/debug/%d_final.pkl' % self.n,'wb') as fp: # BAS: WHAT THE HELL IS THIS ?!?!
            pickle.dump(Y.reshape(shp), fp)
        return Y

    def score(self, X, Y):
        scores = []
        for Xi, Yi in zip(X, Y):
            with open('/flamingo/debug/%d_annotated.pkl' % self.n,'wb') as fp: # BAS: OH, SYSTEMATIC BULLSHIT...
                pickle.dump(Yi, fp)
            scores.append(super(LogisticRegressionRLP, self).score(Xi, Yi))
            self.n += 1
        return np.mean(scores)

    def compute_rlp_features(self, X):
        if self._rlp_initialized():

            # set relative location features to zero
            n = len(self.rlp_maps.keys())
            X_ext = np.concatenate((X, np.zeros(list(X.shape[:-1]) + [n])), axis=-1)

            # first prediction round
            Y = super(LogisticRegressionRLP, self).predict(X_ext).reshape(X.shape[:-1])
            with open('/flamingo/debug/%d_firstprediction.pkl' % self.n,'wb') as fp: # BAS: SHOW ME THE WAY TO THE NEXT DEBUG FLAG
                pickle.dump(Y, fp)

            # voting based on first prediction round and relative location prior
            votes, ivote = relativelocation.vote_image(Y, self.rlp_maps)
            with open('/flamingo/debug/%d_votes.pkl' % self.n,'wb') as fp: # BAS: REALLY? REALLY, REALLY, REALLY?
                pickle.dump(ivote, fp)

            # translate voted probabilities to relative location features
            for c in votes.items:
                s = 'prob_%s' % c
                avg = self.rlp_stats['avg'][s]
                std = np.sqrt(self.rlp_stats['var'][s])
                val = votes[c].as_matrix()[...,np.newaxis]

                X = np.concatenate((X, (val - avg) / std), axis=-1)

        return X

    def _rlp_initialized(self):
        return self.rlp_maps is not None and self.rlp_stats is not None

class SupportVectorMachine(sklearn.svm.LinearSVC):

    def fit(self, X, Y):
        X, Y = linearize_data(X, Y)
        return super(SupportVectorMachine, self).fit(X, Y)

    def predict(self, X):
        X = linearize_data(X)
        return super(SupportVectorMachine, self).predict(X)

    def score(self, X, Y):
        Y = linearize_data(Y=Y)
        return super(SupportVectorMachine, self).score(X, Y)

class ConditionalRandomField(pystruct.learners.OneSlackSSVM):

    def __init__(self, model, max_iter=10000, C=1.0, check_constraints=False, verbose=0, 
                 negativity_constraint=None, n_jobs=1, break_on_bad=False, show_loss_every=0, tol=1e-3,
                 inference_cache=0, inactive_threshold=1e-5, inactive_window=50, logger=None, cache_tol='auto',
                 switch_to=None, clist=None):
        
        super(ConditionalRandomField,self).__init__(model, max_iter=max_iter, C=C,
                                                    check_constraints=check_constraints, verbose=verbose,
                                                    negativity_constraint=negativity_constraint, n_jobs=n_jobs,
                                                    break_on_bad=break_on_bad, show_loss_every=show_loss_every,
                                                    tol=tol, inference_cache=inference_cache,
                                                    inactive_threshold=inactive_threshold,
                                                    inactive_window=inactive_window, logger=logger,
                                                    cache_tol=cache_tol, switch_to=switch_to)

        self.clist = clist

    def fit(self, X, Y):
        self.clist = list({c for y in Y for c in y.ravel()})
        self.clist.sort()

        X = X[:2]
        Y = Y[:2]

        Y = self._labels2int(Y)

        return super(ConditionalRandomField,self).fit(X,Y)

    def predict(self, X):
        Y = super(ConditionalRandomField,self).predict(X)
        
        return self._int2labels(Y)

    def score(self, X, Y):
        Y = self._labels2int(Y)

        return super(ConditionalRandomField,self).score(X,Y)

    def _labels2int(self, Y):

        Yint = Y.copy()
        for i, y in enumerate(Yint):
            for j, c in enumerate(self.clist):
                y[y == c] = j
        
        return [y.astype(int, copy=False) for y in Yint]

    def _int2labels(self, Y):
        
        Ystr = Y.copy()
        for i, y in enumerate(Y):
            for j, c in enumerate(self.clist):
                Ystr[i][y == j] = c

        return Ystr

def get_model(model_type='LR', n_states=None, n_features=None, rlp_maps=None, rlp_stats=None):
    '''Returns a bare model object

    Parameters
    ----------
    model_type : string, optional
        String indicating the type of model to be constructed.
        LR = Logistic Regressor (default), LR_RLP = Logistic Regressor with Relative Location Prior, SVM = Support Vector Machine, CRF = Conditional Random Field
    
    Returns
    -------
    object
        Bare model object

    Other parameters
    ----------------
    n_states : integer
        Number of classes (CRF only)
    n_features : integer
        Number of features (CRF only)

    '''

    if rlp_maps is not None and model_type == 'LR_RLP':
        n_features = n_features + len(rlp_maps.keys())

    if model_type == 'LR':
        return LogisticRegression()
    elif model_type == 'LR_RLP':
        return LogisticRegressionRLP(rlp_maps=rlp_maps, rlp_stats=rlp_stats)
    elif model_type == 'SVM':
        return SupportVectorMachine()
    elif model_type == 'CRF':
        crf = pystruct.models.GridCRF(n_states=n_states, n_features=n_features)
        return ConditionalRandomField(crf, verbose=1, max_iter=1000000)
    else:
        raise ValueError('Unknown model type [%s]' % model_type)

def train_models(models, train_sets, prior_sets=None, callback=None):
    '''Trains a set of model against a series of training sets

    Parameters
    ----------
    models : list
        List of model objects. Model objects should have a fit() method.
    train_sets : list
        List of tuples containing training data.
        The first item in a tuple is a 2D array. Each row is a training instance,
            while each column is a feature.
        The second item in a tuple is an array containing class annotations for each
            training instance.
    prior_sets: list
        List of 2D arrays containing prior data.
        Similar to first tuple item in train_sets
        Each item is a 2D array. Each row is a training instance,
            while each column is a feature.
    callback: function
        Callback function that is called after training of a model finished.
        Function accepts two parameters: the model object and a tuple with location
            indices in the resulting model matrix.

    Returns
    -------
    list
        List of lists with each item a trained instance of one of the models.

    '''

    n = len(models)
    m = len(train_sets)

    # create result matrix
    mtx = [[model for i in range(len(train_sets))] for model in models]

    # train each item in result matrix with corresponding training set
    for i, row in enumerate(mtx):
        for j, model in enumerate(row):
            X_train, Y_train = train_sets[j]
            if prior_sets is not None and type(model) == LogisticRegressionRLP:
                train_model(model, X_train, Y_train, prior_sets[j][0])
            else:
                train_model(model, X_train, Y_train)

            # callback after training
            if callback is not None:
                callback(model, (i,j))

    return mtx

def train_model(model, X_train, Y_train, X_train_prior=None):
    '''Trains a single model against a single training set

    Parameters
    ----------
    model : object
        Bare model object. Model object should hava a fit() method.
    X_train : list or numpy.ndarray
        2D array containing training data. Each row is a training instance,
            while each column is a feature.
    Y_train : list or numpy.ndarray
        Array containing class annotations for each training instance.

    Notes
    -----
    Models are passed by reference and trained without copying.

    '''

    try:
        model.fit(X_train, X_train_prior, Y_train)
    except TypeError:
        model.fit(X_train, Y_train)

def score_models(models, train_sets, test_sets, **kwargs):
    '''Compute train/test scores for a set of trained models

    Parameters
    ----------
    models : list
        List of lists with each item a trained instance of a model.
    train_sets : list
        List of tuples containing training data corresponding to the model list.
    test_sets : list
        List of tuples containing test data corresponding to the model list.

    Returns
    -------
    pandas.DataFrame
        MultiIndex DataFrame containing training and test scores.
        Indices "model" and "set" indicate the model and training set number used.
        Columns "train" and "test" contain the train and test scores respectively.

    Notes
    -----
    Models should be trained.
    Model and set lists should be of equal length.
    In case of N models and M training sets the models should be organized in a 
        N-length list of M-length lists.
    The train and test sets should both be M-length lists.

    Examples
    --------

    >>> models = [models.get_model(model_type='LR'),
                  models.get_model(model_type='CRF', n_states=5, n_features=10)]

    >>> models_trained = models.train_models(models, [(X_train, Y_train)])

    >>> scores = test.score_models(models, [(X_train, Y_train)], [(X_test, Y_test)])
    '''

    check_sets(train_sets, test_sets, models)

    # test for set dimensions
    m = __number_of_sets(train_sets, test_sets)
    n = len(models)

    # create a multiindex result dataframe
    ids = [[__modelname(models[i][j]),j] for i in range(n) for j in range(m)]
    ids = np.array(zip(*ids))
    ix = pandas.MultiIndex.from_arrays(ids, names=['model','set'])
    df = pandas.DataFrame(np.empty((n*m,2)), columns=['train','test'], index=ix)

    # score each combination of model and training set
    for i, model in enumerate(models):
        for j, ((X_train, Y_train), (X_test, Y_test)) in __enumerate_sets(i, train_sets, test_sets):
            idx = df.index.get_loc((__modelname(model[j]), str(j)))
            df['train'][idx], df['test'][idx] = score_model(model[j], X_train, Y_train, X_test, Y_test, **kwargs)

    return df

def score_model(model, X_train, Y_train, X_test, Y_test, maps=None, stats=None, features=None):
    '''Scores a single model using a train and test set

    Parameters
    ----------
    model : object
        Trained model object. Model object should have a score() method.
    X_train : list or numpy.ndarray
        2D array containing training data.
        Each row is a training instance, while each column is a feature.
    Y_train : list or numpy.ndarray
        Array containing class annotations for each training instance.
    X_test : Similar to X_train, but with test data.
    Y_test : Similar to Y_train, but with test data.

    Returns
    -------
    score_train : float
        Training score

    score_test : float
        Test score

    '''

    score_train = model.score(X_train, Y_train)
    score_test = model.score(X_test, Y_test)

    return score_train, score_test

def predict_models(models, sets):

    Y = []

    for i, row in enumerate(models):
        Y.append([])
        for j, model in enumerate(row):
            Y[i].append([])
            for X in sets[j]:
                Y[i][j].append(predict_model(model, X))
            
    return Y

def predict_model(model, X):

    Y = model.predict(X)

    return Y.reshape((X.shape[:-1]))

def __enumerate_sets(i, train_sets, test_sets):
    'Set generator'
    
    try:
        for record in enumerate(zip(train_sets, test_sets)):
            yield record
    except:
        for record in enumerate(zip(train_sets[i], test_sets[i])):
            yield record

def __number_of_sets(train_sets, test_sets):
    try:
        (X_train, Y_train), (X_test, Y_test) = zip(train_sets, test_sets)[0]
        m = len(train_sets)
    except:
        m = np.max([len(s) for s in train_sets])

    return m

def __modelname(model):
    return str(type(model)).replace("<class '","").replace("'>","")

