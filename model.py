from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np

def dataset_preparation(data_file):
    data = pd.read_excel(data_file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data['user_id'], _ = pd.factorize(data['user_id'])
    y = data[['linguistic_range', 'grammatical_accuracy']]
    data.drop(['linguistic_range', 'grammatical_accuracy'], axis=1, inplace=True)
    #We keep the original data for feature selection (column names)
    data2 = data.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data2,  y, test_size=.2)
    return x_train, x_test, y_train, y_test

def train_test_clf(data_file, model_name):
    x_train, x_test, y_train, y_test = dataset_preparation(data_file)
    #Parameter search for each model
    model_optim = find_parameters(model_name, x_train, y_train)
    clf = MultiOutputClassifier(model_optim).fit(x_train, y_train)
    preds = clf.predict(x_test)
    # iterate over the y dimension
    f1_scores=[]
    auc_scores=[]
    labels = list(set(y_test))
    ytest = label_binarize(y_test, classes=labels)
    ypreds = label_binarize(preds, classes=labels)
    for i in range(ytest.shape[1]):
        f1_scores.append(f1_score(y_test[:, i], preds[:,i]))
        auc_scores.append(roc_auc_score(y_test[:, i], preds[:,i]))

    print(f"F1: {np.mean(f1_scores)}")
    print(f"AUC: {np.mean(auc_scores)}")
    return None

def find_parameters(model_name, X, y):
    if model_name =='rf':
        model = RandomForestClassifier()
        params = {
            'n_estimators': [100, 200, 300, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8]
        }
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        model_optim = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                          max_features=best_params['max_features'],
                                          max_depth=best_params['max_depth'])

    elif model_name =='gbm':
        model = GradientBoostingClassifier()
        params = {
                'loss': ['log_loss', 'exponential'],
                'n_estimators': [100, 200, 300, 500],
                'max_features': ['sqrt', 'log2'],
                'max_leaf_nodes': [2, 4, 6, 8]
            }
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        model_optim = GradientBoostingClassifier(loss= best_params['loss'],
                                        n_estimators=best_params['n_estimators'],
                                          max_features=best_params['max_features'],
                                          max_leaf_nodes=best_params['max_leaf_nodes'])

    return model_optim

#Function to determine feature importance
def feature_selection(method, X, y, score):
    if method == 'rf':
        model1 = RandomForestClassifier().fit(X, y['linguistic_range'])
        model2 = RandomForestClassifier().fit(X, y['grammatical_accuracy'])
        for model in [model1, model2]:
            feat_importance = model.feature_importances_
            feature_importance = pd.Series(feat_importance, index=X.columns)
            print(feature_importance.sort_values(ascending=False))

    #Model independent feature selection
    elif method == 'kbest':
        kbest = SelectKBest(score_func=score, k=len(X.columns)-3)
        k1 = kbest.fit_transform(X, y['linguistic_range'])
        print("Selected features for model k1:", X.columns[kbest.get_support()])
        k2 = kbest.fit_transform(X, y['grammatical_accuracy'])
        print("Selected features for model k2:", X.columns[kbest.get_support()])
    elif method == 'corr':
        X['obj1'] = y['linguistic_range']
        X['obj2'] = y['grammatical_accuracy']
        corr = X.corr()
        corr_obj1 = corr['obj1']
        corr_obj2 = corr['obj2']
        corr_obj1=corr_obj1.drop('obj1', axis=0)
        corr_obj1=corr_obj1.drop('obj2', axis=0)
        corr_obj2 = corr_obj2.drop('obj1', axis=0)
        corr_obj2 = corr_obj2.drop('obj2', axis=0)
        k = len(X.columns) - 3
        topk1 = corr_obj1.abs().sort_values(ascending=False)[:k].index
        topk2 = corr_obj2.abs().sort_values(ascending=False)[:k].index

    else:
        print("select one of the following methods: rf, kbest, and corr")