from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np

def dataset_preparation(data_file):
    data = pd.read_excel(data_file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data['user_id'], _ = pd.factorize(data['user_id'])
    lr_index = data.columns.get_loc('linguistic_range')
    ga_index = data.columns.get_loc('grammatical_accuracy')
    y = data.iloc[:, [lr_index, ga_index]].to_numpy()
    data = data.drop(['linguistic_range', 'grammatical_accuracy'], axis=1, inplace=True)
    data = data.to_nunmpy()
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=.2)
    return [x_train, x_test, y_train, y_test]

def train_test_clf(data, model_name):
    x_train, x_test, y_train, y_test = data
    #Parameter search for each model
    model_optim = find_parameters(model_name, x_train, y_train)
    clf = MultiOutputClassifier(model_optim).fit(x_train, y_train)
    preds = clf.predict(x_test)
    # iterate over the y dimension
    f1_scores=[]
    auc_scores=[]
    for i in range(y_test.shape[1]):
        f1_scores.append(f1_score(y_test[:, i], preds[:,i]))
        auc_scores.append(roc_auc_score(y_test[:, i], preds[:,i]))

    print(f"F1: {np.mean(f1_scores)}")
    print(f"AUC: {np.mean(auc_scores)}")
    return None

def find_parameters(model_name, X, y):
    if model_name =='rf':
        model = MultiOutputClassifier(RandomForestClassifier())
        params = {
            'n_estimators': [100, 200, 300, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['friedman_mse']
        }
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        model_optim = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                          max_features=best_params['max_features'],
                                          max_depth=best_params['max_depth'], criterion=best_params['criterion'])

    elif model_name =='gbm':
        model = MultiOutputClassifier(GradientBoostingClassifier())
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