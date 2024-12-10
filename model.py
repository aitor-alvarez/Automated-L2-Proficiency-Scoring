from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd

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
    model_optim.fit(x_train, y_train)
    preds = model_optim.predict(x_test)
    return None

def find_parameters(model_name, X, y):
    if model_name =='rf':
        model = RandomForestClassifier()
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

        if model_name =='gbm':
            model = GradientBoostingClassifier()
            params = {
                'loss' : ['log_loss', 'exponential'],
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