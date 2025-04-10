from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from xgboost.sklearn import XGBClassifier
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
import warnings, random
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

def dataset_preparation(data_file, y_cols =['vocabulary_range']):
    data = pd.read_excel(data_file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data['user_id'], _ = pd.factorize(data['user_id'])
    y = data[y_cols].to_numpy()
    # For xgb it is required for labels to start at index 0
    le = LabelEncoder()
    #y = [le.fit_transform(y[:,0]),le.fit_transform(y[:,1])]
    y = [le.fit_transform(y[:, 0])]
    data.drop(['session_id'], axis=1, inplace=True)
    data.drop(['vocabulary_range', 'grammatical_accuracy'], axis=1, inplace=True)
    data = data.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data,  y[0], test_size=.15, random_state=13)
    return [x_train, x_test, y_train, y_test]


def unlabeled_data_sampling(data, n):
    sample = data.sample(n=n, replace=False)
    data = data.drop(sample.index).reset_index(drop=True)
    return sample.to_numpy(), data


def semi_supervised_ppi_train(data_label, unl_file, model_name, model_params, sample_size=50, alpha=0.1, w_t=0.2):
    x_train_label, x_test_label, y_train_label, y_test_label= data_label
    #Unlabeled data and sampling
    data_unl = pd.read_excel(unl_file)
    data_unl = data_unl.loc[:, ~data_unl.columns.str.contains('^Unnamed')]
    data_unl['user_id'], _ = pd.factorize(data_unl['user_id'])
    data_unl.drop(['session_id'], axis=1,
              inplace=True)
    x_unl, data_unl  = unlabeled_data_sampling(data_unl, sample_size)

    if model_name == 'rf':
        model = RandomForestClassifier(**model_params)

    elif model_name == 'lgb':
        model = lgb.LGBMClassifier(**model_params)

    elif model_name == 'gbm':
        model = GradientBoostingClassifier(**model_params)

    elif model_name == 'xgb':
        model = XGBClassifier(**model_params)
    #Save values during training for plotting
    ws=[]
    accs=[]
    coverage=[]
    clf = model.fit(x_train_label, y_train_label)
    preds_label = clf.predict(x_test_label)
    probs_label = clf.predict_proba(x_test_label)
    probs_label = np.max(probs_label, axis=1)
    probs_unl = clf.predict_proba(x_unl)
    probs_unl = np.max(probs_unl, axis=1)
    preds_unl = clf.predict(x_unl)
    #Estimate mean accuracy, CI and width
    corrects = (preds_label==y_test_label).astype(float)
    ppi_ci = ppi_mean_ci(corrects, probs_label, probs_unl, alpha=alpha)
    acc = ppi_mean_pointestimate(corrects, probs_label, probs_unl)
    accs.append(acc[0])
    width = abs(ppi_ci[0]-ppi_ci[1])[0]
    ws.append(width)
    #Calculate coverage
    coverage.append(1) if ppi_ci[0] <= accs[0] <= ppi_ci[1] else coverage.append(0)
    #add imputed data to train and predicted y
    x_imputation = np.concatenate([x_train_label, x_unl])
    y_imputation = np.concatenate([y_train_label, preds_unl])
    sample_s = sample_size
    # Add new unlabeled samples until condition is no longer met
    while width <= w_t:
        #Get new unlabeled sample
        x_unl, data_unl = unlabeled_data_sampling(data_unl, sample_size)
        #Continue with training with new data
        clf = model.fit(x_imputation, y_imputation)
        preds_label = clf.predict(x_test_label)
        probs_label = clf.predict_proba(x_test_label)
        probs_label = np.max(probs_label, axis=1)
        probs_unl = clf.predict_proba(x_unl)
        probs_unl = np.max(probs_unl, axis=1)
        preds_unl = clf.predict(x_unl)
        # Estimate accuracy, CI and width
        corrects = (preds_label == y_test_label).astype(float)
        ppi_ci = ppi_mean_ci(corrects, probs_label, probs_unl, alpha=alpha)
        acc = ppi_mean_pointestimate(corrects, probs_label, probs_unl)
        accs.append(acc[0])
        width = abs(ppi_ci[0]-ppi_ci[1])[0]
        ws.append(width[0])
        coverage.append(1) if ppi_ci[0] <= accs[0] <= ppi_ci[1] else coverage.append(0)
        # add imputed data to train and predicted y
        x_imputation = np.concatenate([x_imputation, x_unl])
        y_imputation = np.concatenate([y_imputation, preds_unl])
        sample_s += sample_size
        print(f"CI_lower={ppi_ci[0][0]:.3} CI_upper={ppi_ci[1][0]:.3} width={width:.3f} sample_size={sample_s:.3f}")
    else:
        joblib.dump(model, model_name+'.joblib')
        print("Trained model saved")
    return None


def train_test_clf(data_train, model_name):
    x_train, x_test, y_train, y_test = data_train
    #Parameter search for the selected model, since both response variables are highly correlated (>0.85)
    #we use one for param search.
    model_optim = find_parameters(model_name, x_train, y_train[:,0])
    clf = MultiOutputClassifier(model_optim).fit(x_train, y_train)
    preds = clf.predict(x_test)
    precision, recall, f1_scores, accuracy_scores, auc_scores= calculate_metrics(y_test, y_train, preds)

    print(f"Precision: {np.mean(precision)}")
    print(f"Recall: {np.mean(recall)}")
    print(f"F1: {np.mean(f1_scores)}")
    print(f"Accuracy: {np.mean(accuracy_scores)}")
    print(f"AUC: {np.mean(auc_scores)}")
    return None


def  calculate_metrics(y_test, y_train, preds):
    precision = []
    recall = []
    f1_scores = []
    auc_scores = []
    accuracy_scores = []
    for i in range(y_test.shape[1]):
        labels = list(set(y_train[:, i]))
        ytest = label_binarize(list(y_test[:, i]), classes=labels)
        ypreds = label_binarize(list(preds[:, i]), classes=labels)
        precision.append(precision_score(y_test[:, i], preds[:, i], average='weighted'))
        recall.append(recall_score(y_test[:, i], preds[:, i], average='weighted'))
        f1_scores.append(f1_score(y_test[:, i], preds[:, i], average='weighted'))
        accuracy_scores.append(accuracy_score(y_test[:, i], preds[:, i]))
        auc_scores.append(roc_auc_score(ytest, ypreds))

    return precision, recall, f1_scores, accuracy_scores, auc_scores


def find_parameters(model_name, X, y):
    if model_name =='rf':
        model = RandomForestClassifier(random_state=42)
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

    elif model_name == 'lgb':
        model = lgb.LGBMClassifier(num_leaves=35, learning_rate=0.01, n_estimators=20, verbose=-1)
        params = {"learning_rate": [0.2, 0.3, 0.4], "n_estimators": [50, 60, 100]}
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        print(best_params)
        model_optim = lgb.LGBMClassifier(num_leaves=35, learning_rate=best_params['learning_rate'],
                                         n_estimators=best_params['n_estimators'], verbose=-1)

    elif model_name =='gbm':
        model = GradientBoostingClassifier(loss='log_loss', verbose=0,
                                           random_state=42)
        params = {
                'n_estimators': [100, 200, 300, 500],
                'max_features': ['sqrt', 'log2'],
                'max_leaf_nodes': [2, 4, 6, 8]
            }
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        model_optim = GradientBoostingClassifier(loss= 'log_loss',
                                        n_estimators=best_params['n_estimators'],
                                          max_features=best_params['max_features'],
                                          max_leaf_nodes=best_params['max_leaf_nodes'],
                                                 random_state=42, verbose=0)

    elif model_name =='xgb':
        model = XGBClassifier(learning_rate =0.1, min_child_weight=1, num_class=4,
                              gamma=0, subsample=0.8, colsample_bytree=0.8,
                              objective= 'multi:softmax', nthread=4, seed=27,
                              random_state=42, verbose=-1)
        params = {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200, 500, 750]}
        grdsearch = GridSearchCV(estimator=model, param_grid=params)
        grdsearch.fit(X, y)
        best_params = grdsearch.best_params_
        model_optim = XGBClassifier(learning_rate = 0.1, max_depth = best_params['max_depth'], min_child_weight=1,num_class=4,
                                    random_state=42, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                    n_estimators = best_params['n_estimators'],
                              objective= 'multi:softmax', nthread=4,scale_pos_weight=1, seed=27, verbose=0)

    return model_optim

#Function to determine feature importance
def feature_selection(method, X, y, score=mutual_info_regression):
    if method == 'rf':
        model1 = RandomForestClassifier().fit(X, y['vocabulary_range'])
        model2 = RandomForestClassifier().fit(X, y['grammatical_accuracy'])
        for model in [model1, model2]:
            feat_importance = model.feature_importances_
            feature_importance = pd.Series(feat_importance, index=X.columns)
            print(feature_importance.sort_values(ascending=False))

    elif method == 'lgb':
        model1 = lgb.LGBMClassifier(num_leaves=35, learning_rate=0.01, n_estimators=20).fit(X, y['vocabulary_range'])
        model2 = lgb.LGBMClassifier(num_leaves=35, learning_rate=0.01, n_estimators=20).fit(X, y['grammatical_accuracy'])
        for model in [model1, model2]:
            feat_importance = model.feature_importances_
            feature_importance = pd.Series(feat_importance, index=X.columns)
            print(feature_importance.sort_values(ascending=False))

    #Model independent feature selection
    elif method == 'kbest':
        kbest = SelectKBest(score_func=score, k=len(X.columns)-4)
        k1 = kbest.fit(X, y['vocabulary_range'])
        print("Selected features for model k1:", X.columns[kbest.get_support()])
        k2 = kbest.fit(X, y['grammatical_accuracy'])
        print("Selected features for model k2:", X.columns[kbest.get_support()])
        return None

    elif method == 'mi':
        mi = mutual_info_classif(X, y['grammatical_accuracy'])
        mi = pd.Series(mi, name="MI Scores", index=X.columns)
        mi = mi.sort_values(ascending=False)
        print(mi)
        return None

    elif method == 'corr':
        X['obj1'] = y['vocabulary_range']
        X['obj2'] = y['grammatical_accuracy']
        corr = X.corr(method='spearman')
        corr_obj1 = corr['obj1']
        corr_obj2 = corr['obj2']
        corr_obj1=corr_obj1.drop('obj1', axis=0)
        corr_obj1=corr_obj1.drop('obj2', axis=0)
        corr_obj2 = corr_obj2.drop('obj1', axis=0)
        corr_obj2 = corr_obj2.drop('obj2', axis=0)
        k = len(X.columns) - 3
        topk1 = corr_obj1.abs().sort_values(ascending=False)[:k].index
        topk2 = corr_obj2.abs().sort_values(ascending=False)[:k].index
        print(topk1)
        print(topk2)
        return None

    else:
        print("select one of the following methods: rf, kbest, mi, and corr")