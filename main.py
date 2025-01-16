from model import *
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = ArgumentParser()
    #model options are: 'rf', 'gbm', 'xgb'
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--feature_set', type=str)
    args = parser.parse_args()
    if args.dataset_file and args.model_name:
        dataset = dataset_preparation(args.dataset_file)
        train_test_clf(dataset, args.model_name)
    #Feature selection
    elif args.feature_set:
        data = pd.read_excel(args.feature_set)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data['user_id'], _ = pd.factorize(data['user_id'])
        y = data[['vocabulary_range', 'grammatical_accuracy']]
        data.drop(['date', 'vocabulary_range', 'grammatical_accuracy'], axis=1, inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=.2)
        #Options: kbest, rf, lgb, corr
        feature_selection('kbest', x_train, y_train )
    else:
        print("Provide the correct parameters")