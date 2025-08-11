from model import *
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import json


if __name__ == '__main__':
    parser = ArgumentParser()
    #model options are:  'lgbm', 'xgb'
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_file', type=str)
    #Unlabeled dataset
    parser.add_argument('--dataset_unl', type=str)
    #training= 'weak' or 'semi'
    parser.add_argument('--training', type=str)
    #Model parameters
    parser.add_argument('--model_params', type=json.loads, default={})
    args = parser.parse_args()
    if args.dataset_file and args.model_name and args.training:
        #labeled dataset
        dataset = dataset_preparation(args.dataset_file)
        unlabeled_data =args.dataset_unl
        if args.training =='semi':
            semi_supervised_ppi_train(dataset, unlabeled_data, args.model_name, args.model_params, sample_size=100, alpha=0.1,
                                      w_t=0.2, max_sample_size=1100)
        elif args.training == 'weak':
            weakly_supervised_ppi_train(dataset, unlabeled_data, args.model_name, args.model_params, sample_size=100, alpha=0.1,
                                        w_t=0.2, max_sample_size=1100)
        else:
            print("revise the arguments required to run the model")

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