from model import *
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    #model options are: 'rf', 'gbm', 'xgb'
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_file', type=str)
    args = parser.parse_args()
    if args.dataset_file and args.model_name:
        dataset = dataset_preparation(args.dataset_file)
        train_test_clf(dataset, args.model_name)
    else:
        print("Provide the correct parameters")