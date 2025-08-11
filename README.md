## Repository for Automated L2 Proficiency Scoring: Weak Supervision, Large Language Models, and Statistical Guarantees

### Install requirements

```
pip install -r requirements.txt
```

### Obtain scores from GPT models

```
python generate_dataset.py --model_name 'gpt-4o' --dataset_file' 'json_file_with conversational_responses'
```

### Train models and obtain C.I. with PPI in a semisupervised regime

```
python main.py --model_name 'xgb' --dataset_file' 'path to excel or csv file with dataset generated in the previous step' \
--dataset_unl 'path to csv or excel with features but no labels' \
--training 'semi' \
--model_params  '{"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100}'

```
