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
If you use this code for your own research project, please cite the following work

```
@inproceedings{arronte-alvarez-xie-fincham-2025-automated,
    title = "Automated {L}2 Proficiency Scoring: Weak Supervision, Large Language Models, and Statistical Guarantees",
    author = "Arronte Alvarez, Aitor  and
      Xie Fincham, Naiyi",
    editor = {Kochmar, Ekaterina  and
      Alhafni, Bashar  and
      Bexte, Marie  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Tack, Ana{\"i}s  and
      Yaneva, Victoria  and
      Yuan, Zheng},
    booktitle = "Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bea-1.30/",
    doi = "10.18653/v1/2025.bea-1.30",
    pages = "384--397",
    ISBN = "979-8-89176-270-1"
}
```
