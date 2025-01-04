import json
import textstat
#This import is needed to load the pipeline in Spacy
import textdescriptives
import pandas as pd
from generate_labels import get_scores
import spacy
from taaled import ld
from argparse import ArgumentParser

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/all")

def create_dataset(file_path, model, train=True):
    #dataset fields
    proficiency_level=[]
    session_id=[]
    user_id=[]
    date=[]

    #features
    lexical_density=[]
    lexical_variation=[]
    num_noun_chunks=[]
    sentence_length_mean=[]
    sentence_length_std=[]
    num_sentences=[]
    dif_words=[]
    dependency_distance_mean=[]
    dependency_distance_std=[]

    # scores
    linguistic_range = []
    grammatical_accuracy = []
    AI_generated=[]
    n_turns = []

    #json file is ordered by session id by default
    f = json.load(open(file_path))
    df = pd.DataFrame.from_dict(f)
    df = df[df['is_spoken']==False]
    df = df.sort_values(['participant', 'session_id'], ascending=[True, True])
    for d in df.iterrows():
        n_turns.append(len(d[1]))
        txt = '.'.join(d[1]['user_response']).replace("\r\n","")
        doc = nlp(txt)
        if train:
            scores = get_scores(txt, model)
            # scores
            linguistic_range.append(scores['linguistic_range'])
            grammatical_accuracy.append(scores['grammatical_accuracy'])
            AI_generated.append(scores['AI_generated'])

        else:
            # scores
            linguistic_range.append(None)
            grammatical_accuracy.append(None)
            AI_generated.append(None)
        session_id.append(d[1]['session_id'])
        user_id.append(d[1]['participant'])
        proficiency_level.append(d[1]['proficiency_level'])
        date.append(d[1]['session_start'][:10])
        num_chunks = len(set([chunk for chunk in doc.noun_chunks]))
        words=[word.text for word in doc]
        content_words = [word.text for word in doc if word.pos_.startswith('VERB') or word.pos_.startswith('PROPN') or
                         word.pos_.startswith('NOUN') or word.pos_.startswith('ADJ') or word.pos_.startswith('ADV')]
        tokens = [word.text+'_'+word.pos_ for word in doc]

        #features
        lexical_density.append(len(content_words)/len(words))
        lexical_variation.append(ld.lexdiv(tokens).mattr)
        num_noun_chunks.append(num_chunks)
        sentence_length_mean.append(doc._.descriptive_stats['sentence_length_mean'])
        num_sentences.append(doc._.descriptive_stats['n_sentences'])
        sentence_length_std.append(doc._.descriptive_stats['sentence_length_std'])
        dif_words.append(textstat.difficult_words(txt))
        dependency_distance_mean.append(doc._.dependency_distance['dependency_distance_mean'])
        dependency_distance_std.append(doc._.dependency_distance['dependency_distance_std'])

    #create dataframe
    df = pd.DataFrame({
    'proficiency_level':proficiency_level,
    'session_id' : session_id,
    'user_id': user_id,
    'lexical_density' : lexical_density,
    'num_noun_chunks' : num_noun_chunks,
    'lexical_variation' : lexical_variation,
    'num_sentences' : num_sentences,
    'dif_words' : dif_words,
    'sentence_length_mean': sentence_length_mean,
    'sentence_length_std' : sentence_length_std,
    'dependency_distance_mean': dependency_distance_mean,
    'dependency_distance_std':dependency_distance_std,
    'linguistic_range': linguistic_range,
    'grammatical_accuracy': grammatical_accuracy
    })

    df.to_excel('dataset.xlsx')
    print("dataset generation completed")
    return None

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--file_path', type=str)
    args = parser.parse_args()
    create_dataset(args.file_path, args.model_name)