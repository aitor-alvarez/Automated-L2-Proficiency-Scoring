import json
import textstat
import pandas as pd
from generate_labels import get_scores
import spacy
from taaled import ld

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/all")

def create_dataset(file_path):
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

    #json file is ordered by session id by default
    f = json.load(open(file_path))
    df = pd.DataFrame.from_dict(f).sort_values(['participant', 'session_id'], ascending=[True, True])
    for d in df.iterrows():
        txt = '. '.join(d['user_response']).replace("\r\n","")
        doc = nlp(txt)
        scores = get_scores(txt)
        session_id.append(d['session_id'])
        user_id.append(d['participant'])
        proficiency_level.append(d['proficiency_level'])
        date.append(d['session_start'][:10])
        num_chunks = len(set([chunk for chunk in doc.noun_chunks]))
        words=[word.text for word in doc]
        content_words = [word.text for word in doc if word.pos_.startswith('VERB') or word.startswith('PROPN') or
                         word.startswith('NOUN') or word.startswith('ADJ') or word.startswith('ADV')]
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

        #scores
        linguistic_range.append(scores['linguistic_range'])
        grammatical_accuracy.append(scores['grammatical_accuracy'])

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