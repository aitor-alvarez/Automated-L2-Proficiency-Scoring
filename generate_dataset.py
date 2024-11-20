import json
import pandas as pd
from generate_labels import get_scores

def create_dataset(file_path):
    #dataset fields
    proficiency_level=[]
    session_id=[]
    user_id=[]
    date=[]

    #scores
    linguistic_range=[]
    grammatical_accuracy=[]

    #features
    lexical_density=[]
    lexical_sophistication=[]
    lexical_variation=[]
    mean_sentence_length=[]

    #json file is ordered by session id by default
    f = json.load(open(file_path))
    df = pd.DataFrame.from_dict(f).sort_values(['participant', 'session_id'], ascending=[True, True])
    for d in df.iterrows():
        txt = '. '.join(d['user_response']).replace("\r\n","")
        scores = get_scores(txt)
        session_id.append(d['session_id'])
        user_id.append(d['participant'])
        proficiency_level.append(d['proficiency_level'])
        date.append(d['session_start'][:10])
        #features
        lexical_density.append()
        lexical_sophistication.append()
        lexical_variation.append()
        mean_sentence_length.append()
        #scores
        linguistic_range.append(scores['linguistic_range'])
        grammatical_accuracy.append(scores['grammatical_accuracy'])

    #create dataframe
    df = pd.DataFrame({
    'proficiency_level':proficiency_level,
    'session_id' : session_id,
    'user_id': user_id,
    'lexical_density' : lexical_density,
    'lexical_sophistication' :lexical_sophistication,
    'lexical_variation' : lexical_variation,
    'mean_sentence_length' : mean_sentence_length,
    'linguistic_range': linguistic_range,
    'grammatical_accuracy': grammatical_accuracy

    })

    df.to_excel('dataset.xlsx')
    print("dataset completed")
    return None