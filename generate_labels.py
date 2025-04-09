from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


load_dotenv('keys.env')

#API key
api = os.environ.get("OPENAI_API_KEY")

#General template
prompt_template = ChatPromptTemplate.from_template(

"""
Score the following text from a conversation of an intermediate English language student (B1-B2 on CEFR).

Provide the score as an integer and the probability as a float associated with the options in the 'ScoringTexts' function.

Text:
{text}
"""
)

#base model to define the scores for each measure
class ScoringTexts(BaseModel):
  #CEFR vocabulary range.
  vocabulary_range: int = Field(description="Select the option that best describes the text."
                                           "Option 1. Has a good range of vocabulary related to familiar topics and everyday situations."
                                            "Has sufficient vocabulary to express themselves with some circumlocutions on most topics "
                                              "pertinent to their everyday life such as family, hobbies and interests, work, travel and current events."
                                            "Option 2. Has a good range of vocabulary for matters connected to their field and most general topics."
                                            "Can vary formulation to avoid frequent repetition, but lexical gaps can still cause hesitation"
                                            " and circumlocution."
                                            "Can produce appropriate collocations of many words/signs in most contexts fairly systematically."
                                            "Can understand and use much of the specialist vocabulary of their field but has problems with "
                                            "specialist terminology outside it."
                                            "Option 3. Can understand and use technical terminology when discussing "
                                            "areas of specialization. Have access to specialized vocabulary in relation to the topic.")

  vocabulary_range_proba: float = Field(description="Express in the form of a probability the confidence on the vocabulary range score given.")


  #measures of grammatical accuracy as per CEFR
  grammatical_accuracy: int = Field(description="Select the option that best describes the text."
                                              "Option 1. Uses reasonably accurately a repertoire of frequently used “routines” and patterns "
                                              "associated with more predictable situations. "
                                              "Option 2. Communicates with reasonable accuracy in familiar contexts; generally good control, "
                                                "though with noticeable mother-tongue influence."
                                              "Errors occur, but it is clear what they are trying to express."
                                              "Option 3. Has a good command of simple language structures and some complex grammatical forms, "
                                                "although they tend to use complex structures rigidly with some inaccuracy."
                                              "option 4. Good grammatical control; occasional “slips” or non-systematic errors and minor flaws "
                                                "in sentence structure may still occur, "
                                                "but they are rare and can often be corrected in retrospect.")

  grammatical_accuracy_proba: float = Field(description="Express in the form of a probability the confidence on the grammatical accuracy score given.")



#Scoring function returns dict of measures
#Models used gpt-4o-mini gpt-4o
def get_scores(txt, model="gpt-4o-mini"):
    llm = ChatOpenAI(temperature=0, model=model, api_key=api)
    llm_structured = llm.bind_tools([ScoringTexts])
    llm_structured = llm_structured.with_structured_output(ScoringTexts)
    chain = prompt_template | llm_structured
    scores = chain.invoke({"text": txt})
    return scores.dict()