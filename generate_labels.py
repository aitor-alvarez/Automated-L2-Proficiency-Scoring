from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv,find_dotenv
import os

load_dotenv(find_dotenv())

#API key
api = os.environ.get("OPEN_AI_KEY")

#General template
prompt_template = ChatPromptTemplate.from_template(

"""
Score the following text from a conversation of an intermediate English language (B1-B2 on CEFR) student.

Provide only the integer associated with the option in the 'ScoringTexts' function.

Text:
{text}
"""
)

#base model to define the scores for each measure
class ScoringTexts(BaseModel):
  #General linguistic range score combining CEFR linguistic range and vocabulary range.
  linguistic_range: int = Field(description="Select the option that best describes the text."
                                           "Option 1. Has enough language to get by, with sufficient vocabulary to express themselves with some hesitation."
                                            "Lexical limitations cause repetition and even difficulty with formulation at times. "
                                            "Has sufficient vocabulary to express themselves with some circumlocutions on most topics "
                                              "pertinent to their everyday life."
                                            "Option 2. Has a sufficient range of language to describe unpredictable situations, explain the main points "
                                            "in an idea"
                                            "or problem with reasonable precision and express thoughts on abstract or cultural topicsHas a good range "
                                            "of vocabulary "
                                            "related to familiar topics and everyday situations."
                                            "Option 3. Has a sufficient range of language to be able to give clear descriptions, express viewpoints and develop"
                                            "arguments without much conspicuous searching for words/signs, using some complex sentence forms to do so."
                                            "Can understand and use much of the specialist vocabulary of their field but has problems with "
                                              "specialist terminology outside it."
                                            "option 4. Can express themselves clearly without much sign of having to restrict what they want to say."
                                            "Can understand and use the main technical terminology of their field, when discussing their area of "
                                              "specialisation with other specialists.",
                              enum=[1,2, 3, 4])


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
                                                "but they are rare and can often be corrected in retrospect.",
                                          enum=[1, 2, 3, 4])


#Scoring function returns dict of measures
def get_scores(txt, model):
    llm = ChatOpenAI(temperature=0, model=model, max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api).with_structured_output(ScoringTexts)
    chain = prompt_template | llm
    scores = chain.invoke({"text": txt})
    return scores.dict()