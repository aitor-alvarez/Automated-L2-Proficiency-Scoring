from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

prompt_template = ChatPromptTemplate.from_template(
 """
Score the following text from a conversation of an intermediate English language (B1-B2 on CEFR) student that is supposed to have 
the following lexical and grammatical competence:

-Has a good range of vocabulary for matters connected to their field and most general topics.
-Can produce appropriate collocations of many words/signs in most contexts fairly systematically.
-Good grammatical control; occasional slips or non-systematic errors and minor flaws in sentence structure may still occur.
-Does not make mistakes which lead to misunderstanding.
-Has a good command of simple language structures and some complex grammatical forms.

Provide only the measures in the 'ScoringTexts' function.

Text:
{text}
"""
)

#base model to define the scores for each measure
class ScoringTexts(BaseModel):
  #measures of lexical complexity
  lexical_density: int = Field(description="lexical density describes the ratio of the number of lexical words to the total number of words in a text."
                                           "1 indicates very low. 2 low. 3 medium. 4 high. 5 very high.",
                              enum=[1,2, 3, 4, 5])
  lexical_sophistication : int = Field(description="lexical sophistication describes a measure of the proportion of relatively unusual or advanced words in the learner’s text."
                                       "1 indicates very low. 2 low. 3 medium. 4 high. 5 very high.",
                              enum=[1,2, 3, 4, 5])
  lexical_variation: int = Field(
      description="lexical variation describes the number of different words in the text."
      "1 indicates very low. 2 low. 3 medium. 4 high. 5 very high.",
      enum=[1, 2, 3, 4, 5])

  #measures of morphosyntactic accuracy
  morphosyntactic_accuracy: int = Field(
      description="morphosyntactic accuracy describes errors in meaning and vocabulary and sentence structure."
      "1 indicates very low. 2 low. 3 medium. 4 high. 5 very high.",
      enum=[1, 2, 3, 4, 5])

#Scoring function returns dict of measures
def get_scores(txt, model):
    llm = ChatOpenAI(temperature=0, model=model).with_structured_output(ScoringTexts)
    chain = prompt_template | llm
    scores = chain.invoke({"text": txt})
    return scores.dict()