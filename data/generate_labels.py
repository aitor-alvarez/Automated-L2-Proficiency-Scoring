from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

prompt_template = ChatPromptTemplate.from_template(
 """
Score the following text from a conversation.

Provide only the measures in the 'ScoringTexts' function.

Text:
{input}
"""
)

#base model to define the scores for each dimension
class ScoringTexts(BaseModel):
  score_accuracy: int = Field(description="describes how accurate the text is, the higher number the more accurate. ",
                              enum=[1,2, 3, 4])
  score_complexity : int = Field(description="describes how complex the text is, the higher number the more accurate. ",
                              enum=[1,2, 3, 4])