from langchain_openai import ChatOpenAI
from langchain_anthropic  import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch ,RunnableLambda
from langchain_core.output parsers import PydanticOutputParser
from pydantic import BaseModel ,Field
from typin import Literal

load_dotenv()

model1=ChatOpenAI()
parser=StrOutputParser()
class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] =Field(description='gibve the sentiment of the feedback')
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment into the following feedback test into positive or negative \n{feedback} \n {format_instruction}',
    input_variables=['feedback']
    partial_varibales={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 |model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to the positive feeback \n {feedback}',
    input_variables=['feedback']
    
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to the negative feeback \n {feedback}',
    input_variables=['feedback']
)
# result=classifier_chain.invoke({'feedback':'This is a terrible smart phone '}).sentiment
# print(result)

branch_chain=RunnableBranch(
    #when it is true which chain i want to execute
    (lambda x:x['sentiment']=='positive',prompt2|model|parser),
    (lambda x:x['sentiment']=='negative',prompt3|model|parser), 
    RunnableLambda(lambda x:"could not find any sentiment")
)
chain=classifier_chain |branch_chain
print(chain.invoke({'feeback':'This is a terrible Phone'}))


chain.get_graph().print_ascii()