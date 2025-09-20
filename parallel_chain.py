from langchain_openai import ChatOpenAI
from langchain_anthropic  import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1=ChatOpenAI()

model2=ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes on \n{text}',
    input_variables=['text']
)

prompt2= PromptTemplate(
    template='Generate 5 short questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into single documnet \n notes-> {notes} and quiz ->{quiz}',

    input_variables=['notes','quiz']
)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1 |  parser,
    'quiz':prompt2 | model2 |parser
})

merge_chain=prompt3 |model1|parser

chain=parallel_chain | merge_chain

text=""" SVM (Support Vector Machine) is a powerful supervised machine learning algorithm 
used for both classification and regression tasks, which finds the optimal hyperplane (decision boundary) that best separates data points into different categories.
 It achieves this by maximizing the margin between the hyperplane and the nearest data points, called support vectors.
  SVMs are particularly effective in high-dimensional spaces and can handle non-linear data through the use of kernel functions, making them versatile tools for pattern recognition, text analysis, and more
 """

result=chain.invoke({'text':text})
print(result )

chain.get_graph().print_ascii()