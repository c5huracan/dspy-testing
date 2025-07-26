import os
import dspy
api_key = os.getenv('ANTHROPIC_API_KEY')
lm = dspy.LM("anthropic/claude-3-5-haiku-20241022", api_key=api_key)
dspy.configure(lm=lm)

# simple example
'''
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

print(result.reasoning)
print(result.answer)
'''

# basic agentic example with multiple tool use
def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

result = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(result.reasoning)
print(result.answer)
print(f"Actual division:", 9362158/1659)