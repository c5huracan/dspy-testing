import os
import dspy
api_key = os.getenv('ANTHROPIC_API_KEY')
lm = dspy.LM("anthropic/claude-3-5-haiku-20241022", api_key=api_key)
dspy.configure(lm=lm)

'''
# basic example
class WeatherSignature(dspy.Signature):
    location: str = dspy.InputField()
    weather_info: str = dspy.OutputField(desc="Weather information for the location.")
# Define a custom module as a class
class WeatherModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # You can add submodules or tools here if needed
    def forward(self, location):
        # Example logic (replace with real tool or LM call)
        weather = f"The current weather in {location} is sunny."
        return dspy.Prediction(weather_info=weather)
# Instantiate and use the module
weather_module = WeatherModule()
result = weather_module(location="New York")
print(result.weather_info)
'''

# agentic, self improving
# 1. Define tool and agent
def get_weather(location: str) -> str:
    return f"The current weather in (location) is sunny."
weather_tool = dspy.Tool(get_weather)

class WeatherSignature(dspy.Signature):
    location: str = dspy.InputField()
    weather_info: str = dspy.OutputField()

agent = dspy.ReAct(signature=WeatherSignature, tools=[weather_tool])

# 2. Collect user interactions
examples = []
for location in ["New York", "London"]:
    result = agent(location=location)
    # Simulate user feedback (e.g., correct/incorrect)
    feedback = True  # Replace with real feedback
    # examples.append(dspy.Example(location=location, weather_info=result.weather_info, feedback=feedback))
    examples.append(
        dspy.Example(location=location, weather_info=result.weather_info, feedback=feedback).with_inputs("location")
    )

# 3. Define a metric
def metric(example, pred, trace=None):
    return example.weather_info == pred.weather_info  # Replace with real evaluation logic

# 4. Optimize agent
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimized_agent = optimizer.compile(agent, trainset=examples)

# 5. Use the improved agent
result = optimized_agent(location="Tokyo")
print(f"It is {result.weather_info} in {location}")
