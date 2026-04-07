# Day 31 - LangChain Basic Pipeline (Prompt → LLM → Output)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Step 2: Create Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful AI assistant.\n\nUser Input: {input}\n\nResponse:"
)

# Step 3: Create Chain
chain = prompt_template | llm

# Step 4: User Input
user_input = input("Enter your prompt: ")

# Step 5: Run Chain
response = chain.invoke({"input": user_input})

# Step 6: Output
print("\n🤖 AI Response:\n")
print(response.content)