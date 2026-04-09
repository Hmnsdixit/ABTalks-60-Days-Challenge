from langchain_community.llms import Ollama

# Initialize LLM
llm = Ollama(model="phi")

# 🔧 Tool functions
def calculator(query):
    try:
        return str(eval(query))
    except:
        return "Invalid calculation"

def word_counter(text):
    return f"Word count: {len(text.split())}"

print("🤖 Tool-based Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Exiting...")
        break

    # Simple tool detection
    if any(op in user_input for op in ["+", "-", "*", "/"]):
        result = calculator(user_input)
        print("Bot (Calculator):", result)

    elif "count words" in user_input.lower():
        text = user_input.replace("count words in", "")
        result = word_counter(text)
        print("Bot (WordCounter):", result)

    else:
        response = llm.invoke(user_input)
        print("Bot:", response)