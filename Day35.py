from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Initialize LLM
llm = Ollama(model="phi")

# Memory
memory = ConversationBufferMemory()

print("🤖 AI Chatbot with Memory started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Chat ended!")
        break

    # Add user input to memory
    memory.chat_memory.add_user_message(user_input)

    # Prepare conversation history
    history = memory.load_memory_variables({})["history"]

    # Prompt with memory
    prompt = f"""
You are a helpful AI chatbot.

Conversation history:
{history}

User: {user_input}
AI:
"""

    # Get response
    response = llm.invoke(prompt)

    # Add AI response to memory
    memory.chat_memory.add_ai_message(response)

    print("Bot:", response)