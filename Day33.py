from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize LLM (phi model - lightweight)
llm = Ollama(model="phi")

# Initialize Memory
memory = ConversationBufferMemory()

# Create Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("🤖 Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("👋 Chat ended!")
        break
    
    response = conversation.run(user_input)
    print("Bot:", response)