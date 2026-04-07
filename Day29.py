# Day 29: OpenAI API Chatbot (LLM Basics)

from openai import OpenAI

# API Key set karo (IMPORTANT)
client = OpenAI(api_key="YOUR_API_KEY_HERE")

print("🤖 AI Chatbot Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    # API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_input}
        ]
    )

    bot_reply = response.choices[0].message.content
    print("Bot:", bot_reply)