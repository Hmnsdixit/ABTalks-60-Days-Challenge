# Day 30: Prompt Engineering (Summarization + Q&A)

from openai import OpenAI

# 🔑 API Key yaha daalo
client = OpenAI(api_key="YOUR_API_KEY_HERE")

print("🤖 Prompt Engineering Tool Ready (type 'exit' to quit)\n")

while True:
    print("\nChoose Task:")
    print("1. Summarization")
    print("2. Question Answering")

    choice = input("Enter 1 or 2: ")

    if choice.lower() == "exit":
        print("Goodbye 👋")
        break

    text = input("\nEnter your text:\n")

    if choice == "1":
        # 🔥 Summarization Prompt
        prompt = f"""
        You are an expert summarizer.
        Summarize the following text in 3 concise bullet points:

        Text:
        {text}
        """

    elif choice == "2":
        question = input("\nEnter your question: ")

        # 🔥 Q&A Prompt
        prompt = f"""
        You are a helpful AI assistant.
        Answer the question based only on the given text.

        Text:
        {text}

        Question:
        {question}
        """

    else:
        print("Invalid choice ❌")
        continue

    # API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content

    print("\n📌 Result:\n")
    print(result)