from langchain_community.llms import Ollama

# Initialize model
llm = Ollama(model="phi")

print("🤖 GenAI Evaluation Tool (Day 36)\n")

# Test prompts
test_cases = [
    "Explain AI in simple words",
    "What is the capital of India?",
    "Write a short poem on technology",
    "Who is the Prime Minister of India?"
]

# Evaluation function
def evaluate_response(prompt, response):
    score = 0

    # Basic checks
    if len(response) > 20:
        score += 1  # length
    if any(word in response.lower() for word in ["ai", "india", "technology", "prime"]):
        score += 1  # relevance
    if "." in response:
        score += 1  # coherence

    return score

# Run evaluation
for prompt in test_cases:
    print(f"\n🔹 Prompt: {prompt}")
    
    response = llm.invoke(prompt)
    print(f"Response: {response}")
    
    score = evaluate_response(prompt, response)
    print(f"Score (out of 3): {score}")