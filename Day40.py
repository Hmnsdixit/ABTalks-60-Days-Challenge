from langchain_community.embeddings import OllamaEmbeddings
from pymongo import MongoClient

# 🔹 Step 1: Connect MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["ai_db"]
collection = db["documents"]

# 🔹 Step 2: Load embeddings model
embeddings = OllamaEmbeddings(model="phi")

# 🔹 Step 3: Sample documents
documents = [
    "Artificial Intelligence is the future",
    "Machine Learning is a subset of AI",
    "Deep Learning is powerful",
    "I love playing cricket",
    "Data Science is amazing"
]

# 🔹 Step 4: Store documents with embeddings
collection.delete_many({})  # clear old data

for doc in documents:
    vector = embeddings.embed_query(doc)

    collection.insert_one({
        "text": doc,
        "embedding": vector
    })

print("✅ Documents stored in MongoDB with embeddings!")

# 🔹 Step 5: Search function
def search(query):
    query_vector = embeddings.embed_query(query)

    results = []
    for doc in collection.find():
        # cosine similarity manually
        dot = sum(a*b for a, b in zip(query_vector, doc["embedding"]))
        norm_a = sum(a*a for a in query_vector) ** 0.5
        norm_b = sum(b*b for b in doc["embedding"]) ** 0.5

        similarity = dot / (norm_a * norm_b)

        results.append((doc["text"], similarity))

    # sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:3]

# 🔹 Step 6: Query loop
print("\n🤖 MongoDB Semantic Search Ready! Type 'exit' to quit.\n")

while True:
    query = input("🔍 Enter your query: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    results = search(query)

    print("\n📊 Top Results:\n")
    for i, (text, score) in enumerate(results):
        print(f"{i+1}. {text} (Score: {round(score, 2)})")

    print("-" * 40)