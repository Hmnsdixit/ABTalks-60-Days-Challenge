from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Initialize model
llm = Ollama(model="phi")

text = """
Artificial Intelligence is transforming industries by automating tasks,
improving decision making, and enhancing user experiences.
"""

docs = [Document(page_content=text)]

chain = load_summarize_chain(llm, chain_type="stuff")

summary = chain.run(docs)

print("\n✅ SUMMARY:\n")
print(summary)