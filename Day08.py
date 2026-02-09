# Day 08: Train a WordPiece Tokenizer & Analyze Vocabulary Size

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# -----------------------------
# Step 1: Prepare a larger corpus
# -----------------------------
corpus = [
    "Tokenization is a key step in natural language processing",
    "WordPiece tokenizer is widely used in transformer models",
    "Large language models rely on subword tokenization",
    "Tokenization helps handle rare and unknown words effectively",
    "Python is commonly used for NLP and data science tasks",
    "Modern NLP systems use WordPiece or BPE tokenizers"
]

# Save corpus to file (WordPiece needs file input)
with open("corpus.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        f.write(line.lower() + "\n")

# -----------------------------
# Step 2: Initialize tokenizer
# -----------------------------
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# -----------------------------
# Step 3: Train WordPiece tokenizer
# -----------------------------
trainer = WordPieceTrainer(
    vocab_size=200,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

tokenizer.train(["corpus.txt"], trainer)

# -----------------------------
# Step 4: Analyze vocabulary size
# -----------------------------
vocab = tokenizer.get_vocab()
print("Vocabulary Size:", len(vocab))

# -----------------------------
# Step 5: Test tokenization
# -----------------------------
sentence = "Tokenization in language models is powerful"
encoding = tokenizer.encode(sentence.lower())

print("\nOriginal Sentence:")
print(sentence)

print("\nTokens:")
print(encoding.tokens)

print("\nToken IDs:")
print(encoding.ids)
