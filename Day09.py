# Day 09: Compare Tokenization Methods
# Whitespace vs BPE vs WordPiece

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# --------------------------------------------------
# Sample input text (same for all tokenizers)
# --------------------------------------------------
text = "Tokenization in language models is powerful and efficient"

print("\nOriginal Text:")
print(text)

# --------------------------------------------------
# 1. Whitespace Tokenization
# --------------------------------------------------
print("\n--- Whitespace Tokenization ---")
whitespace_tokens = text.lower().split()
print("Tokens:", whitespace_tokens)
print("Total Tokens:", len(whitespace_tokens))

# --------------------------------------------------
# Prepare corpus file for BPE & WordPiece
# --------------------------------------------------
with open("day09_corpus.txt", "w", encoding="utf-8") as f:
    f.write(text.lower())

# --------------------------------------------------
# 2. BPE Tokenization
# --------------------------------------------------
print("\n--- BPE Tokenization ---")

bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = Whitespace()

bpe_trainer = BpeTrainer(
    vocab_size=50,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

bpe_tokenizer.train(["day09_corpus.txt"], bpe_trainer)

bpe_output = bpe_tokenizer.encode(text.lower())
print("Tokens:", bpe_output.tokens)
print("Total Tokens:", len(bpe_output.tokens))

# --------------------------------------------------
# 3. WordPiece Tokenization
# --------------------------------------------------
print("\n--- WordPiece Tokenization ---")

wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_tokenizer.pre_tokenizer = Whitespace()

wp_trainer = WordPieceTrainer(
    vocab_size=50,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

wp_tokenizer.train(["day09_corpus.txt"], wp_trainer)

wp_output = wp_tokenizer.encode(text.lower())
print("Tokens:", wp_output.tokens)
print("Total Tokens:", len(wp_output.tokens))

# --------------------------------------------------
# Observations
# --------------------------------------------------
print("\n--- Observations ---")
print("Whitespace: Simple but cannot handle rare/unknown words")
print("BPE: Breaks words into subwords based on frequency")
print("WordPiece: Uses likelihood-based subword splitting (used in BERT)")
