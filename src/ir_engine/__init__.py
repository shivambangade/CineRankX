"""Classical IR engine: TF-IDF + Trie autocomplete."""

from src.ir_engine.ir_engine import IREngine
from src.ir_engine.tfidf_vectorizer import TFIDFEngine
from src.ir_engine.trie import Trie

__all__ = ["IREngine", "TFIDFEngine", "Trie"]
