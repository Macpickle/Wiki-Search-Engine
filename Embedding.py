from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path

out = Path("onnx_search_model")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save_onnx(out, opset=14, optimization_level=99)

tok = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")
tok.save_pretrained(out / "tokenizer")