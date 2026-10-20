import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = len(self.model.encode(["x"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def split_text(self, text: str, size: int, overlap: int) -> list[str]:
        """按这个模型的 tokenizer 切块：每块 size 个 token，相邻两块重叠 overlap 个，最后一块到头就停。
        走法同 LangChain 的 split_text_on_tokens：先 tokenize，按 token 切，每块再 decode 回文本"""
        ids = self.model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        out = []
        for i in range(0, len(ids), size - overlap):
            out.append(self.model.tokenizer.decode(ids[i:i + size]))
            if i + size >= len(ids):
                break
        return out


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """query 和每段文本配对打分，分高的更相关"""
        return np.asarray(self.model.predict([(query, t) for t in texts]), dtype=np.float32)
