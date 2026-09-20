import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class SciBERTEmbedder:
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def _chunks(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        payload = self.max_length - 2
        return [ids[i:i+payload] for i in range(0, len(ids), payload)] or [[]]

    @torch.no_grad()
    def _embed_ids(self, ids):
        encoded = self.tokenizer.prepare_for_model(
            ids, add_special_tokens=True, max_length=self.max_length,
            truncation=True, return_tensors="pt"
        )
        encoded = {k:v.to(self.device) for k,v in encoded.items()}
        out = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return emb.squeeze(0).cpu().numpy()

    def embed_document(self, text):
        seg = [self._embed_ids(x) for x in self._chunks(text)]
        return np.mean(np.vstack(seg), axis=0)

    def encode(self, texts):
        return np.vstack([self.embed_document(t) for t in tqdm(texts, desc="SciBERT")])
