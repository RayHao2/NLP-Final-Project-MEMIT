import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGERetriever:
    def __init__(self, model_name= MODEL_NAME, device = None, max_length = 512, batch_size = 64, dtype= None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.model.eval()
        self.model.to(self.device)


    @torch.no_grad()
    def _encode(self, texts):
        """
        Encode a list of strings.  returns shape (N, EMBED_DIM) as float32 numpy.
        """
        out_chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            # CLS pooling (token 0)
            hidden = self.model(**enc).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            out_chunks.append(hidden.to(torch.float32).cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    def encode_queries(self, texts):
        """Encode N queries -> (N, 768), L2-normalized."""
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        prefixed = [QUERY_INSTRUCTION + t for t in texts]
        return self._encode(prefixed)

    def encode_docs(self, texts):
        """Encode N documents -> (N, 768), L2-normalized."""
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return self._encode(texts)          # NO prefix on documents

    def encode_query(self, text):
        """Single query -> (768,) float32."""
        return self.encode_queries([text])[0]

    def encode_doc(self, text):
        """Single doc -> (768,) float32."""
        return self.encode_docs([text])[0]

    @property
    def embed_dim(self):
        return EMBED_DIM