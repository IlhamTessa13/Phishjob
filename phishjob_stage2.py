"""
phishjob_stage2.py
Tahap 2: klasterisasi URL yang sudah diprediksi phishing menjadi
  - Phishing Job (rekrutmen)  vs  Phishing General (non-job)
Pipeline mengikuti notebook: URL -> teks -> BPE tokenizer -> Hybrid Multimodal Self-Attention
Autoencoder (token + 23 fitur numerik) -> embedding z (16 dim) -> StandardScaler -> K-Means k=2.

Berkas yang dibutuhkan di folder `stage2_export/` (dibuat oleh sel export di notebook):
    tokenizer_bpe.json, ae_hybrid_s0.pt, stage2_params.npz, stage2_meta.json
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from phishjob_features import extract_features

SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")


# ---------------------------------------------------------------- teks URL (sama dengan notebook)
def url_to_text(u: str) -> str:
    s = SCHEME_RE.sub("", str(u).strip().lower())
    s = re.sub(r"^www\d*\.", "", s)[:400]
    s = re.sub(r"%[0-9a-f]{2}", " ", s)

    def _mask(m):
        w = m.group(0)
        if re.fullmatch(r"[0-9a-f]{16,}", w):
            return " [HEX] "
        if len(w) >= 12 and re.search(r"\d", w) and re.search(r"[a-z]", w):
            return " [ID] "
        return w

    s = re.sub(r"[a-z0-9]{12,}", _mask, s)
    return re.sub(r"\d+", " [NUM] ", s)


# ---------------------------------------------------------------- arsitektur (nama layer identik dgn notebook)
class MHSA(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(~mask[:, None, None, :], torch.finfo(att.dtype).min).softmax(-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHSA(d, h, p)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        x = x + self.drop(self.att(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))


class URLAE(nn.Module):
    """Varian 'hybrid' (token + fitur). Head decoder tetap didefinisikan agar load_state_dict(strict) lolos."""

    def __init__(self, vocab, n_feat, cfg, pad_id):
        super().__init__()
        d, z_dim = cfg["D_MODEL"], cfg["Z_DIM"]
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg["MAX_LEN"], d)
        self.blocks = nn.ModuleList(
            [Block(d, cfg["N_HEAD"], cfg["D_FF"], cfg["DROPOUT"]) for _ in range(cfg["N_LAYER"])]
        )
        self.ln = nn.LayerNorm(d)
        self.attn_pool = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.mlm_head = nn.Linear(d, vocab)
        self.bow_head = nn.Linear(z_dim, vocab)
        self.feat_enc = nn.Sequential(
            nn.Linear(n_feat, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64), nn.GELU()
        )
        self.feat_dec = nn.Sequential(nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, n_feat))
        self.to_z = nn.Sequential(
            nn.Linear(2 * d + 64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, z_dim)
        )

    @torch.no_grad()
    def encode(self, ids, feats):
        B, L = ids.shape
        mask = ids != self.pad_id
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(L, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x, mask)
        H = self.ln(x)
        scores = self.attn_pool(H).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = scores.softmax(dim=-1)
        h_attn = (H * alpha.unsqueeze(-1)).sum(dim=1)
        parts = [H[:, 0], h_attn, self.feat_enc(feats)]
        return self.to_z(torch.cat([p.float() for p in parts], -1))


# ---------------------------------------------------------------- inferensi
class Stage2Clusterer:
    def __init__(self, export_dir):
        export_dir = Path(export_dir)
        self.meta = json.loads((export_dir / "stage2_meta.json").read_text(encoding="utf-8"))
        prm = np.load(export_dir / "stage2_params.npz")
        self.feat_mean, self.feat_scale = prm["feat_mean"], prm["feat_scale"]
        self.z_mean, self.z_scale = prm["z_mean"], prm["z_scale"]
        self.centroids = prm["centroids"]                  # (2, Z_DIM) pada ruang z yang sudah di-scale
        self.job_cluster = int(prm["job_cluster"])         # indeks centroid = klaster Job / Recruitment

        self.tok = Tokenizer.from_file(str(export_dir / "tokenizer_bpe.json"))
        self.max_len = self.meta["cfg"]["MAX_LEN"]
        self.pad_id = self.tok.token_to_id("[PAD]")
        self.cls_id = self.tok.token_to_id("[CLS]")
        self.features = self.meta["features"]

        self.model = URLAE(self.tok.get_vocab_size(), len(self.features), self.meta["cfg"], self.pad_id)
        state = torch.load(export_dir / self.meta["ae_weights"], map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def _ids(self, url):
        ids = [self.cls_id] + self.tok.encode(url_to_text(url)).ids[: self.max_len - 1]
        arr = np.zeros((1, self.max_len), dtype=np.int64)
        arr[0, : len(ids)] = ids
        return torch.as_tensor(arr)

    def _feats(self, url):
        f = extract_features(url)
        raw = np.array([[f[c] for c in self.features]], dtype=np.float64)
        x = np.log1p(np.clip(raw, 0, None))                       # sama dengan feat_tf pada notebook
        x = (x - self.feat_mean) / self.feat_scale                # StandardScaler
        return torch.as_tensor(np.clip(x, -6, 6).astype(np.float32))

    def embed(self, url):
        return self.model.encode(self._ids(url), self._feats(url)).float().numpy()[0]

    def predict(self, url: str) -> dict:
        z = (self.embed(url) - self.z_mean) / self.z_scale
        d = np.linalg.norm(self.centroids - z[None, :], axis=1)   # jarak ke tiap centroid
        c = int(np.argmin(d))
        is_job = c == self.job_cluster
        d_job, d_gen = float(d[self.job_cluster]), float(d[1 - self.job_cluster])
        near = d_gen / (d_job + d_gen) if (d_job + d_gen) > 0 else 0.5   # kedekatan relatif ke klaster Job (0-1)
        affinity = near if is_job else 1.0 - near
        return {
            "cluster": "job" if is_job else "general",
            "label": "Phishing Job" if is_job else "Phishing General",
            "affinity": affinity,
        }
