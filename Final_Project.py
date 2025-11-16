#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install ipywidgets


# In[1]:





# In[2]:





# In[3]:





# In[4]:





# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[9]:





# In[10]:





# In[11]:





# In[12]:





# In[14]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# Cell 2 — Imports, paths, knobs, helpers
from __future__ import annotations
import os, io, re, json, math, hashlib, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

import faiss
import networkx as nx

import torch
import torch.nn.functional as F
import open_clip
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------- Paths -------------------
CATALOG_JSONL  = "E:/Jupyter Notebook/DSA4213_final/content/meta_All_Beauty_not_discontinued.jsonl"  # catalog to build KG and search
OUT_DIR        = "E:/Jupyter Notebook/DSA4213_final/content/mm_index"
IMG_CACHE      = os.path.join(OUT_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_CACHE, exist_ok=True)

# ------------------- Encoders ----------------
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
E5_MODEL_NAME   = "intfloat/e5-base-v2"  # used only to wire item→item sim edges for the KG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- Best hyperparameters ----------
BEST_ALPHA = 0.80           # PPR teleport
LAMBDA_ATTR = 0.80          # product <-> (brand/attr) edges
LAMBDA_SIM  = 0.20          # product <-> product (similarity) edges
PPR_HOPS    = 2
TOPK_PPR    = 50            # final candidate budget (and returned top-K)

# ------------------- Reproducibility ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)


# ------------------- Helpers ------------------
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def normalize_images_field(images_field):
    """Normalize images into [{'variant': str, 'url': str}, ...]."""
    out = []
    if images_field is None:
        return out
    if isinstance(images_field, str):
        out.append({"variant": "MAIN", "url": images_field}); return out
    if isinstance(images_field, list):
        for it in images_field:
            if isinstance(it, str):
                out.append({"variant": "", "url": it})
            elif isinstance(it, dict):
                for k in ("hi_res", "large", "thumb", "url"):
                    url = it.get(k)
                    if url:
                        out.append({"variant": str(it.get("variant", "")), "url": url})
                        break
    return out

def extract_summary_field(s) -> str:
    if s is None: return ""
    if isinstance(s, str): return s
    if isinstance(s, dict):
        for k in ("short","summary","text"):
            if s.get(k): return str(s[k])
        return " ".join(str(v) for v in s.values())
    if isinstance(s, list): return " ".join(map(str, s))
    return str(s)

def build_text_blob(rec: dict) -> str:
    title = str(rec.get("title") or "")
    summ  = extract_summary_field(rec.get("summary"))
    feats = rec.get("features") or []
    feats = " ".join(map(str, feats[:12])) if isinstance(feats, list) else str(feats)
    desc  = rec.get("description") or []
    desc  = " ".join(map(str, desc[:12])) if isinstance(desc, list) else str(desc)
    details = rec.get("details") or {}
    det = " ".join([f"{k}: {v}" for k, v in details.items() if v is not None]) if isinstance(details, dict) else str(details)
    txt = " \n ".join([title, summ, feats, desc, det])
    return re.sub(r"\s+", " ", txt).strip()

def _sanitize_value(v: str) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _split_multi(v: str) -> List[str]:
    s = _sanitize_value(v)
    parts = re.split(r"[;,/|]| and ", s)
    out = [p.strip() for p in parts if p and p.strip()]
    return out or ([s] if s else [])


# In[2]:


# === Cell 3 (UPDATED) — Attr vocab, KG (brand=attr), CLIP & fast image encoding with progress bars ===
import os, io, re, math, json, requests, faiss
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn.functional as F
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Fallbacks / helpers (leverage legacies if present) ----
try:
    md5
except NameError:
    import hashlib
    def md5(s: str) -> str:
        return hashlib.md5(str(s).encode("utf-8")).hexdigest()

try:
    normalize_images_field
except NameError:
    def normalize_images_field(images):
        """Normalize to a list of {'variant': str, 'url': str} dicts."""
        out = []
        if isinstance(images, list):
            for it in images:
                if not isinstance(it, dict):
                    continue
                url = it.get("url") or it.get("hi_res") or it.get("large") or it.get("thumb")
                if url:
                    out.append({"variant": it.get("variant") or "", "url": url})
        return out

try:
    _split_multi
except NameError:
    def _split_multi(v):
        """Split multi-valued attribute strings into normalized tokens."""
        if v is None: return []
        s = str(v).strip().lower()
        if not s: return []
        parts = re.split(r"[;,/|]", s)
        return [p.strip() for p in parts if p.strip()]

try:
    ensure_dir
except NameError:
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

# Default cache dir for images (fallback)
try:
    IMG_CACHE
except NameError:
    IMG_CACHE = "/content/img_cache"
ensure_dir(IMG_CACHE)

# Hyperparameter fallbacks (will use your globals if already set elsewhere)
try:
    LAMBDA_ATTR
except NameError:
    LAMBDA_ATTR = 0.80
try:
    LAMBDA_SIM
except NameError:
    LAMBDA_SIM  = 0.20

# ---- Attribute vocab (same as your original) ----
item_forms = {'cream','liquid','gel','pair','powder','spray','oil','bar','lotion','pencil','stick','wand','balm','wrap','scrunchie','individual','elastic','sheet','clay','serum','foam','wax','butter','clip','wipes','spiral','ribbon','mask','aerosol','strip'}
materials = {'human hair','synthetic','plastic','acrylic','human','metal','cotton','rubber','faux mink','silicone','silk','ceramic','nylon','polyester','mink fur','stainless steel','wood','acrylonitrile butadiene styrene (abs)'}
hair_types = {'straight','wavy','curly','kinky','coily','all','dry','thick','fine','normal','frizzy','color','damaged'}
age_ranges = {'adult','kid','child','baby','all ages'}
material_features = {'natural','cruelty free','organic','latex free','non-toxic','reusable','vegan','disposable','biodegradable warning','gluten free','certified organic'}
colors = {'black','pink','white','blue','brown','red','natural','clear','gold','silver','green','purple','multicolor','beige'}
skin_types = {'all','sensitive','dry','acne prone','oily','normal','combination'}
styles = {'modern','french','straight','compact','curly','african','classic','art deco','wavy','earloop'}

ATTR_KEYS: Dict[str, Dict[str, Any]] = {
    "item_form":        {"vocab": item_forms,         "beta": 1.0},
    "material":         {"vocab": materials,          "beta": 1.0},
    "hair_type":        {"vocab": hair_types,         "beta": 1.0},
    "age_range":        {"vocab": age_ranges,         "beta": 1.0},
    "material_feature": {"vocab": material_features,  "beta": 1.0},
    "color":            {"vocab": colors,             "beta": 1.0},
    "skin_type":        {"vocab": skin_types,         "beta": 1.0},
    "style":            {"vocab": styles,             "beta": 1.0},
}
# treat brand like any attribute
ATTR_KEYS.setdefault("brand", {"vocab": None, "beta": 1.0})

def _norm_brand(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = re.sub(r"\s+", " ", str(s).strip())
    return s if s else None

class OptimizedKnowledgeGraph:
    """
    Nodes: product, brand, attr(key,value)
    Edges: product↔brand, product↔attr, product↔product (similar_to)
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def _add_product_node(self, product: Dict):
        pid = product.get('parent_asin') or product.get('asin')
        if not pid: return None
        node_id = f"product_{pid}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id, type='product',
                parent_asin=pid,
                title=str(product.get('title') or ""),
                main_category=str(product.get('main_category') or ""),
                price=float(product['price']) if (product.get('price') not in (None, "", "None")) else None,
                average_rating=float(product['average_rating']) if (product.get('average_rating') not in (None, "", "None")) else (
                    float(product['avg_rating']) if (product.get('avg_rating') not in (None, "", "None")) else None
                ),
                rating_number=product.get('rating_number')
            )
        return node_id

    def _attr_node_id(self, key: str, value: str) -> str:
        return f"attr|{key}|{value}"

    def build_knowledge_graph(self, meta_data: List[Dict]) -> nx.MultiDiGraph:
        G = self.graph
        # pass 1: ensure product nodes
        for product in tqdm(meta_data, desc="KG: add product nodes"):
            self._add_product_node(product)
        # pass 2: edges
        for product in tqdm(meta_data, desc="KG: add brand/attr edges"):
            pid = product.get('parent_asin') or product.get('asin')
            if not pid: continue
            pnode = f"product_{pid}"

            # brand
            brand_raw = (product.get('details') or {}).get('Brand') or product.get('store')
            brand = _norm_brand(brand_raw)
            if brand:
                bnode = f"brand_{brand}"
                if not G.has_node(bnode):
                    G.add_node(bnode, type='brand', brand=brand)
                G.add_edge(pnode, bnode, relation='belongs_to_brand', etype='attr', key_name='brand')
                G.add_edge(bnode, pnode, relation='contains_product', etype='attr', key_name='brand')

            # selected attributes
            details = product.get('details') or {}
            raw_map = {
                "Item Form": "item_form","Material": "material","Hair Type": "hair_type",
                "Age Range (Description)": "age_range","Material Feature": "material_feature",
                "Color": "color","Skin Type": "skin_type","Style": "style",
            }
            for raw_key, canon_key in raw_map.items():
                val = details.get(raw_key)
                if not val: continue
                vocab = ATTR_KEYS.get(canon_key, {}).get("vocab")
                for v in _split_multi(val):
                    if (vocab is None) or (v in vocab):
                        anode = self._attr_node_id(canon_key, v)
                        if not G.has_node(anode):
                            G.add_node(anode, type='attr', key=canon_key, value=v)
                        G.add_edge(pnode, anode, relation='has_attr', key_name=canon_key, etype='attr')
                        G.add_edge(anode, pnode, relation='attr_of', key_name=canon_key, etype='attr')
        return G

    def add_item_item_edges_from_text(self, items_df: pd.DataFrame, e5_emb: np.ndarray,
                                      k: int = 10, sim_threshold: float = 0.5, same_category_only: bool = True):
        G = self.graph
        ids = items_df["id"].astype(str).tolist()
        cat = items_df["category"].astype(str).fillna("").tolist()
        index = faiss.IndexFlatIP(e5_emb.shape[1])
        index.add(e5_emb.astype(np.float32))
        D, I = index.search(e5_emb.astype(np.float32), min(k+1, len(ids)))
        for i, (scores, nbrs) in enumerate(tqdm(zip(D, I), total=len(ids), desc="KG: add sim edges")):
            pid_i = ids[i]; cat_i = cat[i]
            for s, j in zip(scores, nbrs):
                if j == i: continue
                if s < sim_threshold: continue
                if same_category_only and cat[j] != cat_i: continue
                pid_j = ids[int(j)]
                u, v = f"product_{pid_i}", f"product_{pid_j}"
                self.graph.add_edge(u, v, relation='similar_to', etype='sim', weight_raw=float(s))
                self.graph.add_edge(v, u, relation='similar_to', etype='sim', weight_raw=float(s))

    # brand treated like any attr; λ’s from globals
    def finalize_edge_weights(self):
        G = self.graph
        deg = dict(G.degree())
        for u, v, k, data in G.edges(keys=True, data=True):
            if data.get("etype") == "sim":
                s = float(data.get("weight_raw", 1.0))
                data["weight"] = float(LAMBDA_SIM * s)
            else:
                key_name = data.get("key_name", None)  # includes 'brand'
                beta = float(ATTR_KEYS.get(key_name, {}).get("beta", 1.0))
                hub = v if G.nodes[v].get("type") in ("attr","brand") else u
                hdeg = max(1, deg.get(hub, 1))
                data["weight"] = float(LAMBDA_ATTR * beta / math.log(1.0 + hdeg))

# ---- CLIP encoder (progress + OOM-adaptive), used later for queries & images ----
class CLIPEncoder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED, device: str = DEVICE):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        # Only used for *queries* in this pipeline; not for the whole catalog
        vecs: List[np.ndarray] = []
        texts = list(texts)
        bs = int(max(1, batch_size))
        pbar = tqdm(range(0, len(texts), bs), total=(len(texts)+bs-1)//bs, desc="Encoding TEXT (CLIP)")
        for i in pbar:
            toks = self.tokenizer(texts[i:i+bs]).to(self.device)
            try:
                if isinstance(self.device, str) and self.device.startswith("cuda"):
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = self.model.encode_text(toks)
                else:
                    feats = self.model.encode_text(toks)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e).lower() and bs > 8 and isinstance(self.device, str) and self.device.startswith("cuda"):
                    bs = max(8, bs // 2)
                    pbar.set_postfix_str(f"OOM→reduce bs={bs}")
                    toks = self.tokenizer(texts[i:i+bs]).to(self.device)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = self.model.encode_text(toks)
                else:
                    raise
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.float().cpu().numpy())
        if vecs:
            return np.vstack(vecs)
        # empty fallback
        dummy = self.model.encode_text(self.tokenizer(["."]).to(self.device))
        return np.zeros((0, dummy.shape[-1]), dtype=np.float32)

# --------- Fast MAIN image encoding (parallel download + DataLoader + AMP) ----------
FAST_MAX_WORKERS = min(32, os.cpu_count() or 16)
IMG_TIMEOUT = 6.0
RETRIES = 2
FAST_BATCH = 16
NUM_WORKERS = 2   # max(2, (os.cpu_count() or 4)//2)
USE_FP16 = True
ONLY_MAIN = True

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=FAST_MAX_WORKERS, pool_maxsize=FAST_MAX_WORKERS, max_retries=RETRIES)
_session.mount("http://", _adapter); _session.mount("https://", _adapter)

def _pick_main_url(rec):
    imgs = normalize_images_field((rec or {}).get("images"))
    chosen = None
    if ONLY_MAIN:
        mains = [im for im in imgs if str(im.get("variant","")).upper() == "MAIN"]
        chosen = mains[0] if mains else (imgs[0] if imgs else None)
    else:
        chosen = imgs[0] if imgs else None
    if not chosen: return None
    return chosen.get("url")

def predownload_main_images(df: pd.DataFrame) -> list:
    index_to_path = [None]*len(df)
    to_fetch = []
    for idx, rec in enumerate(df["raw"].tolist()):
        url = _pick_main_url(rec)
        if not url: continue
        cache_path = os.path.join(IMG_CACHE, md5(url) + ".jpg")
        index_to_path[idx] = cache_path
        if not os.path.exists(cache_path):
            to_fetch.append((url, cache_path))

    def _fetch(url, path):
        try:
            r = _session.get(url, timeout=IMG_TIMEOUT); r.raise_for_status()
            Image.open(io.BytesIO(r.content)).convert("RGB").save(path)
            return True
        except Exception:
            return False

    if to_fetch:
        ok = 0
        with ThreadPoolExecutor(max_workers=FAST_MAX_WORKERS) as ex:
            futs = [ex.submit(_fetch, u, p) for (u, p) in to_fetch]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading MAIN images (parallel)"):
                ok += int(fut.result() is True)
        miss = len(to_fetch) - ok
        if miss:
            print(f"[Images] downloaded: {ok}/{len(to_fetch)} (missed {miss})")
    return index_to_path

class DiskImageDataset(Dataset):
    def __init__(self, index_to_path, preprocess):
        self.samples = [(i, p) for i, p in enumerate(index_to_path) if p and os.path.exists(p)]
        self.preprocess = preprocess
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i, path = self.samples[idx]
        try:
            im = Image.open(path).convert("RGB")
            return self.preprocess(im), i
        except Exception:
            return None

def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    imgs, idxs = zip(*batch)
    return torch.stack(imgs, 0), torch.tensor(idxs, dtype=torch.long)

@torch.no_grad()
def fast_encode_main_images(enc: CLIPEncoder, df: pd.DataFrame,
                            batch_size=FAST_BATCH, num_workers=NUM_WORKERS, fp16=USE_FP16) -> np.ndarray:
    # infer dim from model
    try:
        dim = enc.model.text_projection.shape[1]
    except Exception:
        dim = enc.encode_text(["."], 1).shape[1]
    img_vecs = np.zeros((len(df), dim), dtype=np.float32)

    index_to_path = predownload_main_images(df)
    ds = DiskImageDataset(index_to_path, enc.preprocess)
    if len(ds) == 0:
        print("[Images] No cached images to encode.")
        return img_vecs

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers>0),
        prefetch_factor=(2 if num_workers>0 else None), collate_fn=_collate
    )

    if isinstance(enc.device, str) and enc.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    bs = int(max(1, batch_size))
    for batch in tqdm(loader, total=len(loader), desc="Encoding MAIN images (fast)"):
        if batch[0] is None: continue
        imgs, idxs = batch
        imgs = imgs.to(enc.device, non_blocking=True)
        try:
            if fp16 and isinstance(enc.device, str) and enc.device.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feats = enc.model.encode_image(imgs)
            else:
                feats = enc.model.encode_image(imgs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e).lower() and bs > 16 and isinstance(enc.device, str) and enc.device.startswith("cuda"):
                # fallback: re-run this mini-batch in smaller chunks
                half = max(16, bs // 2)
                feats_list = []
                for i in range(0, imgs.size(0), half):
                    sub = imgs[i:i+half]
                    if fp16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            f = enc.model.encode_image(sub)
                    else:
                        f = enc.model.encode_image(sub)
                    feats_list.append(f)
                feats = torch.cat(feats_list, dim=0)
            else:
                raise
        feats = F.normalize(feats, p=2, dim=-1).float().cpu().numpy()
        img_vecs[idxs.numpy()] = feats
    return img_vecs


# In[3]:


# === Cell 4 (UPDATED) — Build catalog + KG + features with progress bars, NO catalog CLIP-text encoding ===
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Fallbacks if needed
try:
    read_jsonl
except NameError:
    def read_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

try:
    build_text_blob
except NameError:
    def build_text_blob(rec: dict) -> str:
        """Best-effort text blob (fallback)."""
        title = str(rec.get("title") or "")
        desc  = " ".join(map(str, (rec.get("description") or [])))
        feats = " ".join(map(str, (rec.get("features") or [])))
        store = str((rec.get("details") or {}).get("Brand") or rec.get("store") or "")
        return " ".join([title, store, feats, desc]).strip()

try:
    _sanitize_value
except NameError:
    def _sanitize_value(s):
        return re.sub(r"\s+", " ", str(s).strip()).lower()

# Model name fallbacks (use your globals if already set)
try:
    CLIP_MODEL_NAME
except NameError:
    CLIP_MODEL_NAME = "ViT-B-32"
try:
    CLIP_PRETRAINED
except NameError:
    CLIP_PRETRAINED = "openai"
try:
    E5_MODEL_NAME
except NameError:
    E5_MODEL_NAME = "intfloat/e5-base-v2"
try:
    DEVICE
except NameError:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Corpus:
    df: pd.DataFrame
    text_emb: Optional[np.ndarray]   # None (we skip catalog-wide CLIP text)
    img_emb: np.ndarray
    e5_emb: np.ndarray
    kg: OptimizedKnowledgeGraph
    enc: CLIPEncoder
    e5: SentenceTransformer

def build_catalog_corpus(jsonl_path: str, out_dir: str,
                         sim_k: int = 10, sim_threshold: float = 0.5) -> Corpus:
    ensure_dir(out_dir)

    # Load raw
    raw_records = [rec for rec in read_jsonl(jsonl_path)]
    print(f"Loaded raw records: {len(raw_records)}")

    # Build rows with progress
    rows = []
    for i, rec in enumerate(tqdm(raw_records, desc="Catalog: materialize rows")):
        pid = rec.get("parent_asin") or rec.get("asin") or md5(rec.get("title", f"row{i}"))
        title = str(rec.get("title") or "")
        brand = (rec.get("details") or {}).get("Brand") or rec.get("store")
        category = rec.get("main_category") if isinstance(rec.get("main_category"), str) else None
        price = rec.get("price")
        average_rating = rec.get("average_rating")
        if average_rating is None:
            average_rating = rec.get("avg_rating")
        text_blob = build_text_blob(rec)
        rows.append({
            "id": pid,
            "title": title,
            "brand": (brand if brand is None else _sanitize_value(brand)),
            "category": (None if category is None else _sanitize_value(category)),  # kept in DF for analytics/same_category_only
            "price": float(price) if isinstance(price, (int,float,str)) and str(price) not in ("", "None") else None,
            "average_rating": float(average_rating) if isinstance(average_rating, (int,float,str)) and str(average_rating) not in ("", "None") else None,
            "text": text_blob,
            "raw": rec,
        })
    df = pd.DataFrame(rows)
    print(f"Loaded catalog: {len(df)} products")

    # ---- Build KG with selected attributes (no numeric & no category nodes) ----
    kg = OptimizedKnowledgeGraph()
    kg.build_knowledge_graph(raw_records)

    # ---- Encoders & vectors ----
    print("Init CLIP encoder…")
    enc = CLIPEncoder(CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE)

    # IMPORTANT: Skip catalog-wide CLIP text encoding (we only need CLIP text per-query at inference).
    enc = CLIPEncoder(CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE)
    text_emb = enc.encode_text(df["text"].tolist())
    dim = text_emb.shape[1]

    # Images (catalog only) — fast path with progress bars
    print("Encoding MAIN images (progress)…")
    img_emb  = fast_encode_main_images(enc, df, batch_size=FAST_BATCH, num_workers=0, fp16=USE_FP16)

    # E5 (text-only) for item→item edges in KG
    print("Init E5 text encoder…")
    e5 = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
    e5_inputs = (df["title"].astype(str) + " \n " + df["text"].astype(str)).tolist()
    print("Encoding E5 for item→item edges (progress)…")
    e5_emb = e5.encode(e5_inputs, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    assert e5_emb.shape[0] == len(df)

    # ---- Build item→item edges, finalize weights ----
    kg.add_item_item_edges_from_text(items_df=df, e5_emb=e5_emb, k=sim_k, sim_threshold=sim_threshold, same_category_only=True)
    kg.finalize_edge_weights()

    # Device sanity
    try:
        print(f"DEVICE={DEVICE}, cuda_available={torch.cuda.is_available()}, CLIP on {next(enc.model.parameters()).device}")
    except Exception:
        pass

    return Corpus(df=df, text_emb=text_emb, img_emb=img_emb, e5_emb=e5_emb, kg=kg, enc=enc, e5=e5)

# Build (shows progress for downloads & encoders)
corpus = build_catalog_corpus(CATALOG_JSONL, OUT_DIR, sim_k=5, sim_threshold=0.5)
items_df = corpus.df.copy()


# In[4]:


# Cell 5 — Unified PPR with hard price/rating filters (α=0.80, hops=2)
DETAIL_KEYS = ["item_form","material","hair_type","age_range","material_feature","color","skin_type","style"]

def _parse_float_safe(x):
    if x is None: return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            xf = float(x);  return None if np.isnan(xf) else xf
        except Exception: return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    if s == "" or s.lower() in {"none","null","nan","n/a"}: return None
    try:
        return float(s)
    except Exception:
        return None

def _norm_str(v):
    if v is None: return None
    s = str(v).strip()
    return s if s else None

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return [str(v).strip().lower() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s: return []
    parts = re.split(r"[;,/|]", s)
    out = [p.strip().lower() for p in parts if p.strip()]
    return out if out else [s.lower()]

def make_constraints(C_raw) -> SimpleNamespace:
    if isinstance(C_raw, SimpleNamespace):
        d = dict(C_raw.__dict__)
    elif isinstance(C_raw, dict):
        d = C_raw
    else:
        d = {}
    brand      = _norm_str(d.get("brand"))
    category   = _norm_str(d.get("category"))
    price_min  = _parse_float_safe(d.get("price_min"))
    price_max  = _parse_float_safe(d.get("price_max"))
    rating_min = _parse_float_safe(d.get("rating_min"))
    if price_min is not None and price_max is not None and price_min > price_max:
        price_min, price_max = price_max, price_min

    details = {}
    for k in DETAIL_KEYS:
        vals = _as_list(d.get(k))
        if not vals: continue
        if k in ATTR_KEYS:
            vocab = ATTR_KEYS[k]["vocab"]
            vals = [vv for vv in vals if (vocab is None or vv in vocab)]
        if vals: details[k] = vals

    return SimpleNamespace(
        brand=brand, category=category,
        price_min=price_min, price_max=price_max, rating_min=rating_min,
        details=details
    )

def _expand_k_hop(G: nx.MultiDiGraph, starts: List[str], hops: int) -> set:
    frontier = set(starts); curr = set(starts)
    for _ in range(max(0, int(hops))):
        nxt = set()
        for n in curr:
            if n not in G: continue
            nxt.update(G.predecessors(n)); nxt.update(G.successors(n))
        curr = nxt - frontier
        frontier |= nxt
    return frontier

def _allowed_product_nodes(corpus, C: SimpleNamespace) -> set:
    df = corpus.df
    ids = df["id"].astype(str).tolist()
    prices  = pd.to_numeric(df["price"], errors="coerce").to_numpy()
    ratings = pd.to_numeric(df["average_rating"], errors="coerce").to_numpy()
    pmn  = getattr(C, "price_min", None)
    pmx  = getattr(C, "price_max", None)
    rmin = getattr(C, "rating_min", None)
    mask = np.ones(len(ids), dtype=bool)
    if pmn is not None: mask &= (~np.isnan(prices)) & (prices >= float(pmn))
    if pmx is not None: mask &= (~np.isnan(prices)) & (prices <= float(pmx))
    if rmin is not None: mask &= (~np.isnan(ratings)) & (ratings >= float(rmin))
    return {f"product_{pid}" for pid, keep in zip(ids, mask) if keep}

def _build_unified_personalization(G: nx.MultiDiGraph, C: SimpleNamespace) -> Dict[str, float]:
    pers = {}
    if getattr(C, "brand", None):
        bnode = f"brand_{C.brand}"
        if bnode in G:
            w = float(ATTR_KEYS["brand"].get("beta", 1.0))
            pers[bnode] = pers.get(bnode, 0.0) + w
    if getattr(C, "details", None):
        for key, values in (C.details or {}).items():
            if not values: continue
            beta = float(ATTR_KEYS.get(key, {}).get("beta", 1.0))
            for v in values:
                anode = f"attr|{key}|{v}"
                if anode in G:
                    pers[anode] = pers.get(anode, 0.0) + beta
    return pers

_PPR_UNIFIED_CACHE = {}

def compute_ppr_prior_vector_unified(
    corpus,
    C_raw,
    alpha: float = BEST_ALPHA,
    hops: int = PPR_HOPS,
    normalize: bool = True,
) -> np.ndarray:
    C = C_raw if isinstance(C_raw, SimpleNamespace) else make_constraints(C_raw)
    G = corpus.kg.graph
    ids_all = corpus.df["id"].astype(str).tolist()
    allowed_products = _allowed_product_nodes(corpus, C)
    if (C.price_min is not None or C.price_max is not None or C.rating_min is not None) and not allowed_products:
        return np.zeros((len(ids_all),), dtype=np.float32)

    pers = _build_unified_personalization(G, C)
    if not pers: return np.zeros((len(ids_all),), dtype=np.float32)

    detail_key = tuple((k, tuple(sorted(vs))) for k, vs in sorted((C.details or {}).items()))
    key = (md5("\x1e".join(ids_all)), C.brand, C.category, C.price_min, C.price_max, C.rating_min,
           detail_key, float(alpha), int(hops), float(LAMBDA_ATTR), float(LAMBDA_SIM))
    if key in _PPR_UNIFIED_CACHE:
        return _PPR_UNIFIED_CACHE[key]

    seed_nodes = list(pers.keys())
    keep = _expand_k_hop(G, seed_nodes, hops=hops)
    # pad one-hop neighbors
    extra = set()
    for n in list(keep):
        if n not in G: continue
        extra |= set(G.predecessors(n)) | set(G.successors(n))
    keep |= extra
    H = G.subgraph(keep).copy()

    # prune disallowed products
    if allowed_products:
        drop = [n for n in list(H.nodes()) if (G.nodes[n].get("type") == "product" and n not in allowed_products)]
        if drop: H.remove_nodes_from(drop)

    if H.number_of_nodes() == 0:
        v = np.zeros((len(ids_all),), dtype=np.float32); _PPR_UNIFIED_CACHE[key] = v; return v

    s = sum(pers.get(n, 0.0) for n in pers if n in H)
    persH = {n: (pers[n] / s) for n in pers if n in H and pers[n] > 0.0}
    if not persH:
        v = np.zeros((len(ids_all),), dtype=np.float32); _PPR_UNIFIED_CACHE[key] = v; return v

    pr = nx.pagerank(H, alpha=alpha, personalization=persH, max_iter=100, tol=1e-6, weight="weight")
    v = np.array([float(pr.get(f"product_{pid}", 0.0)) for pid in ids_all], dtype=np.float32)
    if normalize and v.size:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        v = (v - vmin)/(vmax - vmin) if vmax > vmin else np.zeros_like(v, dtype=np.float32)
    v = v.astype(np.float32)
    _PPR_UNIFIED_CACHE[key] = v
    return v

def ppr_topk_unified(constraints: dict, topk: int = TOPK_PPR, corpus=None) -> List[str]:
    ids = corpus.df["id"].astype(str).tolist()
    v = compute_ppr_prior_vector_unified(corpus, constraints, alpha=BEST_ALPHA, hops=PPR_HOPS)
    if float(np.max(v)) <= 0.0:
        return []
    order = np.argsort(-v)[:topk]
    return [ids[i] for i in order]


# In[5]:


# === Cell A — Summary Index Builder (JSONL → ID/Title maps + lookup) ===
# Config: set path to your uploaded JSONL
SUMMARY_JSONL_PATH = "/content/summarized_for_review_merged_final.jsonl"
USE_TITLE_FALLBACK = True  # set False to disable title-based matching

import os, re, json

def _norm_id(s):
    if s in (None, "", "None"): return None
    return str(s).strip().upper()

def _norm_title(s):
    s = (s or "")
    s = s.lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)  # keep letters/digits
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _iter_jsonl_rows(path):
    """Robust JSONL reader: fixes bare NaN, trims trailing commas."""
    if not os.path.exists(path):
        print(f"[Summary] File not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Replace bare NaN tokens with null (valid JSON)
            s = re.sub(r'(?<=[:\s])NaN(?=[,\s}\]])', 'null', s)
            try:
                yield json.loads(s)
            except Exception:
                s2 = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
                try:
                    yield json.loads(s2)
                except Exception:
                    continue

def build_summary_indices_jsonl(path: str):
    """
    Returns:
      id_map:    {UPPER(parent_asin/asin/id) -> summary}
      title_map: {norm_title(title) -> summary}
    """
    id_map, title_map = {}, {}
    id_cnt = title_cnt = 0
    for row in _iter_jsonl_rows(path) or []:
        summary = row.get("summary")
        if summary is None or str(summary).strip() == "":
            continue
        # Index by ID-like keys
        for k in ("parent_asin", "asin", "id"):
            kid = _norm_id(row.get(k))
            if kid:
                id_map[kid] = str(summary)
                id_cnt += 1
        # Index by normalized title (fallback)
        t = _norm_title(row.get("title"))
        if USE_TITLE_FALLBACK and t:
            title_map[t] = str(summary)
            title_cnt += 1
    print(f"[Summary] Indexed: {len(id_map)} by ID, {len(title_map)} by title (from {path})")
    return id_map, title_map

def load_summary_indices(path: str = SUMMARY_JSONL_PATH):
    global SUMMARY_ID_MAP, SUMMARY_TITLE_MAP
    SUMMARY_ID_MAP, SUMMARY_TITLE_MAP = build_summary_indices_jsonl(path)

def summary_lookup(pid: str, prod_row: dict, title: str, use_title_fallback: bool = USE_TITLE_FALLBACK):
    """
    Try ID → (parent_asin/asin/id) then optional title fallback.
    Returns: (summary or None, hit_type: 'id'|'title'|None)
    """
    # ID candidates
    for candidate in (
        _norm_id(pid),
        _norm_id((prod_row or {}).get("parent_asin")),
        _norm_id((prod_row or {}).get("asin")),
        _norm_id((prod_row or {}).get("id")),
    ):
        if candidate and candidate in SUMMARY_ID_MAP:
            return SUMMARY_ID_MAP[candidate], "id"
    # Title fallback
    if use_title_fallback:
        t = _norm_title(title)
        if t and t in SUMMARY_TITLE_MAP:
            return SUMMARY_TITLE_MAP[t], "title"
    return None, None

# Build indices now (call again anytime after replacing the JSONL file)
load_summary_indices(SUMMARY_JSONL_PATH)


# In[6]:


# === Cell B — Interactive Inference UI (Hard Filters + Fusion + uses prebuilt summary indices) ===
# - Requires SUMMARY_ID_MAP, SUMMARY_TITLE_MAP, summary_lookup from Cell A

import json, math, re
import numpy as np
import pandas as pd
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output

# ---- Preconditions ----
need = [
    "corpus","items_df","compute_ppr_prior_vector_unified","ATTR_KEYS","DEVICE",
    "SUMMARY_ID_MAP","SUMMARY_TITLE_MAP","summary_lookup"
]
for n in need:
    assert n in globals(), f"Missing {n}. Run your build & indexing cells first."

# ---- Small helpers / fallbacks ----
try:
    _split_multi
except NameError:
    def _split_multi(v):
        if v is None: return []
        s = str(v).strip().lower()
        if not s: return []
        parts = re.split(r"[;,/|]", s)
        return [p.strip() for p in parts if p.strip()]

RAW_ATTR_MAP = {
    "Item Form": "item_form",
    "Material": "material",
    "Hair Type": "hair_type",
    "Age Range (Description)": "age_range",
    "Material Feature": "material_feature",
    "Color": "color",
    "Skin Type": "skin_type",
    "Style": "style",
}
CANON_KEYS = list(RAW_ATTR_MAP.values())

if 'pid2row' not in globals():
    pid2row = {pid: i for i, pid in enumerate(corpus.df["id"].astype(str).tolist())}

if 'IMG_MAT' not in globals():
    IMG_MAT = torch.from_numpy(corpus.img_emb).to(DEVICE, dtype=torch.float32)
    IMG_MAT = IMG_MAT / (IMG_MAT.norm(dim=1, keepdim=True) + 1e-12)

if 'reranker' not in globals():
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

def _parse_float_ui(x):
    if x is None: return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none","null","nan","n/a"}: return None
    try: return float(s)
    except Exception: return None

def _to_lower_list(vals):
    if vals is None: return []
    return [str(v).strip().lower() for v in vals if str(v).strip()]

def _collect_constraints_from_widgets():
    C = {
        "brand": (w_brand.value or "").strip().lower() or None,
        "price_min": _parse_float_ui(w_price_lo.value),
        "price_max": _parse_float_ui(w_price_hi.value),
        "rating_min": _parse_float_ui(w_rating.value),
        "item_form":        _to_lower_list(w_item_form.value),
        "material":         _to_lower_list(w_material.value),
        "hair_type":        _to_lower_list(w_hair_type.value),
        "age_range":        _to_lower_list(w_age_range.value),
        "material_feature": _to_lower_list(w_material_feature.value),
        "color":            _to_lower_list(w_color.value),
        "skin_type":        _to_lower_list(w_skin_type.value),
        "style":            _to_lower_list(w_style.value),
    }
    return {k:v for k,v in C.items() if (v is not None and v != [] )}

def _hard_flags_from_widgets():
    return {
        "brand":            bool(w_hard_brand.value),
        "price_min":        bool(w_hard_pmin.value),
        "price_max":        bool(w_hard_pmax.value),
        "rating_min":       bool(w_hard_rmin.value),
        "item_form":        bool(w_hard_item_form.value),
        "material":         bool(w_hard_material.value),
        "hair_type":        bool(w_hard_hair_type.value),
        "age_range":        bool(w_hard_age_range.value),
        "material_feature": bool(w_hard_material_feature.value),
        "color":            bool(w_hard_color.value),
        "skin_type":        bool(w_hard_skin_type.value),
        "style":            bool(w_hard_style.value),
    }

@torch.no_grad()
def _encode_clip_text(texts: list[str]) -> torch.Tensor:
    vec = corpus.enc.encode_text(texts)
    t = torch.tensor(vec, dtype=torch.float32, device=DEVICE)
    return t / (t.norm(dim=1, keepdim=True) + 1e-12)

@torch.no_grad()
def _gather_img_sims(q_clip_vec: torch.Tensor, pid_list: list[str]) -> np.ndarray:
    if not pid_list: return np.zeros((0,), dtype=np.float32)
    idx = torch.tensor([pid2row[p] for p in pid_list], device=DEVICE, dtype=torch.long)
    sims = (q_clip_vec.view(1, -1) @ IMG_MAT[idx].T).detach().float().cpu().numpy().ravel().astype(np.float32)
    return sims

def _mk_pairs_fulltext(query_text: str, pid_list: list[str]):
    id2doc = dict(zip(items_df["id"].astype(str), items_df["text"].astype(str)))
    return [(query_text, id2doc.get(pid, "")) for pid in pid_list]

def _standardize_local(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m, s = float(x.mean()), float(x.std() + 1e-8)
    return (x - m) / s

def _fused_scores(ce_probs: np.ndarray, img_sims: np.ndarray) -> np.ndarray:
    # fixed LR weights (z-normalized locally)
    w_ce, w_img, b = 0.994, 0.875, -1.381
    z_ce  = _standardize_local(ce_probs)
    z_img = _standardize_local(img_sims)
    logits = w_ce*z_ce + w_img*z_img + b
    return 1.0 / (1.0 + np.exp(-logits))

# -------- HARD MASK (strict) --------
def _allowed_mask_fast(soft_constraints: dict, hard_flags: dict) -> np.ndarray:
    n = len(items_df)
    mask = np.ones(n, dtype=bool)

    # Brand
    if hard_flags.get("brand") and ("brand" in soft_constraints) and soft_constraints["brand"]:
        b = soft_constraints["brand"]
        left = items_df["brand"].fillna("").astype(str).str.strip().str.lower()
        mask &= (left == b)

    # Price_min / price_max
    if hard_flags.get("price_min") and ("price_min" in soft_constraints):
        pmin = soft_constraints["price_min"]
        price = items_df["price"]
        mask &= price.notna() & (price.astype(float) >= float(pmin))
    if hard_flags.get("price_max") and ("price_max" in soft_constraints):
        pmax = soft_constraints["price_max"]
        price = items_df["price"]
        mask &= price.notna() & (price.astype(float) <= float(pmax))

    # Rating_min
    if hard_flags.get("rating_min") and ("rating_min" in soft_constraints):
        rmin = soft_constraints["rating_min"]
        rating = items_df["average_rating"]
        mask &= rating.notna() & (rating.astype(float) >= float(rmin))

    # Attribute keys (OR within-attr, AND across attrs)
    if any(hard_flags.get(k, False) for k in CANON_KEYS):
        raws = items_df["raw"].tolist()

        def _row_has_any(i: int, canon_key: str, want_vals: list[str]) -> bool:
            rec = raws[i] if i < len(raws) else {}
            details = (rec or {}).get("details") or {}
            raw_keys = [rk for rk, ck in RAW_ATTR_MAP.items() if ck == canon_key]
            vals = []
            for rk in raw_keys:
                vals.extend(_split_multi(details.get(rk)))
            if not vals:
                return False
            set_vals = set(vals)
            return any((w in set_vals) for w in want_vals)

        idxs = np.nonzero(mask)[0]
        for canon_key in CANON_KEYS:
            if hard_flags.get(canon_key) and (canon_key in soft_constraints) and soft_constraints[canon_key]:
                want = soft_constraints[canon_key]
                ok = np.array([_row_has_any(i, canon_key, want) for i in idxs], dtype=bool)
                m2 = np.zeros_like(mask)
                m2[idxs] = ok
                mask &= m2

    return mask

# -------- Widgets --------
hard_note = widgets.HTML(
    "<b>Hard filters:</b> Checked fields are enforced <i>before</i> PPR. "
    "Unmatched/missing values are excluded. CE always uses the full item text."
)

w_query = widgets.Textarea(
    value="", description="Intent",
    placeholder="Describe what you’re looking for…",
    layout=widgets.Layout(width="100%", height="72px"),
)

w_brand    = widgets.Text(value="", description="Brand")
w_price_lo = widgets.Text(value="", description="Price_min")
w_price_hi = widgets.Text(value="", description="Price_max")
w_rating   = widgets.Text(value="", description="Rating_min")

# HARD toggles
w_hard_brand = widgets.Checkbox(value=False, description="Hard brand")
w_hard_pmin  = widgets.Checkbox(value=True,  description="Hard price_min")
w_hard_pmax  = widgets.Checkbox(value=True,  description="Hard price_max")
w_hard_rmin  = widgets.Checkbox(value=True,  description="Hard rating_min")

def _opts(key):
    vocab = ATTR_KEYS.get(key, {}).get("vocab")
    return sorted(vocab) if vocab else []

w_item_form        = widgets.SelectMultiple(options=_opts("item_form"),        description="item_form",        rows=6)
w_material         = widgets.SelectMultiple(options=_opts("material"),         description="material",         rows=6)
w_hair_type        = widgets.SelectMultiple(options=_opts("hair_type"),        description="hair_type",        rows=6)
w_age_range        = widgets.SelectMultiple(options=_opts("age_range"),        description="age_range",        rows=6)  # <-- FIXED (no trailing comma)
w_material_feature = widgets.SelectMultiple(options=_opts("material_feature"), description="material_feature", rows=6)
w_color            = widgets.SelectMultiple(options=_opts("color"),            description="color",            rows=6)
w_skin_type        = widgets.SelectMultiple(options=_opts("skin_type"),        description="skin_type",        rows=6)
w_style            = widgets.SelectMultiple(options=_opts("style"),            description="style",            rows=6)

w_hard_item_form        = widgets.Checkbox(value=False, description="Hard item_form")
w_hard_material         = widgets.Checkbox(value=False, description="Hard material")
w_hard_hair_type        = widgets.Checkbox(value=False, description="Hard hair_type")
w_hard_age_range        = widgets.Checkbox(value=False, description="Hard age_range")
w_hard_material_feature = widgets.Checkbox(value=False, description="Hard material_feature")
w_hard_color            = widgets.Checkbox(value=False, description="Hard color")
w_hard_skin_type        = widgets.Checkbox(value=False, description="Hard skin_type")
w_hard_style            = widgets.Checkbox(value=False, description="Hard style")

w_topn    = widgets.IntText(value=20, description="Top-N")
w_topk    = widgets.IntText(value=50, description="PPR@K")
w_outpath = widgets.Text(value="/content/query_output.jsonl", description="Output")

w_btn    = widgets.Button(description="Recommend", button_style="primary")
w_status = widgets.Output()

grid_filters = widgets.GridBox(
    children=[
        w_brand, w_price_lo, w_price_hi, w_rating,
        w_topn, w_topk, w_outpath,
        w_hard_brand, w_hard_pmin, w_hard_pmax, w_hard_rmin
    ],
    layout=widgets.Layout(grid_template_columns="repeat(3, minmax(240px, 1fr))", grid_gap="10px", width="100%"),
)
grid_attrs = widgets.GridBox(
    children=[w_item_form, w_material, w_hair_type, w_age_range, w_material_feature, w_color, w_skin_type, w_style],
    layout=widgets.Layout(grid_template_columns="repeat(4, minmax(220px, 1fr))", grid_gap="8px", width="100%"),
)
grid_attr_hard = widgets.GridBox(
    children=[w_hard_item_form, w_hard_material, w_hard_hair_type, w_hard_age_range,
              w_hard_material_feature, w_hard_color, w_hard_skin_type, w_hard_style],
    layout=widgets.Layout(grid_template_columns="repeat(4, minmax(220px, 1fr))", grid_gap="8px", width="100%"),
)

ui = widgets.VBox([
    hard_note,
    widgets.HTML("<b>Query</b>"),
    w_query,
    widgets.HTML("<b>Constraints & Settings</b>"),
    grid_filters,
    widgets.HTML("<b>Attributes</b>"),
    grid_attrs,
    widgets.HTML("<b>Hard toggles for attributes</b>"),
    grid_attr_hard,
    w_btn,
    w_status
])
display(ui)

# ---------- Inference click ----------
@w_btn.on_click
def _on_click_recommend(_):
    with w_status:
        clear_output()
        print("Running inference…")

        soft_constraints = _collect_constraints_from_widgets()
        hard_flags = _hard_flags_from_widgets()

        mask_allowed = _allowed_mask_fast(soft_constraints, hard_flags)
        allowed_count = int(mask_allowed.sum())
        active_hards = [k for k,v in hard_flags.items() if v and k in soft_constraints]
        print(f"[Hard filters] active={active_hards} → allowable products: {allowed_count} / {len(items_df)}")

        if allowed_count == 0:
            print("No products satisfy the selected HARD filters.")
            with open((w_outpath.value or "/content/query_output.jsonl").strip(), "w", encoding="utf-8") as f: pass
            return

        # Query text
        query_text = (w_query.value or "").strip()
        if not query_text:
            parts = []
            if soft_constraints.get("brand"): parts.append(f"brand: {soft_constraints['brand']}")
            if "price_min" in soft_constraints or "price_max" in soft_constraints:
                parts.append(f"price: [{soft_constraints.get('price_min','?')}, {soft_constraints.get('price_max','?')}]")
            if "rating_min" in soft_constraints: parts.append(f"rating ≥ {soft_constraints['rating_min']}")
            for k in CANON_KEYS:
                if k in soft_constraints and soft_constraints[k]:
                    parts.append(f"{k}: {', '.join(soft_constraints[k])}")
            query_text = " / ".join(parts) if parts else "find suitable products"

        topn        = int(max(1, w_topn.value))
        candidate_k = int(max(1, w_topk.value))
        out_path    = (w_outpath.value or "/content/query_output.jsonl").strip()

        # PPR prior, zero disallowed
        ids = items_df["id"].astype(str).tolist()
        v = compute_ppr_prior_vector_unified(corpus, soft_constraints)
        v = np.asarray(v, dtype=np.float32)
        if v.shape[0] != len(items_df):
            raise RuntimeError(f"PPR vector length {v.shape[0]} != catalog {len(items_df)}")
        v[~mask_allowed] = 0.0

        if float(np.max(v)) > 0.0:
            order = np.argsort(-v)[:min(candidate_k, len(v))]
            pid_list = [ids[i] for i in order if v[i] > 0]
        else:
            pid_list = []

        # Fallback restricted to allowed rows
        if not pid_list:
            qv = _encode_clip_text([query_text])[0:1]
            allowed_idx = np.nonzero(mask_allowed)[0]
            if len(allowed_idx) == 0:
                with open(out_path, "w", encoding="utf-8") as f: pass
                print("No allowed rows for fallback.")
                return
            sims = (qv @ IMG_MAT[allowed_idx].T).detach().float().cpu().numpy().ravel()
            take = min(candidate_k, len(allowed_idx))
            order_local = np.argsort(-sims)[:take]
            pid_list = items_df["id"].iloc[allowed_idx[order_local]].astype(str).tolist()

        if not pid_list:
            with open(out_path, "w", encoding="utf-8") as f: pass
            print(f"No candidates found. Wrote empty file to {out_path}")
            return

        # Features
        pairs    = _mk_pairs_fulltext(query_text, pid_list)
        ce_probs = (np.asarray(reranker.predict(pairs, batch_size=256, show_progress_bar=False), dtype=np.float32)
                    if len(pairs) else np.zeros((0,), np.float32))
        q_clip   = _encode_clip_text([query_text])[0:1]
        img_sims = _gather_img_sims(q_clip[0], pid_list)

        K = len(pid_list)
        if ce_probs.shape[0] != K or img_sims.shape[0] != K:
            print(f"Feature length mismatch (CE={ce_probs.shape}, IMG={img_sims.shape}, K={K}). Aborting.")
            return

        # Fusion
        try:
            fused = np.asarray(blender.predict_proba(np.stack([ce_probs, img_sims], 1))[:,1], dtype=np.float32)
        except Exception:
            fused = _fused_scores(ce_probs, img_sims)

        # Rank
        order = np.argsort(-fused)[:min(topn, K)]
        ranked_pids   = [pid_list[i] for i in order]
        ranked_scores = fused[order]
        ranked_ce     = ce_probs[order]
        ranked_img    = img_sims[order]

        record_constraints = dict(soft_constraints)
        record_constraints["_hard_fields"] = active_hards

        # Write JSONL with summaries
        id_hits = 0
        title_hits = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for rank, (pid, s, c, im) in enumerate(zip(ranked_pids, ranked_scores, ranked_ce, ranked_img), start=1):
                row  = items_df.iloc[pid2row[pid]]
                prod = row["raw"]
                if isinstance(prod, (dict, list)):
                    def _strip(o):
                        if isinstance(o, dict): return {k:_strip(v) for k,v in o.items() if k!="generated"}
                        if isinstance(o, list): return [_strip(v) for v in o]
                        return o
                    prod = _strip(prod)

                summary, how = summary_lookup(pid, prod if isinstance(prod, dict) else {}, row["title"])
                if how == "id": id_hits += 1
                elif how == "title": title_hits += 1

                rec = {
                    "rank":        rank,
                    "query":       query_text,
                    "constraints": record_constraints,
                    "pid":         pid,
                    "title":       row["title"],
                    "summary":     summary,
                    "scores":      {"fused": float(s), "ce": float(c), "img": float(im)},
                    "product":     prod,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[Summary] hits: {id_hits} by ID, {title_hits} by title, "
              f"misses: {len(ranked_pids) - id_hits - title_hits} / {len(ranked_pids)}")

        # Preview
        preview = pd.DataFrame({
            "rank": range(1, len(ranked_pids)+1),
            "pid": ranked_pids,
            "title": [items_df.loc[pid2row[p], "title"] for p in ranked_pids],
            "fused": ranked_scores,
            "ce": ranked_ce,
            "img": ranked_img,
            "summary_found": [1 if summary_lookup(p, items_df.iloc[pid2row[p]]["raw"], items_df.loc[pid2row[p], "title"])[0] else 0
                              for p in ranked_pids],
        })
        print(f"Saved top-{len(ranked_pids)} results to {out_path}")
        display(preview.head(min(10, len(preview))))


# In[ ]:




