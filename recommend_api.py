#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install fastapi


# In[2]:


# !pip install uvicorn


# In[3]:


# !pip install nbimporter


# In[4]:


# !pip install nbformat


# In[5]:


#!conda list torch


# In[1]:


# recommend_api.py
import json, numpy as np, torch, re, pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder

from Final_Project import (
    corpus, items_df, compute_ppr_prior_vector_unified,
    ATTR_KEYS, summary_lookup, pid2row, IMG_MAT,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)


# =========== 工具函数 ==============
def extract_main_image(raw):
    images = raw.get("images") or []
    if isinstance(images, list):
        for it in images:
            if isinstance(it, dict):
                url = it.get("hi_res") or it.get("large") or it.get("thumb") or it.get("url")
                if url:
                    return url
    if isinstance(images, str):
        return images
    return None


# =========== vocab 给前端用 ==========
@app.get("/vocab")
def vocab_api():
    out = {}
    for k, info in ATTR_KEYS.items():
        vocab = info.get("vocab")
        if vocab:
            out[k] = sorted(list(vocab))
    return out


# ==================== 主推荐 API ======================
@app.post("/recommend")
async def recommend(request: Request):

    body = await request.json()
    query_text = (body.get("query") or "").strip()
    if not query_text:
        return {"results": [], "message": "Query is required."}

    # -------- constraints（可能为空） --------
    soft_constraints = body.get("constraints") or {}
    hard_flags = body.get("hard_flags") or {}

    # -------- TopN / PPRK：前端若没填，则使用默认 --------
    topn = body.get("topn")
    ppr_k = body.get("ppr_k")

    topn = int(topn) if topn not in (None, "", 0) else 20
    ppr_k = int(ppr_k) if ppr_k not in (None, "", 0) else 50

    # -------- hard mask --------
    from Final_Project import _allowed_mask_fast  # 引用原 notebook 的硬过滤
    mask_allowed = _allowed_mask_fast(soft_constraints, hard_flags)
    if mask_allowed.sum() == 0:
        return {"results": [], "message": "No products match HARD filters."}

    ids = items_df["id"].astype(str).tolist()

    # ===== PPR prior =====
    v = compute_ppr_prior_vector_unified(corpus, soft_constraints)
    v = np.asarray(v, dtype=np.float32)
    v[~mask_allowed] = 0.0

    if np.max(v) > 0:
        order = np.argsort(-v)[:min(ppr_k, len(v))]
        pid_list = [ids[i] for i in order if v[i] > 0]
    else:
        pid_list = []

    # ===== fallback：CLIP =====
    if not pid_list:
        qv = corpus.enc.encode_text([query_text])[0]
        qv = torch.tensor(qv, dtype=torch.float32, device=DEVICE)
        qv = qv / (qv.norm() + 1e-12)

        allowed_idx = np.nonzero(mask_allowed)[0]
        sims = (qv @ IMG_MAT[allowed_idx].T).cpu().numpy()
        take = min(ppr_k, len(sims))
        idx_local = np.argsort(-sims)[:take]
        pid_list = items_df["id"].iloc[allowed_idx[idx_local]].astype(str).tolist()

    # ===== rerank =====
    id2doc = dict(zip(items_df["id"].astype(str), items_df["text"].astype(str)))
    pairs = [(query_text, id2doc.get(pid, "")) for pid in pid_list]

    ce_probs = np.asarray(
        reranker.predict(pairs, batch_size=256, show_progress_bar=False),
        dtype=np.float32
    )

    # clip sims
    q_clip = corpus.enc.encode_text([query_text])[0]
    q_clip = torch.tensor(q_clip, dtype=torch.float32, device=DEVICE)
    q_clip = q_clip / (q_clip.norm() + 1e-12)
    img_sims = (q_clip @ IMG_MAT[[pid2row[p] for p in pid_list]].T).cpu().numpy()

    # fusion
    from Final_Project import _fused_scores
    fused = _fused_scores(ce_probs, img_sims)

    order = np.argsort(-fused)[:min(topn, len(fused))]

    results = []
    for rank, idx in enumerate(order, 1):
        pid = pid_list[idx]
        row = items_df.iloc[pid2row[pid]]
        summary, how = summary_lookup(pid, row["raw"], row["title"])
        img_url = extract_main_image(row["raw"])

        results.append({
            "rank": rank,
            "pid": pid,
            "title": row["title"],
            "summary": summary,
            "image_url": img_url,
            "scores": {
                "fused": float(fused[idx]),
                "ce": float(ce_probs[idx]),
                "img": float(img_sims[idx]),
            },
            "match_type": how
        })

    return {
        "results": results,
        "message": f"Top-{len(results)} results returned."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommend_api:app", host="127.0.0.1", port=8000)


# In[ ]:




