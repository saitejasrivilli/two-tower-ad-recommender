"""
FAISS index builder — pre-allocated numpy array, no vstack, no MPS.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import faiss
import sys
import os
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_tower_model import TwoTowerModel
from data_preprocessing import CriteoDataPreprocessor

MODEL_DIR = "./models"
DATA_PATH = "./data/synthetic_criteo.txt"

print("=== Rebuilding FAISS Index ===\n")

with open(f"{MODEL_DIR}/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

feature_dims   = preprocessor["feature_dims"]
numerical_cols = preprocessor["numerical_cols"]

user_feature_dims = {f"C{i}": feature_dims[f"C{i}"] for i in range(1, 7)  if f"C{i}" in feature_dims}
ad_feature_dims   = {f"C{i}": feature_dims[f"C{i}"] for i in range(7, 27) if f"C{i}" in feature_dims}

model = TwoTowerModel(
    user_feature_dims=user_feature_dims,
    ad_feature_dims=ad_feature_dims,
    numerical_dim=len(numerical_cols),
    embedding_dim=16,
    hidden_dims=[512, 256],
    output_dim=256,
)
ckpt = torch.load(f"{MODEL_DIR}/two_tower_best.pt", map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt))

def remove_batchnorm(m):
    for name, child in m.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(m, name, nn.Identity())
        else:
            remove_batchnorm(child)

remove_batchnorm(model)
model.eval()
print("Model ready.")

print("Loading data...")
prep = CriteoDataPreprocessor()
df = prep.load_criteo_data(DATA_PATH, nrows=100000)
train_df, _ = train_test_split(df, test_size=0.3, random_state=42)
train_data = prep.fit_transform(train_df)

ad_cat_np = train_data["categorical"][:, 6:]
n = len(ad_cat_np)

# Pre-allocate output array — avoids vstack memory spike
ad_embeddings = np.zeros((n, 256), dtype="float32")

print(f"Generating {n} embeddings...")
CHUNK = 5000
with torch.no_grad():
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        batch = torch.tensor(ad_cat_np[start:end], dtype=torch.long)
        emb = model.get_ad_embeddings(batch)
        ad_embeddings[start:end] = emb.detach().numpy()
        print(f"  {end}/{n}")

faiss.normalize_L2(ad_embeddings)

# Build and save index
print("Building index...")
index = faiss.IndexFlatIP(256)
index.add(ad_embeddings)
faiss.write_index(index, f"{MODEL_DIR}/faiss_index.bin")

metadata = {
    "dimension": 256,
    "index_type": "Flat",
    "nlist": 100,
    "nprobe": 10,
    "id_map": list(range(n)),
}
with open(f"{MODEL_DIR}/faiss_index.bin.metadata", "wb") as f:
    pickle.dump(metadata, f)

print(f"\n✓ Saved faiss_index.bin ({n} ads)")
print("Now run: python3 inference.py --demo")