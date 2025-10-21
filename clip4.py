'''
Use a hybrid score at first stage: score = α*s_img + β*s_meta already at retrieval.
Or run parallel retrievals (one by image, one by meta) and merge top-K results before re-ranking.

✅ Recommended practical approach
Stage 1: combined score retrieval (α≈0.5 for general tasks).
Stage 2: re-rank smaller candidate set using cross-modal model (e.g., CLIP cross-encoder or reranker).
Log which modality dominates per query — you’ll see patterns (visual vs textual dominance).

A **cross-modal reranker** takes both the query text *and* each candidate image (or image + metadata) together as input and scores their matching jointly, not via precomputed embeddings.
It’s slower but more accurate — used only on the top-N results from the first retrieval.
Examples: CLIP cross-encoder, BLIP, or SigLIP fine-tuned for text-image matching.

Not strictly compulsory — but highly recommended.
L2 normalization makes cosine similarity equivalent to a simple dot product, ensuring all embeddings lie on the same unit sphere.
Without it, vectors with larger magnitudes can dominate scores and distort similarity rankings, especially when mixing image and text embeddings.

Encode query once (text_emb), then compute separately: s_img = cos(text_emb, image_emb) and s_meta = cos(text_emb, meta_emb)
 and combine scores with weights: score = α*s_img + β*s_meta (tune α,β on a validation set).

retreive top k from query_to_img match and top k from query_to_meta match, merge and rerank
'''

# To utilize both vCPUs for generating embeddings, use Python's `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor` for parallel processing. Below is an example using `ThreadPoolExecutor` to process images in parallel:

import concurrent.futures
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
PC_API_KEY = 'pcsk_7PDM6c_71PYtNVJhLKNsR19RU8rxjfSjbm4zRt3YeFdkJsNfk6ZmKR8dZpxsgqZG2t6N51'
df = pd.read_csv("data/fashion_subset.csv")  # image_id, other metadata columns
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def norm(x): return x / x.norm(dim=-1, keepdim=True)
folder_path = "data/test/"

def process_image(file_name):
    image_path = os.path.join(folder_path, file_name)
    img = Image.open(image_path)
    img_feat = model.get_image_features(**proc(images=img, return_tensors="pt"))
    meta_feat = model.get_text_features(**proc(text="product red dress cotton", return_tensors="pt"))
    img_feat, meta_feat = norm(img_feat), norm(meta_feat)
    img_feat = str(img_feat.detach().cpu().numpy().squeeze())
    meta_feat = str(meta_feat.detach().cpu().numpy().squeeze())
    image_id = file_name.split('.')[0]
    row = df[df['id'] == int(image_id)]
    row['img_embedding'] = img_feat
    row['meta_embedding'] = meta_feat
    return row

def generate_and_save_embeddings(folder_path):
    records = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(process_image, image_files)
        for row in results:
            records.append(row)
    result_df = pd.concat(records, ignore_index=True)
    # print(result_df.head())
    result_df.to_csv("s3://content-marketing-pixabay-images/clip_fashion_embeddings_2.csv", index=False)

generate_and_save_embeddings(folder_path)
# This will use both vCPUs to process images in parallel, speeding up embedding generation.

