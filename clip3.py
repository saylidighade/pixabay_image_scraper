'''
# combined = norm(0.6*img_feat + 0.4*meta_feat)   # tune weights
# d = combined.shape[-1]
'''

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, numpy as np
import os
import pandas as pd
from pinecone import Pinecone

PC_API_KEY = 'pcsk_7PDM6c_71PYtNVJhLKNsR19RU8rxjfSjbm4zRt3YeFdkJsNfk6ZmKR8dZpxsgqZG2t6N51'
df = pd.read_csv("data/fashion_subset.csv")  # image_id, other metadata columns
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

INDEX_NAME = 'clip'
pc = Pinecone(api_key=PC_API_KEY, environment="us-east-1")
index = pc.Index(INDEX_NAME)

def norm(x): return x / x.norm(dim=-1, keepdim=True)

def save_to_pinecone(image_id, img_feat, meta_feat, other_metadata_dict, top_category, sub_category):
    # Prepare vectors for upsert
    vectors = []
    for i, row in df.iterrows():
        # Parse embedding string to list if needed
        embedding = eval(row['embedding']) if isinstance(row['embedding'], str) else row['embedding']
        vector_id = f"{i}"
        vectors.append((vector_id, embedding, {"file_name": row['file_name'], "chunk_text": row['chunk_text']}))

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace="US")

def generate_and_save_embeddings(folder_path):
    records = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            img = Image.open(image_path)

            # encode image and metadata
            img_feat = model.get_image_features(**proc(images=img, return_tensors="pt"))
            meta_feat = model.get_text_features(**proc(text="product red dress cotton", return_tensors="pt"))
            img_feat, meta_feat = norm(img_feat), norm(meta_feat)
            img_feat = img_feat.detach().cpu().numpy()
            meta_feat = meta_feat.detach().cpu().numpy()

            # get image id from img filename and extract metadata from csv for that id column
            # same namespace for all -pc id uni - img id is for metadat match

            #filename without extension remove jpg at end
            image_id = file_name.split('.')[0]

            # obtain all columns for that image id from df
            row = df[df['id'] == int(image_id)]
            img_feat = str(img_feat.squeeze())  # shape becomes (512,)
            meta_feat = str(meta_feat.squeeze())  # shape becomes (512,)

            row['img_embedding'] = img_feat
            row['meta_embedding'] = meta_feat

            records.append(row)

    result_df = pd.concat(records, ignore_index=True)
    result_df.to_csv("data/clip_fashion_embeddings.csv", encoding="utf-8", index=False)

# save to index
# save_to_pinecone(image_id, img_feat, meta_feat, other_metadata, top_category="retail_ecommerce", sub_category="fashion_and_apparel")

folder_path = "data/test"
generate_and_save_embeddings(folder_path)


################################# TEST query ################################
# q = model.get_text_features(**proc(text="red summer dress", return_tensors="pt"))
# q = norm(q).detach().cpu().numpy()
