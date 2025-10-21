import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import pinecone


# def generate_and_save_embeddings(folder_path, pinecone_index):
def generate_and_save_embeddings(folder_path):

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            embedding = image_features.squeeze().tolist()

            print(type(embedding), len(embedding))

            # Use file name as unique ID
            # pinecone_index.upsert([(file_name, embedding)])

# Example usage:
# pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
# index = pinecone.Index("your-index-name")
generate_and_save_embeddings("data/instagram")
