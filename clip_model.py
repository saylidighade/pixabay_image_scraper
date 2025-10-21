from PIL import Image
import requests
import time
from transformers import CLIPProcessor, CLIPModel
import torch


# (1, 768) for ViT-B/32, (1, 1024) for ViT-L/14, etc
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities



image = Image.open('data/instagram/final_instagram_post.png')
# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Get image embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# print(image_features.shape)  # (1, 768) for ViT-B/32, (1, 1024) for ViT-L/14, etc.
# print(image_features)


start = time.time()
print(probs)
end = time.time()

print(end-start)



# Load the processor and model
# The 'openai/clip-vit-base-patch32' corresponds to ViT-B/32,
# where the 'base' indicates the 'B' (base) size and 'patch32' refers to the 32x32 patch size.
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# The model and processor are now loaded and ready for use.
# You can then use them for tasks like image-text embedding or zero-shot classification.

# Get image embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

print(image_features.shape)
print(image_features)