import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import pandas as pd
import math


# Function to load and process images
def get_image_embedding(image_path, model_clip, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
         embedding = model_clip.get_image_features(**inputs)
         embedding = embedding.squeeze()
    return embedding

def get_image_embeddings(image_paths, model_clip, processor):
    embeddings_list = []
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = model_clip.get_image_features(**inputs)
        embeddings_list.append(embeddings.squeeze())
    return embeddings_list


def get_sorted_file_paths(directory):
    image_extensions = [".jpg", ".JPG", ".Jpg",".jpeg", ".JPEG", ".Jpeg", ".png", ".PNG", ".Png"]
    file_paths=[]
    for ext in image_extensions:
               file_paths.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)])
    file_paths.sort()  # Ordina i file in base ai loro nomi
    return file_paths

# Load the CLIP model for the images 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#paths to images and directories containig inputs having the same label 
main_folder = 'F:/Music_to_figurative_art/output/performance_measures_test/07/GIILS/test'
image_directory = 'F:/Music_to_figurative_art/output/performance_measures_test/07/AIS/image'

image_path = get_sorted_file_paths(image_directory) 
image_embeddings = get_image_embeddings(image_path, clip_model, clip_processor)

results = []
directory_summaries = []

# Traverse each sub-directory in the main directory
for subdir, _, files in os.walk(main_folder):
    if not files:
        continue
    
    # Estract the sub-directory name
    dir_name = os.path.basename(subdir)

    # Load and process all the immages in the current directory
    image_features = []
    image_files = []

    similarities_1 = []
    similarities_2 = []

    for file in files:
        image_path = os.path.join(subdir, file)
        features = get_image_embedding(image_path, clip_model, clip_processor)
        image_features.append(features)
        image_files.append(file)
         
        
    # Compute the cosine similarity between two images generated with inputs having the same label
    num_images = len(image_features)
    total_similarity = 0
    num_pairs = 0
             
    for i in range(num_images):
        for j in range(i + 1, num_images):
            similarity = F.cosine_similarity(image_features[i], image_features[j], dim=0).item()
            total_similarity += similarity
            num_pairs += 1
            results.append({
                'Directory': dir_name,
                'Image 1': image_files[i],
                'Image 2': image_files[j],
                'Cosine Similarity': similarity
            })
            
    
    #Compute the mean of the similarities obtained from images generated with inputs having the same label 
    if num_pairs > 0:
        sim = 0
        for i in range(num_images):
            for j in range(i + 1, num_images):
                    for k in range(len(image_embeddings)):
                            image_emb_2 = image_embeddings[k]
                            if not np.array_equal(image_emb_2, image_features[j]) and not np.array_equal(image_emb_2, image_features[i]) :
                                     sim_intermediate = F.cosine_similarity(image_emb_2, image_features[i], dim=0).item()
                                     sim += sim_intermediate

                            else:
                                  continue
        
        sim = sim / (k - 1)                          
        mean_similarity = (total_similarity - sim) / num_pairs

        # Normalization of the interval [-1,1] into [0,100]%
        x = math.acos(mean_similarity)
        y = 1 - x * (2 / math.pi)
        giils = ((y + 1) / 2) * 100  
        directory_summaries.append({
            'Directory': dir_name,
            'GIILS': giils
        })




# Create a DataFrame Pandas and save the results in a CSV file
df = pd.DataFrame(results)
df.to_csv('images_similarity.csv', index=False)

# Create a Pandas DataFrame for the similarities obtained and save the results in a CSV file
df_summary = pd.DataFrame(directory_summaries)
df_summary.to_csv('label_images_similarity.csv', index=False)

print("Calculation of similarities between images generated with the same label completed and saved in 'images_cosine_similarity.csv'")
print("Images similarities saved in 'label_images_similarity.csv'")

# Print the GIILS values for the different labels, sorted
print(df_summary.sort_values(by='GIILS', ascending=False))
