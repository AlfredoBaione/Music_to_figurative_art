import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import random
import os
import math



# ------------------------------------------------------------------------------

# Funzione per caricare e processare i file immagine
def get_image_embeddings(image_paths, model_clip, processor):
    embeddings_list = []
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = model_clip.get_image_features(**inputs)
        embeddings_list.append(embeddings.squeeze().numpy())
    return np.array(embeddings_list)

# Ottieni i percorsi dei file nelle directory e ordinali
def get_sorted_file_paths(directory):
    image_extensions = [".jpg", ".JPG", ".Jpg",".jpeg", ".JPEG", ".Jpeg", ".png", ".PNG", ".Png"]
    file_paths=[]
    for ext in image_extensions:
               file_paths.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)])
    file_paths.sort()  # Ordina i file in base ai loro nomi
    return file_paths


# Percorsi alle directory di audio e immagini
ground_truth_directory = 'F:/Music_to_figurative_art/output/performance_measures_test/07/IIS/ground_truth'
image_directory = 'F:/Music_to_figurative_art/output/performance_measures_test/07/IIS/image'
image_directory_2 = 'F:/image_test'


# Ottieni i percorsi ordinati dei file audio e immagine
ground_truth_paths = get_sorted_file_paths(ground_truth_directory)
image_paths = get_sorted_file_paths(image_directory)
#ottieni i percorsi dei file immagine di test
image_paths_2 = get_sorted_file_paths(image_directory_2) 


# Carica il modello CLIP per le immagini
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Ottieni gli embeddings per tutte le immagini
ground_truth_embeddings = get_image_embeddings(ground_truth_paths, clip_model, clip_processor)
image_embeddings = get_image_embeddings(image_paths, clip_model, clip_processor)
image_embeddings_2 = get_image_embeddings(image_paths_2, clip_model, clip_processor)



# Confronto degli embeddings (ad esempio, il prodotto scalare)
def compute_similarity(ground_embeddings, image_embeddings):
    similarities = []
    for ground_emb, image_emb in zip(ground_embeddings, image_embeddings):
        ground_emb_normalized = ground_emb / np.linalg.norm(ground_emb)
        image_emb_normalized = image_emb / np.linalg.norm(image_emb)
        similarity = np.dot(ground_emb_normalized, image_emb_normalized.T)
        similarities.append(similarity)
    return np.array(similarities)

similarities_1 = compute_similarity(ground_truth_embeddings, image_embeddings)
similarities_2 = []
for j in range(len(image_embeddings)):
    image_emb = image_embeddings[j]
    image_emb_normalized = image_emb / np.linalg.norm(image_emb)
    sim = 0
    for i in range(len(image_embeddings_2)):
            image_emb_2 = image_embeddings_2[i]
            if not np.array_equal(image_emb_2, image_embeddings[j]):
               image_emb_normalized_2 = image_emb_2 / np.linalg.norm(image_emb_2)
               sim_intermediate = np.dot(image_emb_normalized_2, image_emb_normalized.T)
               sim += sim_intermediate
            else:
                continue
    sim = (sim / i)
    similarities_2.append(sim)

similarities = (similarities_1 - similarities_2)

# Calcola la media delle similarità
mean_similarity = np.mean(similarities)

# Mappa la media delle similarità da [-1, 1] a [0, 100] percentuale
x = math.acos(mean_similarity)
y = 1 - x * (2 / math.pi)
mean_similarity_percentage = ((y + 1) / 2) * 100


for i, similarity in enumerate(similarities):
    x_2 = math.acos(similarity)
    y_2 = 1 - x_2 * (2 / math.pi)
    similarity_percentage = ((y_2 + 1) / 2) * 100
    print(f"Similarity between audio {i} and image {i}: {similarity} ({similarity_percentage}%)")
    
print(f"IIS: {mean_similarity_percentage}%")

# Salva gli embeddings per un uso successivo (opzionale)
np.save("ground_truth_image_embeddings.npy", ground_truth_embeddings)
np.save("image_embeddings.npy", image_embeddings)