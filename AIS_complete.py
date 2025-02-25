import wav2clip
import librosa
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import math



# ------------------------------------------------------------------------------
# Function to upload and process audio
def get_audio_embeddings(audio_paths, model):
    embeddings_list = []
    for audio_path in audio_paths:
        audio, sr = librosa.load(audio_path, sr=16000)
        embeddings = wav2clip.embed_audio(audio, model)
        embeddings_list.append(embeddings)
    return np.array(embeddings_list)

# Function to upload and process immages
def get_image_embeddings(image_paths, model_clip, processor):
    embeddings_list = []
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = model_clip.get_image_features(**inputs)
        embeddings_list.append(embeddings.squeeze().numpy())
    return np.array(embeddings_list)

# Retrieve the file paths from the directories and sort them
def get_sorted_file_paths(directory, file_extension):
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_extension)]
    file_paths.sort()  # Ordina i file in base ai loro nomi
    return file_paths


# Paths to the audio and image directories
audio_directory = 'F:/Music_to_figurative_art/output/performance_measures_test/07/AIS/audio'
audio_directory_2 = 'F:/audio_test'
image_directory = 'F:/Music_to_figurative_art/output/performance_measures_test/07/AIS/image'


# Retrieve the sorted paths of the audio and image files
audio_paths = get_sorted_file_paths(audio_directory, ".wav")
image_paths = get_sorted_file_paths(image_directory, ".jpg")
#ottieni i percorsi dei file audio nella directory di test
audio_paths_2 = [os.path.join(audio_directory_2, f) for f in os.listdir(audio_directory_2) if f.endswith('.wav')]

# Load the wav2clip audio model
audio_model = wav2clip.get_model()

# Load the CLIP model for images
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Retrieve the embeddings for all audio files
audio_embeddings = get_audio_embeddings(audio_paths, audio_model)
audio_embeddings_2 = get_audio_embeddings(audio_paths_2, audio_model)
# Ottieni gli embeddings per tutte le immagini
image_embeddings = get_image_embeddings(image_paths, clip_model, clip_processor)


# Comparison of the embeddings (for example, the dot product)
def compute_similarity(audio_embeddings, image_embeddings):
    similarities = []
    for audio_emb, image_emb in zip(audio_embeddings, image_embeddings):
        audio_emb_normalized = audio_emb / np.linalg.norm(audio_emb)
        image_emb_normalized = image_emb / np.linalg.norm(image_emb)
        similarity = np.dot(audio_emb_normalized, image_emb_normalized.T)
        similarities.append(similarity)
    return np.array(similarities)

similarities_1 = compute_similarity(audio_embeddings, image_embeddings)
similarities_2 = []

for j in range(len(image_embeddings)):
    image_emb = image_embeddings[j]
    image_emb_normalized = image_emb / np.linalg.norm(image_emb)
    sim = 0
    for i in range(len(audio_embeddings_2)):
            audio_emb = audio_embeddings_2[i]
            if not np.array_equal(audio_emb, audio_embeddings[j]):
               audio_emb_normalized = audio_emb / np.linalg.norm(audio_emb)
               sim_intermediate = np.dot(audio_emb_normalized, image_emb_normalized.T)
               sim += sim_intermediate
            else:
                continue
    sim = (sim / i)
    similarities_2.append(sim)

similarities = (similarities_1 - similarities_2)

# Compute the mean similarity
mean_similarity = np.mean(similarities)

# Mapping the mean similarity from [-1, 1] to [0, 100] (percentage)
x = math.acos(mean_similarity)
y = 1 - x * (2 / math.pi)
mean_similarity_percentage = ((y + 1) / 2) * 100

# Print the simliarities and the mean percentage
for i, similarity in enumerate(similarities):
    x_2 = math.acos(similarity)
    y_2 = 1 - x_2 * (2 / math.pi)
    similarity_percentage = ((y_2 + 1) / 2) * 100
    print(f"Similarity between audio {i} and image {i}: {similarity} ({similarity_percentage}%)")

print(f"AIS: {mean_similarity_percentage}%")

# Save the embeddings for later use (optional)
np.save("audio_embeddings.npy", audio_embeddings)
np.save("image_embeddings.npy", image_embeddings)
