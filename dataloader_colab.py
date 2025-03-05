import torchaudio
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import pandas as pd
from pathlib import Path

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

# ------------------------------------------------------------------------------
imagenet_templates_small = [
    "an art image of {}"
]

class Museart(Dataset):  # Cambiato il nome della classe da VGGMus a Museart

    def __init__(self, args, tokenizer, logger, size=512, interpolation='bicubic'):
       """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
       """
       image_lst = 'image/'
       audio_lst = 'audio/'
       
       self.image_root_dir = args.data_dir + image_lst
       self.audio_root_dir = args.data_dir + audio_lst

       self.df_image = pd.read_csv('/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/Museart/images.csv')
       self.df_audio = pd.read_csv('/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/Museart/audio.csv')

       self.image_path = list()
       self.audio_path = list()
       self.label = list()

       self.tokenizer = tokenizer
       self.size = size
       self.placeholder_token = args.placeholder_token
       self.data_set = args.data_set
       self.input_length = args.input_length

       if self.data_set == 'train' or self.data_set == 'validation':
            self.center_crop = args.center_crop
       else:
            self.center_crop = False

       audios = set([file_path[:-4] for file_path in os.listdir(self.audio_root_dir)])
       samples_audio = audios 
       
       self.df_music =  self.df_audio[self.df_audio["set"] == self.data_set]
       self.df_images = self.df_image[self.df_image["set"] == self.data_set]
       self.prepare_dataset(samples_audio)
       
       self.num_samples = len(self.audio_path)
       self._length = self.num_samples

       logger.info(f"{args.data_set}, num samples: {self.num_samples}")
      
       self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
       
       self.templates = imagenet_templates_small


    def __len__(self):
        # len(dataset) 
        return self._length
    
    
    def sample_image(self, aud, df_music):   
        # Extract the file audio name without extension
           audio_filename = os.path.splitext(os.path.basename(aud))[0]

        # Find the set and the class of the file audio
           audio_info = df_music.loc[df_music['id'] == audio_filename]
           audio_info_df = pd.DataFrame(audio_info)
           audio_class = audio_info_df.iat[0, 1]
        
        # Find the corresponding images belonging to the same set and class
           corresponding_images = self.df_images.loc[self.df_images['class'] == audio_class]
           if corresponding_images.empty:
                self.audio_path.remove(aud)

        # Chose randomly an image from the list
           corr_images_list = corresponding_images.values.tolist()
           random.shuffle(corr_images_list)
           random_corresponding_image = random.choice(corr_images_list)
           random_image = random_corresponding_image[0]     

           return random_image
    
    
    def prepare_dataset(self, samples_audio):
          
          for aud in list(samples_audio):
             df_aud = self.df_music[self.df_music.id == aud]
             if df_aud.empty:
                continue
             label = df_aud["class"].unique()[0]
             self.audio_path.append(os.path.join(self.audio_root_dir, aud + ".wav"))
             self.label.append(label) 

          for aud in self.audio_path:
             image = self.sample_image(aud, self.df_music)
             self.image_path.append(os.path.join(self.image_root_dir, image + ".jpg")) 

    
    def img_proc(self, aud, df_music):
        image_extensions = [".jpg", ".JPG", ".Jpg",".jpeg", ".JPEG", ".Jpeg", ".png", ".PNG", ".Png"]
        image = self.sample_image(aud, df_music)
        for ext in image_extensions:
            image_path = Path(self.image_root_dir)/f"{image}{ext}"
            if image_path.exists():
               break
        image_file = Image.open(image_path)
        img = np.array(image_file).astype(np.uint8)

        if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                (
                   h,
                   w,
                ) = (
                      img.shape[0],
                      img.shape[1],
                )
                img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 320), resample=self.interpolation)

            # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1) 
    
    
    def aud_proc_beats(self, aud):  
        wav, sr = torchaudio.load(aud)
        # Resample at  44100 Hz 
        sample_rate = 44100
        if sr != sample_rate:
           wav = torchaudio.functional.resample(wav, sr, 44100)
        wav = torch.tile(wav, (1, 10))
        wav = wav[:, :sample_rate*30]
        start = 0
        end = (self.input_length) * sample_rate
        wav = wav[:, start:end]
        return wav[0]
    
    def txt_proc(self):
        text = random.choice(self.templates).format(self.placeholder_token)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
    
    def __getitem__(self, idx):
        # dataset[0]
        example = {}
        
        example["input_ids"] = self.txt_proc()

        audio_id = self.audio_path[idx % self.num_samples].split('/')[-1]
        image_id = self.image_path[idx % self.num_samples].split('/')[-1]
        example['aud_id'] = audio_id
        example['image_id'] = image_id

        example['label'] = self.label[idx % self.num_samples]

        aud = self.audio_path[idx % self.num_samples]
        #im = self.image_path[idx % self.num_samples]
        example["pixel_values"] = self.img_proc(aud, self.df_music)
        example["audio_values"] = self.aud_proc_beats(aud)
    
        return example
