import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from transformers import CLIPTokenizer

from dataloader_colab import VGGMus
from modules.MusicToken.MusicToken_no_accel import MusicTokenWrapper

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a testing script.")
    parser.add_argument("--learned_embeds", type=str, default='/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/output/train_validation_no_accel/embedder_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_vae", type=str, default='output/train_validation_no_accel/vae_learned_embeds.bin',
                        help="Path to pretrained vae")
    parser.add_argument("--learned_aud_encoder", type=str, default='output/train_validation_no_accel/aud_encoder_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_unet", type=str, default='output/train_validation_no_accel/unet_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_embeds_lora", type=str, default='/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/train_validation_no_accel/lora_layers_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--data_dir", type=str,
                        help="A folder containing the data.")
    parser.add_argument("--placeholder_token", type=str, default="<*>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/output/test_no_accel/",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="A seed for reproducible testing.")
    parser.add_argument("--resolution", type=int, default=768,
                        help="The resolution for input images, all the images in the train/validation/test"
                             " dataset will be resized to this resolution")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of subprocesses to use for data loading."
                             " 0 means that the data will be loaded in the main process.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
                             " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
                             " Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
                             " For more information, see"
                             " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                             ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test', choices=['train','validation','test'],
                        help="Whether use train or test set")
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='MusicToken',
                        help="Insert run name")
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str, default='an art image of <*>, 4k, high resolution')
    parser.add_argument("--input_length", type=int, default=30,
                        help="Select the number of seconds of audio you want in each test-sample.")
    parser.add_argument("--lora", type=bool, default=False,
                        help="Whether load Lora layers or not")
    parser.add_argument("--aud_encoder", type=bool, default=False,
                        help="Whether load aud_encoder layers or not")
    parser.add_argument("--unet", type=bool, default=False,
                        help="Whether load unet layers or not")
    parser.add_argument("--vae", type=bool, default=False,
                        help="Whether load vae layers or not")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a data directory.")

    return args


def inference(args):

    Accelerator()


    folder_path = args.output_dir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path =args.output_dir + "imgs/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = args.output_dir + f"/imgs/{args.run_name}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
   
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
    
   

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # Define the test device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = VGGMus(
        args=args,
        tokenizer=tokenizer,
        logger=logger,
        size=args.resolution,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=args.dataloader_num_workers, generator = torch.Generator(device)
    )
    
    # the model
    weight_dtype = torch.float32
    model = MusicTokenWrapper(args).to(weight_dtype).eval()
    # send the model to the device
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
        model = nn.DataParallel(model)
    else: 
        model = model.to(device)
    

    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    prompt = args.prompt

    for step, batch in enumerate(test_dataloader):
        for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                   batch[key] = value.to(device)
                   
        if step >= args.generation_steps:
            break
        
        # Audio's feature extraction
        audio_values = batch["audio_values"].to(dtype=weight_dtype)
        aud_features = model.aud_encoder.extract_features(audio_values)[1]
        audio_token = model.embedder(aud_features)

        token_embeds = model.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = audio_token.clone()

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=model.text_encoder,
            vae=model.vae,
            unet=model.unet,
        ).to(device)

        # Disable the NSFW filter
        #pipeline.safety_checker = lambda images, **kwargs: (images, False)
        pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        
        
        seed = random.randint(0, 10000)
        generator = torch.Generator(device).manual_seed(seed)

        image = pipeline(prompt, num_inference_steps=args.num_inference_steps, guidance_scale=7.5, generator=generator).images[0]
        image.save(f'/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/output/test_no_accel/imgs/{args.run_name}/{batch["aud_id"]}_{batch["image_id"]}_{batch["label"]}.jpg')


if __name__ == "__main__":
    args = parse_args()
    inference(args)