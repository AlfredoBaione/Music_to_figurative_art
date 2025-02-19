import argparse
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/output/train_validation_no_accel/runs/music_to_image_{}".format(timestamp))


import datasets
import diffusers
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, whoami
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from modules.MusicToken.MusicToken_no_accel import MusicTokenWrapper
from dataloader_colab import VGGMus

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

logger = get_logger(__name__)


def save_progress(embedder, save_embedder):
      logger.info("Saving embeddings")
      torch.save(embedder.state_dict(), save_embedder)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save learned_embeds.bin every X updates steps.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--data_dir", type=str, default ="/content/drive/MyDrive/Tirocinio_Tesi_Baione/Music_to_figurative_art/VGGMus/",
                        help="A folder containing the data.")
    parser.add_argument("--placeholder_token", type=str, default="<*>",
                        help="A token to use as a placeholder for the concept.")
    parser.add_argument("--center_crop", action="store_true",
                        help="Whether to center crop images before resizing to resolution.")
    parser.add_argument("--multiple_tokens", type=bool, default = False)
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many times to repeat the training data.")
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=8765,
                        help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="The resolution for input images, all the images in the train/validation/test dataset will"
                             " be resized to this resolution")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=60000,
                        help="Total number of steps to perform.  If provided, overrides num_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-05,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", type=bool, default=True,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts",'
                             ' "polynomial" "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded"
                             " in the main process.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
                             " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
                             " Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
                             " For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument( "--report_to", type=str, default="tensorboard",
                         help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                              ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--data_set", type=str, default='train', choices=['train', 'validation', 'test'],
                        help="Whether use train, validation or test set")
    parser.add_argument("--lambda_a", type=float, default=0.01,
                        help="Regularization lambda - l1")
    parser.add_argument("--lambda_b", type=float, default=0,
                        help="Regularization lambda - l2")
    parser.add_argument("--lambda_c", type=float, default=0.01,
                        help="Regularization lambda - classification loss")
    parser.add_argument("--run_name", type=str, default='MusicToken',
                        help="Insert run name")
    parser.add_argument("--cosine_loss", type=bool, default=True,
                        help="Use classification loss")
    parser.add_argument("--input_length", type=int, default=30,
                        help="Select the number of seconds of audio you want in each training-sample.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--validation_batch_size", type=int, default=1,
                        help="Batch size (per device) for the validation dataloader.")
    parser.add_argument("--lora", type=bool, default=False,
                        help="Whether train Lora layers or not")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
          token = HfFolder.get_token()
    if organization is None:
          username = whoami(token)["name"]
          return f"{username}/{model_id}"
    else:
          return f"{organization}/{model_id}"
     


def train_validation():

    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    Accelerator(log_with=args.report_to,
          project_dir=logging_dir)
   
    folder_path = args.output_dir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = args.output_dir + 'weights/'
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
         raise ValueError(
             f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
             " `placeholder_token` that is not already in the tokenizer."
        )
    
    # Define the training/validation device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        if torch.cuda.device_count() >= 1:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * torch.cuda.device_count() 
        )
        else:
             args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size 
        )

    

    train_dataset = VGGMus(
        args=args,
        tokenizer=tokenizer,
        logger=logger
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, generator = torch.Generator(device)
    )

     
    # Scheduler and math around the number of steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    model = MusicTokenWrapper(args).to(weight_dtype)

    # send the model to the device
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
        model = nn.DataParallel(model)
    else: 
        model = model.to(device)
    
    params = model.embedder.parameters()
    if args.lora:
        params = list(model.embedder.parameters()) + list(model.lora_layers.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # validation data
    args.data_set = 'validation'
    validation_dataset = VGGMus(
        args=args,
        tokenizer=tokenizer,
        logger=logger
    )   
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.validation_batch_size, shuffle=False, num_workers=args.dataloader_num_workers
    )


    # We need to recalculate our total steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    
    # Train and validation!
    if torch.cuda.device_count() >= 1:
         total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
    else :
         total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
   

    logger.info("***** Running training/validation *****")
    logger.info(f"  Num training examples = {len(train_dataset)}")
    logger.info(f"  Num validating examples = {len(validation_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    first_epoch = 0
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    txt_embeddings = model.text_encoder.get_input_embeddings().weight
    
    epoch_number = 0
    best_vloss = 1_000_000

    for epoch in range(first_epoch, args.num_epochs):

        running_train_loss = 0

        print('EPOCH {}:'.format(epoch_number + 1))
        
        model.train(True)

        for i, batch in enumerate(train_dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                   batch[key] = value.to(device)
            # Convert images to latent space
            latents = model.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            audio_values = batch["audio_values"].to(dtype=weight_dtype)
            aud_features = model.aud_encoder.extract_features(audio_values)[1]
            audio_token = model.embedder(aud_features)

            # Get the text embedding for conditioning
            encoder_hidden_states = model.text_encoder(
                audio_token, input_ids=batch['input_ids'])[0].to(dtype=weight_dtype)

            # Predict the noise residual
            model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


            # Compute the loss and its gradients
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if len(audio_token.shape) > 2:
                norm_dim = 2
            else:
                norm_dim = 1

            # add regularization
            reg_loss = args.lambda_a * torch.mean(torch.abs(audio_token)) + \
                    args.lambda_b * (torch.norm(audio_token, p=2, dim=norm_dim)**2).mean()

            loss += reg_loss

            if args.cosine_loss:
                input_ids = tokenizer(batch['label']).data['input_ids']
                input_ids = [ids[1:-1] for ids in input_ids]
                target = torch.cat([txt_embeddings[ids].mean(dim=0).view(1, -1) for ids in input_ids])
                if args.multiple_tokens:
                    embedds = audio_token[:, -1, :]
                else:
                    embedds = audio_token
                cosine_sim = F.cosine_similarity(embedds, target, dim=1).mean()
                cosine_penalty = (1 - cosine_sim) ** 2
                loss += args.lambda_c * cosine_penalty
            # normalize loss to account for batch accumulation (not needed if loss already averaged within each batch)
            #loss = loss/args.gradient_accumulation_steps
            loss.backward()
            if ((i + 1) % args.gradient_accumulation_steps == 0) or (i + 1 == len(train_dataloader)): 
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                running_train_loss += loss.detach().item()

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"weights/{args.run_name}_learned_embeds-{global_step}.bin")
                save_progress(model.embedder, save_path)
                if args.lora:
                   save_path = os.path.join(args.output_dir, f"weights/{args.run_name}_lora_layers_learned_embeds-{global_step}.bin")
                   save_progress(model.lora_layers, save_path)
            
            if global_step >= args.max_train_steps:
                break

        # train loss computation
        avg_loss_train_epoch = running_train_loss / (i + 1)


        # validation
        running_valid_loss = 0

        model.eval()

        with torch.no_grad():
            for i, vbatch in enumerate(validation_dataloader):
                for key, value in vbatch.items():
                    if isinstance(value, torch.Tensor):
                       vbatch[key] = value.to(device) 
                       
                # Convert images to latent space
                latents = model.vae.encode(vbatch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                audio_values = vbatch["audio_values"].to(dtype=weight_dtype)
                aud_features = model.aud_encoder.extract_features(audio_values)[1]
                audio_token = model.embedder(aud_features)

                # Get the text embedding for conditioning
                encoder_hidden_states = model.text_encoder(
                    audio_token, input_ids=vbatch['input_ids'])[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                     target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                # Compute the loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if len(audio_token.shape) > 2:
                    norm_dim = 2
                else:
                    norm_dim = 1

                # add regularization
                reg_loss = args.lambda_a * torch.mean(torch.abs(audio_token)) + \
                           args.lambda_b * (torch.norm(audio_token, p=2, dim=norm_dim)**2).mean()

                loss += reg_loss

                if args.cosine_loss:
                    input_ids = tokenizer(vbatch['label']).data['input_ids']
                    input_ids = [ids[1:-1] for ids in input_ids]
                    target = torch.cat([txt_embeddings[ids].mean(dim=0).view(1, -1) for ids in input_ids])
                    if args.multiple_tokens:
                        embedds = audio_token[:, -1, :]
                    else:
                        embedds = audio_token
                    cosine_sim = F.cosine_similarity(embedds, target, dim=1).mean()
                    cosine_penalty = (1 - cosine_sim) ** 2
                    loss += args.lambda_c * cosine_penalty
                
                progress_bar.update(1)
                running_valid_loss += loss.detach().item()
                logs = {"running validation loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

     
            # validation loss computation
            avg_loss_valid_epoch = running_valid_loss / (i + 1)

        print('LOSS train {} valid {}'.format(avg_loss_train_epoch, avg_loss_valid_epoch))

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss_train_epoch, 'Validation' : avg_loss_valid_epoch},
                    epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if  avg_loss_valid_epoch < best_vloss:
            best_vloss =  avg_loss_valid_epoch
            model_path = os.path.join(args.output_dir, 'model_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)
        
        epoch_number += 1
    

    # save the progress
    save_path_embedder = os.path.join(args.output_dir, f"learned_embeds.bin")
    save_progress(model.embedder, save_path_embedder)
    if args.lora:
        save_path_lora = os.path.join(args.output_dir, f"learned_embeds_lora_layers.bin")
        save_progress(model.lora_layers, save_path_lora)  

    return best_vloss

if __name__ == "__main__":
    train_validation()
    writer.close()
    

       


        
 



        
       

    
    
    
    

   


           
        
       
   



