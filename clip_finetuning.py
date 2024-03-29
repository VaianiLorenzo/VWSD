from PIL import Image
Image.MAX_IMAGE_PIXELS = 631770000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import os
import pandas as pd
import argparse
import random

import torch
import torch.utils.data as data
import torch.nn as nn
from torch import optim
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="CLIP finetuning")
parser.add_argument(
    "--textual_input",
    help="Part of sentence to be used as input",
    default=None,
    choices=['full_phrase', 'target_word', 'main_topic'],
    required=True)
parser.add_argument(
    "--log_filename",
    help="Name of the log file",
    default="log.txt",
    required=False)
parser.add_argument(
    "--epochs",
    help="Number of epochs",
    type=int,
    default=30,
    required=False)
parser.add_argument(
    "--batch_size",
    help="Batch size",
    type=int,
    default=16,
    required=False)
parser.add_argument(
    "--model_size",
    help="Size of the CLIP model",
    default="large",
    choices=["large", "base"],
    required=False)    
ta_parser = parser.add_mutually_exclusive_group(required=False)
ta_parser.add_argument('--textual_augmentation', dest='textual_augmentation', action='store_true')
ta_parser.add_argument('--no-textual_augmentation', dest='textual_augmentation', action='store_false')
parser.set_defaults(textual_augmentation=False)
va_parser = parser.add_mutually_exclusive_group(required=False)
va_parser.add_argument('--visual_augmentation', dest='visual_augmentation', action='store_true')
va_parser.add_argument('--no-visual_augmentation', dest='visual_augmentation', action='store_false')
va_parser = parser.add_mutually_exclusive_group(required=False)
va_parser.add_argument('--use_all_data', dest='use_all_data', action='store_true')
va_parser.add_argument('--no-use_all_data', dest='use_all_data', action='store_false')
parser.set_defaults(use_all_data=False)
args = parser.parse_args()

bs = args.batch_size
epochs = args.epochs
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
if args.model_size == "large":
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
elif args.model_size == "base":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# Load the dataset
input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
train_data_path = os.path.join(input_folder_path, "train_data_v1.txt")
train_label_path = os.path.join(input_folder_path, "train_label_v1.txt")
val_data_path = os.path.join(input_folder_path, "val_data_v1.txt")
val_label_path = os.path.join(input_folder_path, "val_label_v1.txt")
images_path = os.path.join(input_folder_path, "train_images_v1")
output_path_root = "checkpoints"

# Load the dataset
train_df = pd.read_csv(train_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
with open(train_label_path, "r") as f:
    train_labels = f.readlines()
train_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in train_labels]
if args.textual_augmentation:
    train_augmented_sentences = []
    languages = ["it", "de", "fr", "fa"]
    for language in languages:
        with open(os.path.join(input_folder_path, "train_back_translation_aug_"+language+".txt"), "r") as f:
            train_augmented_sentences.append([sentence[:-1] for sentence in f.readlines()])
else:
    train_augmented_sentences = None

val_df = pd.read_csv(val_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
with open(val_label_path, "r") as f:
    val_labels = f.readlines()
val_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in val_labels]
if args.textual_augmentation:
    val_augmented_sentences = []
    languages = ["it", "de", "fr", "fa"]
    for language in languages:
        with open(os.path.join(input_folder_path, "val_back_translation_aug_"+language+".txt"), "r") as f:
            val_augmented_sentences.append([sentence[:-1] for sentence in f.readlines()])
else:
    val_augmented_sentences = None

with open(os.path.join("logs", args.log_filename), "w") as f:
    f.write("TRAINING LOG\n")

# Define the dataset
class CLIP_dataset(data.Dataset):
    def __init__(self, list_image_path, list_txt, processor, list_txt_aug=None, ta=False, va=False):
        self.processor = processor
        self.image_path = list_image_path
        self.txt = list_txt
        if list_txt_aug is not None and ta:
            self.list_txt_aug = list_txt_aug
        self.ta = ta
        self.va = va
        
    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):

        if self.ta:
            random_value = random.random()
            if random_value < 0.1:
                text = self.list_txt_aug[0][idx]
            elif random_value < 0.2:
                text = self.list_txt_aug[1][idx]
            elif random_value < 0.3:
                text = self.list_txt_aug[2][idx]
            elif random_value < 0.4:
                text = self.list_txt_aug[3][idx]
            else:
                text = self.txt[idx]       
        else:
            text = self.txt[idx]
        processed_text = self.processor(text=text, return_tensors="pt", max_length=16, padding="max_length", truncation=True)

        if self.va:
            image = self.augment_image(Image.open(self.image_path[idx]))
        else:
            image = Image.open(self.image_path[idx])
        image = self.processor(images=[image], return_tensors="pt")

        #return image, {"input_ids":self.processed_sentences["input_ids"][idx], "attention_mask":self.processed_sentences["attention_mask"][idx]}
        return image, {"input_ids":processed_text["input_ids"][0], "attention_mask":processed_text["attention_mask"][0]}
    
    def augment_image(self, image):
        if random.random() < 0.2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(image)
        if random.random() < 0.2:
            image = image.convert("L")
        return image

# Select the part of sentence to use to train the model        
train_sentences = list(train_df["full_phrase"])
train_ambiguities = list(train_df["target_word"])
train_main_topics = [(sentence[:sentence.find(ambiguity)] + sentence[sentence.find(ambiguity)+len(ambiguity):]).strip() for sentence, ambiguity in zip(train_sentences, train_ambiguities)]

val_sentences = list(val_df["full_phrase"])
val_ambiguities = list(val_df["target_word"])
val_main_topics = [(sentence[:sentence.find(ambiguity)] + sentence[sentence.find(ambiguity)+len(ambiguity):]).strip() for sentence, ambiguity in zip(val_sentences, val_ambiguities)]

if args.textual_input == "full_phrase":
    train_textual_input = train_sentences
    val_textual_input = val_sentences
elif args.textual_input == "target_word":
    train_textual_input = train_ambiguities
    val_textual_input = val_ambiguities
elif args.textual_input == "main_topic":
    train_textual_input = train_main_topics
    val_textual_input = val_main_topics

if args.use_all_data:
    train_gt_image_paths = train_gt_image_paths + val_gt_image_paths
    train_textual_input = train_textual_input + val_textual_input
    if args.textual_augmentation:
        train_augmented_sentences = [train_augmented_sentences[i] + val_augmented_sentences[i] for i in range(len(train_augmented_sentences))]
    train_dataset = CLIP_dataset(train_gt_image_paths, train_textual_input, processor, list_txt_aug=train_augmented_sentences, ta=args.textual_augmentation, va=args.visual_augmentation)
    train_dataloader = data.DataLoader(train_dataset, batch_size=bs, shuffle = True, num_workers=24)
else:
    train_dataset = CLIP_dataset(train_gt_image_paths, train_textual_input, processor, list_txt_aug=train_augmented_sentences, ta=args.textual_augmentation, va=args.visual_augmentation)
    train_dataloader = data.DataLoader(train_dataset, batch_size=bs, shuffle = True, num_workers=24)
    val_dataset = CLIP_dataset(val_gt_image_paths, val_textual_input, processor)
    val_dataloader = data.DataLoader(val_dataset, batch_size=bs, num_workers=24)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

#start_lr = 1e-7
#end_lr = 1e-8
# compute delta for linear scheduler
#gamma = (end_lr / start_lr) ** (1 / epochs)
#print("GAMMA:", gamma)

optimizer = optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=len(train_dataloader), gamma=gamma)

best_val_loss = None
total_loss = 0

# finetuning clip
for epoch in range(epochs):

    print("EPOCH", epoch+1)
    # TRAIN 
    model.train()        
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        images,texts = batch 
        images = images["pixel_values"].squeeze(1).to(device)
        for k in texts.keys():
            texts[k] = texts[k].to(device)
        outputs = model(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], pixel_values=images)
        ground_truth = torch.arange(len(images), device=device)
        loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
        loss.backward()
        optimizer.step()
        #scheduler.step()
        total_loss += loss.item()
    
    print("Train Loss at epoch", epoch+1, "->", total_loss/len(train_dataloader))
    with open(os.path.join("logs", args.log_filename), "a") as f:
        f.write("Epoch " + str(epoch+1) + "\n")
        f.write("\tTraining loss: " + str(total_loss/len(train_dataloader)) + "\n")
    total_loss = 0

    if not args.use_all_data:
        # VALIDATION 
        model.eval()    
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader)):
                images,texts = batch 
                images = images["pixel_values"].squeeze(1).to(device)
                for k in texts.keys():
                    texts[k] = texts[k].to(device)
                outputs = model(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], pixel_values=images)
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

                loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
                total_loss += loss.item()
        
        if best_val_loss == None or total_loss < best_val_loss:
            print("BEST model found")
            torch.save(model, os.path.join(output_path_root, "clip_" + args.model_size + "_mono_" + args.textual_input + "_" + str(args.textual_augmentation) + str(args.visual_augmentation) + ".pt"))
            best_val_loss = total_loss
            best_epoch = epoch

        print("Validation", epoch+1, "->", total_loss/len(val_dataloader))
        with open(os.path.join("logs", args.log_filename), "a") as f:
            f.write("\tValidation loss: " + str(total_loss/len(val_dataloader)) + "\n")
        total_loss = 0

if not args.use_all_data:
    with open(os.path.join("logs", args.log_filename), "a") as f:
        f.write("Best model found at epoch " + str(best_epoch+1) + " with validation loss: " + str(best_val_loss/len(val_dataloader)) + "\n")
else:
    torch.save(model, os.path.join(output_path_root, "all_clip_" + args.model_size + "_mono_" + args.textual_input + "_" + str(args.textual_augmentation) + str(args.visual_augmentation) + ".pt"))