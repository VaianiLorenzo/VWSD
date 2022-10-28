from cgitb import text
from re import A
from PIL import Image
Image.MAX_IMAGE_PIXELS = 631770000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import os
import glob
import pandas as pd
import scipy.stats as ss

import torch
import torch.utils.data as data
import torch.nn as nn
from torch import optim

from transformers import CLIPProcessor, CLIPModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"

bs = 16
epochs = 50

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
train_data_path = os.path.join(input_folder_path, "train_data_v1.txt")
train_label_path = os.path.join(input_folder_path, "train_label_v1.txt")
val_data_path = os.path.join(input_folder_path, "val_data_v1.txt")
val_label_path = os.path.join(input_folder_path, "val_label_v1.txt")
images_path = os.path.join(input_folder_path, "train_images_v1")
output_path_root = "checkpoints"

train_df = pd.read_csv(train_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
with open(train_label_path, "r") as f:
    train_labels = f.readlines()
train_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in train_labels]

val_df = pd.read_csv(val_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
with open(val_label_path, "r") as f:
    val_labels = f.readlines()
val_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in val_labels]

with open("single_clip_training_log", "w") as f:
    f.write("TRAINING LOG")


# print("Start")
# tokenized_sentences = processor(text=list(df["full_phrase"]), return_tensors="pt", padding=True)
# print(len(tokenized_sentences))
# print(len(tokenized_sentences["input_ids"]))
# print("End")
# images = dict()
# preprocessed_images = [processor(images=Image.open(os.path.join(train_images_path, image_name)), return_tensors="pt", padding=True) for image_name in tqdm(os.listdir(train_images_path))]
# print(len(preprocessed_images))

class CLIP_dataset(data.Dataset):
    def __init__(self, list_image_path, list_txt, processor):
        self.processor = processor
        self.image_path = list_image_path
        self.processed_sentences  = processor(text=list_txt, return_tensors="pt", padding=True)
        
    def __len__(self):
        return len(self.processed_sentences["input_ids"])

    def __getitem__(self, idx):
        image = self.processor(images=[Image.open(self.image_path[idx])], return_tensors="pt")
        return image, {"input_ids":self.processed_sentences["input_ids"][idx], "attention_mask":self.processed_sentences["attention_mask"][idx]}


train_dataset = CLIP_dataset(train_gt_image_paths, list(train_df["full_phrase"]), processor)
train_dataloader = data.DataLoader(train_dataset, batch_size=bs, shuffle = True, num_workers=16)
val_dataset = CLIP_dataset(val_gt_image_paths, list(val_df["full_phrase"]), processor)
val_dataloader = data.DataLoader(val_dataset, batch_size=bs, num_workers=16)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

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
        #ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        ground_truth = torch.arange(len(images), device=device)
        loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print("Train Loss at epoch", epoch+1, "->", total_loss/len(train_dataloader))
    with open("single_clip_training_log", "a") as f:
        f.write("Epoch " + str(epoch+1) + " - training loss: " + str(total_loss))
    total_loss = 0

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
            total_loss += loss

    
    if best_val_loss == None or total_loss < best_val_loss:
        print("BEST model found")
        torch.save(model, os.path.join(output_path_root, "single_clip_finetuned.model"))
        best_val_loss = total_loss
        best_epoch = epoch

    print("Validation", epoch+1, "->", total_loss/len(val_dataloader))
    with open("single_clip_training_log", "a") as f:
        f.write("Epoch " + str(epoch+1) + " - validation loss: " + str(total_loss))
    total_loss = 0

with open("single_clip_training_log", "a") as f:
    f.write("Best model found at epoch " + str(best_epoch+1) + " with validation loss: " + str(best_val_loss))