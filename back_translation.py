from tqdm import tqdm
import os
import pandas as pd

import torch
import nlpaug.augmenter.word as naw

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
train_data_path = os.path.join(input_folder_path, "train_data_v1.txt")
val_data_path = os.path.join(input_folder_path, "val_data_v1.txt")

train_df = pd.read_csv(train_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
val_df = pd.read_csv(val_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])

back_translation_aug = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de', 
                to_model_name='facebook/wmt19-de-en',
                device='cuda',
                max_length=50
            )

with open(os.path.join(input_folder_path, "train_back_translation_aug.txt"), "w") as f:
    for text in tqdm(list(train_df["full_phrase"])):
        new_text = back_translation_aug.augment(text)
        f.write(new_text[0] + "\n")

with open(os.path.join(input_folder_path, "val_back_translation_aug.txt"), "w") as f:
    for text in tqdm(list(val_df["full_phrase"])):
        new_text = back_translation_aug.augment(text)
        f.write(new_text[0] + "\n")

    


            