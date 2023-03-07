from tqdm import tqdm
import os
import pandas as pd
import nlpaug.augmenter.word as naw
import argparse

parser = argparse.ArgumentParser(description='Back translation augmentation')
parser.add_argument('--language',
    help='Language to translate to',
    default="de",
    required=False,
    type=str)
args = parser.parse_args()

language = args.language

input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
train_data_path = os.path.join(input_folder_path, "train_data_v1.txt")
val_data_path = os.path.join(input_folder_path, "val_data_v1.txt")

train_df = pd.read_csv(train_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
val_df = pd.read_csv(val_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])

if language == "fa":
    from_model_name = "persiannlp/mt5-large-parsinlu-translation_en_fa"
    to_model_name = "persiannlp/mt5-large-parsinlu-opus-translation_fa_en"
else:
    from_model_name = "Helsinki-NLP/opus-mt-en-"+language
    to_model_name = "Helsinki-NLP/opus-mt-"+language+"-en"

back_translation_aug = naw.BackTranslationAug(
                from_model_name=from_model_name, 
                to_model_name=to_model_name,
                device='cuda',
                max_length=16
            )

with open(os.path.join(input_folder_path, "train_back_translation_aug_"+language+".txt"), "w") as f:
    for text in tqdm(list(train_df["full_phrase"])):
        new_text = back_translation_aug.augment(text)
        f.write(new_text[0].replace(".", "").lower() + "\n")

with open(os.path.join(input_folder_path, "val_back_translation_aug_"+language+".txt"), "w") as f:
    for text in tqdm(list(val_df["full_phrase"])):
        new_text = back_translation_aug.augment(text)
        f.write(new_text[0].replace(".", "").lower() + "\n")

# load files
languages = ["it", "fa", "fr"]
for language in languages:
    with open(os.path.join(input_folder_path, "train_back_translation_aug_"+language+".txt"), "r") as f:
        lines = f.readlines()
    with open(os.path.join(input_folder_path, "train_back_translation_aug_"+language+".txt"), "w") as f:
        for line in lines:
            f.write(line.replace(".", "").lower())

    with open(os.path.join(input_folder_path, "val_back_translation_aug_"+language+".txt"), "r") as f:
        lines = f.readlines()
    with open(os.path.join(input_folder_path, "val_back_translation_aug_"+language+".txt"), "w") as f:
        for line in lines:
            f.write(line.replace(".", "").lower())



            