from PIL import Image
Image.MAX_IMAGE_PIXELS = 631770000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import os
import pandas as pd
import scipy.stats as ss
import argparse

import torch

from transformers import CLIPProcessor, CLIPModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description="CLIP inference")

parser.add_argument(
    "--clip_finetuned_model_name",
    help="Name of the finetuned CLIP model",
    default=None,
    required=False)
parser.add_argument(
    "--log_filename",
    help="Name of the log file",
    default="log.txt",
    required=False)
parser.add_argument(
    "--log_step",
    help="Number of steps after which to log",
    type=int,
    default=200,
    required=False)
parser.add_argument(
    "--phase",
    help="Phase of the inference",
    default="val",
    choices=["test", "val", "trial"],
    required=False)
parser.add_argument(
    "--model_size",
    help="Size of the CLIP model",
    default="large",
    choices=["large", "base"],
    required=False)
parser.add_argument(
    "--test_submission_folder",
    help="Folder where to save the test submision",
    default="submissions/test_submissions",
    required=False)
args = parser.parse_args()

# Load the model
if args.clip_finetuned_model_name is None:
    if args.model_size == "large":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.model_size == "base":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
else:
    model = torch.load(os.path.join("checkpoints", args.clip_finetuned_model_name)).to(device)
    size = "large" if "large" in args.clip_finetuned_model_name else "base"
    if size == "large":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif size == "base":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load the data
if args.phase == "trial":
    input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "trial_v1")
    data_path = os.path.join(input_folder_path, "trial.data.v1.txt")
    label_path = os.path.join(input_folder_path, "trial.gold.v1.txt")
    images_path = os.path.join(input_folder_path, "trial_images_v1")
elif args.phase == "val":
    input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
    data_path = os.path.join(input_folder_path, "val_data_v1.txt")
    label_path = os.path.join(input_folder_path, "val_label_v1.txt")
    images_path = os.path.join(input_folder_path, "train_images_v1")
elif args.phase == "test":
    input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "test")
    en_data_path = os.path.join(input_folder_path, "en.test.data.txt")
    it_data_path = os.path.join(input_folder_path, "it.test.data.translated.txt")
    fa_data_path = os.path.join(input_folder_path, "fa.test.data.translated.txt")
    data_paths = [en_data_path, it_data_path, fa_data_path]
    languages = ["en", "it", "fa"]
    images_path = os.path.join(input_folder_path, "test_images_resized")

if args.phase == "trial" or args.phase == "val":
    df = pd.read_csv(data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])

    with open(label_path, "r") as f:
        labels = f.readlines()

    # Create log files
    with open(os.path.join("logs", "error_log.txt"), "w") as f:
        f.write("ERROR LOG\n")
    with open(os.path.join("logs", args.log_filename), "w") as f:
        f.write("INFERENCE LOG\n")

    # Create submission files
    versions = ["FullSentence", "MainTopic", "AmbiguousWord", "FS+MT", "FS+AW", "MT+AW", "FS+MT+AW"]
    for version in versions:
        with open(os.path.join("submissions", args.phase+"_submission_"+version+".txt"), "w") as f:
            pass

    hit_rates = [0] * 7
    mrrs = [0] * 7
    most_frequent_ranks = [[0]*10 for i in range(7)] 

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                image_names = [row["image_"+str(i)] for i in range(10)]
                images = [Image.open(os.path.join(images_path, image_name)) for image_name in image_names]
                sentence = row["full_phrase"]
                ambiguous = row["target_word"]

                # remove the ambiguous part from the sentence
                ambiguous_index = sentence.find(ambiguous)
                main_topic = sentence[:ambiguous_index] + sentence[ambiguous_index+len(ambiguous):]
                main_topic = main_topic.strip()
                
                inputs = processor(text=[sentence, main_topic, ambiguous], images=images, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                logits = outputs.logits_per_text # this is the image-text similarity score

                
                # compute all pairs and triplet of ensambles
                logits = torch.cat((
                    logits,
                    torch.mean(torch.cat((logits[0].unsqueeze(0), logits[1].unsqueeze(0)), dim=0), dim=0).unsqueeze(0),
                    torch.mean(torch.cat((logits[0].unsqueeze(0), logits[2].unsqueeze(0)), dim=0), dim=0).unsqueeze(0),
                    torch.mean(torch.cat((logits[1].unsqueeze(0), logits[2].unsqueeze(0)), dim=0), dim=0).unsqueeze(0),
                    torch.mean(logits, dim=0).unsqueeze(0)
                ), dim=0)
                
                ranks = []
                for j, text_logits in enumerate(logits):
                    rank = (len(text_logits) + 1) - ss.rankdata(text_logits.detach().cpu())
                    image_names_ordered = [image_names[i] for i in rank.argsort()]
                    with open(os.path.join("submissions", args.phase+"_submission_"+versions[j]+".txt"), "a") as f:
                        f.write("\t".join(image_names_ordered)+"\n")
                    ranks.append(rank)

                    if labels[index][:-1] == image_names[torch.argmax(text_logits).item()]:
                        hit_rates[j] += 1
                    mrrs[j] += 1/ranks[j][image_names.index(labels[index][:-1])]
                    most_frequent_ranks[j][int(ranks[j][image_names.index(labels[index][:-1])])-1] += 1

                if (index+1)%args.log_step == 0:
                    print("STEP", index+1)
                    with open(os.path.join("logs", args.log_filename), "a+") as f:
                        f.write("STEP " + str(index+1) + "\n")
                        for version, hit_rate, mrr, mfr in zip(versions, hit_rates, mrrs, most_frequent_ranks):
                            print("VERSION ->", version)
                            print("\tHIT RATE:", hit_rate/(index+1))
                            print("\tMRR:", mrr/(index+1))
                            print("\tMOST FREQUENT RANK:", mfr)

                            f.write("VERSION -> " + version + "\n")
                            f.write("\tHIT RATE: " + str(hit_rate/(index+1)) + "\n")
                            f.write("\tMRR: " + str(mrr/(index+1)) + "\n")
                            f.write("\tMOST FREQUENT RANK: " + str(mfr) + "\n")

            except OSError as e:
                print(e)
                print("ERROR LOG")
                print("\t", index)
                print("\t",sentence)
                print("\t",image_names)

                with open(os.path.join("logs", "error_log.txt"), "a+") as f:
                    f.write("\tError at index: " + str(index) + "\n")
                    f.write("\tSentence: " + sentence + "\n")
                    f.write("\tImages: " + str(image_names) + "\n\n")


    print("FINAL RESULTS")
    with open(os.path.join("logs", args.log_filename), "a+") as f:
        f.write("\nFINAL RESULTS\n")
        for version, hit_rate, mrr, mfr in zip(versions, hit_rates, mrrs, most_frequent_ranks):
            print("VERSION ->", version)
            print("\tHIT RATE:", hit_rate/(index+1))
            print("\tMRR:", mrr/(index+1))
            print("\tMOST FREQUENT RANK:", mfr)

            f.write("VERSION -> " + version + "\n")
            f.write("\tHIT RATE: " + str(hit_rate/(index+1)) + "\n")
            f.write("\tMRR: " + str(mrr/(index+1)) + "\n")
            f.write("\tMOST FREQUENT RANK: " + str(mfr) + "\n")

elif args.phase == "test":

    if not os.path.exists(args.test_submission_folder):
        os.makedirs(args.test_submission_folder)

    for language in languages:
        with open(os.path.join(args.test_submission_folder, "prediction."+language+".txt"), "w") as f:
                        f.write("")

    for j, data_path in enumerate(data_paths):
        df = pd.read_csv(data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])

        with torch.no_grad():
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                try:
                    image_names = [row["image_"+str(i)] for i in range(10)]
                    images = [Image.open(os.path.join(images_path, image_name)) for image_name in image_names]
                    sentence = row["full_phrase"]
                    inputs = processor(text=[sentence], images=images, return_tensors="pt", padding=True).to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits_per_text[0] # this is the image-text similarity score
                    
                    rank = (len(logits) + 1) - ss.rankdata(logits.detach().cpu())
                    image_names_ordered = [image_names[i] for i in rank.argsort()]
                    with open(os.path.join(args.test_submission_folder, "prediction."+languages[j]+".txt"), "a") as f:
                        f.write("\t".join(image_names_ordered)+"\n")

                except OSError as e:
                    print(e)
                    print("ERROR LOG")
                    print("\t", index)
                    print("\t",sentence)
                    print("\t",image_names)

                    with open(os.path.join("logs", "error_log.txt"), "a+") as f:
                        f.write("\tError at index: " + str(index) + "\n")
                        f.write("\tSentence: " + sentence + "\n")
                        f.write("\tImages: " + str(image_names) + "\n\n")




