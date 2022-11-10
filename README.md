# PoliTo Team @ SemEval 2023 Task 1: Visual-Word Sense Disambiguation
This repository contains the code to setup the experiments for the SemEval 2023 Visual-Word Sense Disambiguation task.

## Finetuning
To finetune the the standard CLIP large pretraing from Hugging Face [(here)](https://huggingface.co/openai/clip-vit-large-patch14), you can run:
```shell
$ python clip_finetuning.py \
  --textual_input full_phrase \
  --log_filename clip_training.txt \
  --epochs 30 \
  --batch_size 16
```

## Inference
If you want to test the standard CLIP large pretraing from Hugging Face [(here)](https://huggingface.co/openai/clip-vit-large-patch14) you can run:
```shell
$ python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200
```
Otherwise, if you alredy finetuned the model, you can load and test your checkpoint, stored in the ```checkpoints``` folder, running the following:
```shell
$ python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --clip_finetuned_model_name clip_finetuned.model \
  --textual_augmentation \
  --visual_augmentation
```
