# MIRROR

This document explains the process of building the MIRROR dataset.

## **Step 1: Multimodal Dialogue Design**
In this step, we process facial image data to annotate age, emotion, and gender using DeepFace.

**Run Facial Annotation**

The following command will analyze images and generate annotations.

```bash
python -m mirror.step1.face_annot --data_path ../data/celeba.csv --img_data_dir /data/celeba/img_align_celeba --save_path ../data/proc_celeba.csv --drop_duplicated
```
- `--data_path`: Path to the dataset file (CSV format).
- `--img_data_dir`: Directory containing facial images.
- `--save_path`: Path where the processed dataset will be saved.
- `--drop_duplicated`: Removes duplicate entries.

-----


## **Step 2: Counseling Screenplay Generation**

This step generates conversation scripts based on the Cactus dataset.

**Generate Prompts with GPT**

If `--fetch_api` is used, the script will call the GPT API. Otherwise, it will generate prompts in batch mode and save them in `save_dir`.

```bash
python -m mirror.step2.run --model gpt-4o-mini --prompt_ver session_v3 --data_path ../data/cactus_data.csv --save_dir ../data/prompts/
```
- `--model`: The language model to use (e.g., gpt-4o-mini).
- `--prompt_ver`: Version of the prompt template.
- `--data_path`: Path to the Cactus dataset (CSV format).
- `--save_dir`: Directory where prompts will be saved.

**Postprocessing the Generated Scripts**

This step processes the generated prompts and converts them into a structured dataset.

```bash
python -m mirror.step2.postprocess --batch_input_dir ../data/prompts/ --batch_output_dir ../data/batch_outputs/ --save_path ../data/mirror_data.csv
```

- `--batch_input_dir`: Directory containing generated prompts.
- `--batch_output_dir`: Directory where processed outputs will be saved.
- `--save_path`: Final processed dataset file.

-----

## **Step 3: Facial Expression Synthesis**

This step generates descriptions of facial expressions and uses them to create synthetic images.

**(1) Generate Facial Expression Descriptions Using LLM**

The first step is to create textual descriptions of facial expressions based on the `mirror_data.csv` dataset.

```bash
python -m mirror.step3.preprocess_for_llm --data_path ../data/mirror_data.csv --model_name Meta-Llama-3-8B-Instruct --save_path ../data/llama3_8b_prompt.jsonl
```
- `--data_path`: Path to the processed `mirror_data.csv` file.
- `--model_name`: The LLM model used to generate descriptions.
- `--save_path`: Path to save the generated prompts.


Next, we use an LLM model to generate descriptions and contrasting facial expressions.

```bash
python -m mirror.step3.annotate_llm --prompt_path ../data/llama3_8b_prompt.jsonl --model_name_or_path /model/Meta-Llama-3-8B-Instruct --save_path ../data/llama3_8b_result.jsonl
```

- `--prompt_path`: JSONL file containing prompts.
- `--model_name_or_path`: Path to the LLM model used for annotation.
- `--save_path`: Path where the generated results will be saved.


**(2) Image Generation Using PhotoMaker**  
Using the generated facial expression descriptions, we create prompts for image generation.

```bash
python -m mirror.step3.preprocess_for_photomaker --llm_result_path ../data/llama3_8b_result.jsonl --celeba_path ../data/proc_celeba.csv  --save_path photomaker_prompts/prompt.jsonl
```
- `--prompt_path`: Path to the LLM-generated facial expression descriptions.
- `--model_name_or_path`: Path to the processed CelebA dataset.
- `--save_path`: Path to save the PhotoMaker prompts.


Finally, we generate facial expression images using PhotoMaker.

```bash
python -m mirror.step3.run --prompt_path photomaker_prompts/prompt.jsonl --save_dir ../data/images/ 
```
- `--prompt_path`: Path to the prompt JSON file.
- `--save_dir`: Directory where generated images will be saved.

-----

## **Step 4: Filtering for Quality and Safety**

**Dialogue Safety Filtering**  
```bash
python -m mirror.step4.safety --canary_dir ./step4/data/models/canary --data_path ../data/mirror_data.csv
```

**CLIP-based Image-Text Similarity Filtering**  
```bash
python -m mirror.step4.clip --data_path ../data/mirror_data.csv --image_dir /data/images/
```
- `--data_path`: Path to the processed `mirror_data.csv` file.
- `--image_dir`: Directory where generated images are saved.


**Annotate Attribution for Gender Preservation Filtering**  
```bash
python -m mirror.step4.attr --image_dir /data/images/
```

**NSFW Filtering**  
```bash
python -m mirror.step4.nsfw --image_dir /data/images/
```

**Identity Preservation Filtering**  
```bash
python -m mirror.step4.identity --data_path ../data/mirror_data_w_annot.csv --image_dir /data/images/
```

- `--data_path`: Path to `mirror_data_w_annot.csv`, which maps the original CelebA images with `mirror_data.csv`.




