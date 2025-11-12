# MIRROR-LLaVA: Fine-tuning & Inference Instructions

Our training code is based on [LLaVA](https://github.com/haotian-liu/LLaVA), while inference includes image generation using [PhotoMaker](https://github.com/TencentARC/PhotoMaker). To run inference successfully, you must set up both the LLaVA and PhotoMaker environments.


## **Training**

The following script is used for fine-tuning the MIRROR model with different configurations. The script takes the following parameters:

- **Epochs**: Number of training epochs
- **Model Size**: Size of the model (e.g. 7b or 13b)
- **Type of CoT**: Choose from `base`, `planning`, or `ec_planning`

Execute the appropriate command based on the desired training setup:

**MIRROR-LLAVA Model Training**  
```bash
bash scripts/v1_5/finetune_mirror_lora.sh 5 7b base
```
**MIRROR-LLAVA_P Model Training**  
```bash
bash scripts/v1_5/finetune_mirror_lora.sh 5 7b planning
```

**MIRROR-LLAVA_{P+EC} Model Training**
```bash
bash scripts/v1_5/finetune_mirror_lora.sh 5 7b ec_planning
```
---

## **Inference**

During inference, the system generates a facial image of the client at each turn and applies an image filtering process (with a maximum of 10 attempts).

To achieve this, we utilize an ***LLM-based facial description generation*** and an ***identity preservation filtering*** process via APIs. You must run the applications located in:

- `../api/verification`
- `../api/description`

### **API Configuration**
Before executing the inference script, you need to configure the API URL and set your OpenAI API key:  

```bash
export OPENAI_API_KEY="your-openai-api-key"
export IMG_VERIFICATION_URL="your-verification-api-url"
export IMG_DESCRIPTION_URL="your-description-api-url"
```

You can run inference using the following evaluation scripts.

### **LLaVA Counseling**
```bash
bash scripts/v1_5/eval/llava_counseling.sh 7b
```

### **MIRROR-LLAVA Counseling**
```bash
bash scripts/v1_5/eval/mirror_counseling.sh 5 7b planning
```
- **Epochs**: Number of training epochs
- **Model Size**: Size of the model (e.g. 7b or 13b)
- **Type of CoT**: Choose from `base`, `planning`, or `ec_planning`

> **Note**:  
The trained model must be located in the following path:  
./checkpoints/llava-v1.5-`<model size>`-mirror_`<type of cot>`-task-lora-epoch`<epoch>`
