# **MIRROR**

This repository will provide the official code and dataset for our paper:
[🪞 MIRROR: Multimodal Cognitive Reframing Therapy for Rolling with Resistance](https://aclanthology.org/2025.emnlp-main.751/)

We are in the process of preparing the release.  
The code and dataset will be made publicly available shortly.

### **🔧 Environment Setup**

After setting up the environment for the target VLM (Vision-Language Model),
please install the required packages for PhotoMaker by running:

```bash
pip install -r photomaker_requirements.txt
```

### **📁 Repository Structure**

```graphql
MIRROR/
├── api/                     # API for ensuring image generation quality in the mirror-llava virtual client
│   ├── description/         # Runs app.py during VLM inference for image description generation
│   └── verification/        # Runs app.py during VLM inference for image quality verification
│
├── mirror/                  # Dataset generation pipeline
│   ├── agent/
│   ├── notebook/
│   ├── src/
│   ├── step1/               # Step 1: Data collection and preprocessing
│   ├── step2/               # Step 2: Cognitive reframing generation
│   ├── step3/               # Step 3: Multimodal alignment and annotation
│   ├── step4/               # Step 4: Dataset refinement and final packaging
│   ├── utils/
│   └── README.md            # Detailed usage and data preparation guide
│
├── llm_therapist/           # Inference scripts for LLM-based therapist models
│   ├── src/
│   └── run_scripts/         # Execution scripts for inference tasks
│
├── mirror-llava/            # Training and inference framework based on LLaVA v1.5
│   ├── llava/
│   ├── playground/          # Contains evaluation and training datasets for LLaVA
│   ├── scripts/             # Execution scripts for training and inference
│   └── README.md            # Instructions for fine-tuning and evaluation
│
├── photomaker_requirements.txt   # Additional dependencies for the PhotoMaker environment
└── README.md                     # Project overview and setup instructions
```

### **📄 Citation**

If you find this work useful, please cite our paper:

```csharp
@inproceedings{kim-etal-2025-mirror,
    title = "{MIRROR}: Multimodal Cognitive Reframing Therapy for Rolling with Resistance",
    author = "Kim, Subin  and Kim, Hoonrae  and Lee, Jihyun  and Jeon, Yejin  and Lee, Gary",
    editor = "Christodoulopoulos, Christos  and Chakraborty, Tanmoy  and Rose, Carolyn  and Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.751/",
    doi = "10.18653/v1/2025.emnlp-main.751",
    pages = "14851--14880",
    ISBN = "979-8-89176-332-6",
    abstract = "Recent studies have explored the use of large language models (LLMs) in psychotherapy; however, text-based cognitive behavioral therapy (CBT) models often struggle with client resistance, which can weaken therapeutic alliance. To address this, we propose a multimodal approach that incorporates nonverbal cues, which allows the AI therapist to better align its responses with the client{'}s negative emotional state.Specifically, we introduce a new synthetic dataset, Mirror (Multimodal Interactive Rolling with Resistance), which is a novel synthetic dataset that pairs each client{'}s statements with corresponding facial images. Using this dataset, we train baseline vision language models (VLMs) so that they can analyze facial cues, infer emotions, and generate empathetic responses to effectively manage client resistance.These models are then evaluated in terms of both their counseling skills as a therapist, and the strength of therapeutic alliance in the presence of client resistance. Our results demonstrate that Mirror significantly enhances the AI therapist{'}s ability to handle resistance, which outperforms existing text-based CBT approaches.Human expert evaluations further confirm the effectiveness of our approach in managing client resistance and fostering therapeutic alliance."
}
```

### **📬 Contact**

For questions or collaborations, please contact the authors via the corresponding information provided in the paper.
