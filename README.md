# 👸Beauty-and-the-Bot🤖
In the increasingly competitive e-commerce industry, customer satisfaction and personalized engagement are critical to brand loyalty. With a wide and diverse customer base, handling large volumes of product inquiries and after-sales questions efficiently has become increasingly important. To address this need, our project develops an intelligent customer service chatbot designed specifically for the Beauty category.

The dataset we have chosen are the **Amazon Reviews'23** [https://amazon-reviews-2023.github.io] and **Amazon Q&A** [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/]. The Amazon Reviews'23 consists of two parts: the structured meta_All_Beauty dataset, containing product metadata fields such as *brand, price and category*; the unstructured All_Beauty dataset contains product review information like *user id, rating, review title and content*. Both datasets joined by *parent_asin*. The cleaned and processed dataset used for this project can be found in [https://huggingface.co/datasets/xldzha/beauty_and_the_bot_dataset_used/tree/main]

Our approach combines a two-stage **knowledge-graph-augmented multimodal recommendation system**, and deploy **QLoRA Fine-tuning on LLAMA model**. Through this, we aim to create a chatbot that is fast, reliable, and aligned with industrial Beauty’s customer service standards.

## Repo Structure

```
Beauty-and-the-Bot/
├── main.py # Main script to run the chatbot
├── requirements.txt # Required Python packages
├── dataset
│ ├── All_Beauty.jsonl
│ ├── meta_All_Beauty.jsonl
│ └── qa_Beauty.json
├── Amazon_eda.ipynb
├── Resources
│ ├── Recommendation notebook
│ │ ├──Plotting.ipynb
│ │ ├──Recommender_system_experiments.ipynb
│ │ ├──pseudodata.ipynb
│ │ ├──summarizing_reviews.ipynb
│ ├── Chatbot notebook
│ │ ├──Fine-tuning experiment1.ipynb
│ │ ├──Fine-tuning experiment2.ipynb
└── ??? 
```

## Replication Steps
### Recommender Notebooks

This folder contains the Jupyter notebooks used for the recommender-system part of **Beauty-and-the-Bot**.

All datasets required for the recommender experiments are hosted on Hugging Face:

https://huggingface.co/datasets/xldzha/beauty_and_the_bot_dataset_used/tree/main

Download the relevant files from this dataset and update / keep the paths in the notebooks so they point to your local copy (or your own mounted data directory).

- `pseudodata.ipynb` – generates the synthetic query–item training and evaluation data.
- `summarizing_reviews.ipynb` – creates abstractive summaries of product reviews.
- `Recommender_system_experiments.ipynb` – runs the knowledge-graph + PPR + reranking experiments.
- `Plotting.ipynb` – code to reproduce the plots used in the report.

To reproduce our results, run these notebooks from a GPU-enabled runtime (e.g. Colab or a local machine with CUDA GPU).

## Fine-tuned Language Model Usage
1. Set up Hugging Face token
```
import os
os.environ['HF_TOKEN'] = 'your_hugging_face_token_here
```
2. Import directly from Hugging Face Hub
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Dellalala1/LlaMA_3.2_1B_Fine-tuned"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
```
