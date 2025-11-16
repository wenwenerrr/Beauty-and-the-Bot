# 👸Beauty-and-the-Bot🤖
In the increasingly competitive e-commerce industry, customer satisfaction and personalized engagement are critical to brand loyalty. With a wide and diverse customer base, handling large volumes of product inquiries and after-sales questions efficiently has become increasingly important. To address this need, our project develops an intelligent customer service chatbot designed specifically for the Beauty category.

The dataset we have chosen are the **Amazon Reviews'23** [https://amazon-reviews-2023.github.io] and **Amazon Q&A** [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/]. The Amazon Reviews'23 consists of two parts: the structured meta_All_Beauty dataset, containing product metadata fields such as *brand, price and category*; the unstructured All_Beauty dataset contains product review information like *user id, rating, review title and content*. Both datasets joined by *parent_asin*.

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
├── eda.ipynb
├── ???
└── ??? 
```

## Replication Steps
