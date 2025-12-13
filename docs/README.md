# MSIS822 Project

# Arabic AI-Generated Text Detection Using Stylometric Features

## Project Overview
This project investigates the detection of AI-generated Arabic academic abstracts using **stylometric features** and **traditional machine learning models**.  
Instead of relying on large neural language models, the study focuses on **interpretable linguistic indicators** such as vocabulary richness and entity usage patterns.

The main objective is to distinguish between **human-written** and **AI-generated** Arabic texts while maintaining transparency, efficiency, and reproducibility.

---

## Dataset Description
- **Language:** Arabic  
- **Text Type:** Academic abstracts  
- **Labels:**
  - `0` → Human-written
  - `1` → AI-generated  
- **Class Distribution:** Imbalanced (AI-generated texts dominate the dataset)

## Stylometric Features
Four interpretable stylometric features were extracted from each text:

1. **Brunet’s W** – Measures vocabulary richness  
2. **Proper Nouns Count** – Approximation of named entity usage  
3. **Singular Words Count** – Frequency of singular morphological forms  
4. **Entity Diversity** – Ratio of unique entities to total entities  

These features aim to capture **lexical, morphological, and structural writing patterns**.

## Modeling Approach

### Baseline Model
- **Logistic Regression**  
  A linear classifier used to establish a baseline performance.

### Traditional Machine Learning Models
- **Random Forest Classifier**
- **Linear Support Vector Machine (Linear SVM)**

### Neural Network Model
- **Multi-Layer Perceptron (MLP)** trained on the extracted stylometric features.

---

## Training and Evaluation
- Data split into **training, validation, and test sets** using stratified sampling.
- Models trained using **only the four stylometric features**.
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrices

---

## Key Results
- **Random Forest** achieved the highest accuracy and best balance between classes.
- Logistic Regression and Linear SVM showed bias toward the AI class.
- Stylometric features were sufficient for strong classification when combined with ensemble methods.
- Neural models did not outperform Random Forest in this setting.

---

## Feature Importance Analysis
Two methods were used for interpretability:

### 1. Gini Importance (Random Forest)
- Highlights features most frequently used for decision splits.

### 2. Permutation Importance
- Measures performance drop when a feature is randomly shuffled.

Both analyses identified **Brunet’s W** and **entity-related features** as the most influential predictors.

---

## How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt 
```
---
# Execute the Notebooks in Order

Phase1 → Phase2 → Phase3 → Phase4 → Phase5

---

# Conclusion

This project demonstrates that simple stylometric features, when combined with traditional machine learning models, can effectively distinguish between human-written and AI-generated Arabic texts.
The results emphasize the value of linguistic insight and model interpretability in AI-text detection.

---
 ## Author
Mohammed Abuselmi  
Master’s Student in Big Data Analytics  
Department of Information Systems  
College of Computer Science and Engineering  
Taibah University  
Course: MSIS822 – Data Analytics Techniques
