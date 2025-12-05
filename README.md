# Task Word Explainability — Reproduction Study

This repository contains a reproduction of the paper:

**Agarwal et al., 2023 — “Representing Visual Classification as a Linear Combination of Words”  
Machine Learning for Healthcare (ML4H)**

The original paper proposes a simple but interpretable approach for visual classification: instead of training a deep model end-to-end, the authors use CLIP text embeddings as a linear basis for prediction, producing “word weights” that indicate which words help explain a classification decision.

This reproduction focuses on:

- Re-implementing the core word-weight linear model  
- Training the method on the SIIM-ISIC Melanoma dataset  
- Evaluating accuracy and AUROC  
- Visualizing learned word-importance scores  
- Comparing results against the original paper’s reported performance  

---

## Installation

Clone the repository:

    git clone https://github.com/andyroo123/task_word_explainability_reproduction
    cd task_word_explainability_reproduction

Install dependencies:

    pip install -r requirements.txt

Install CLIP:

    pip install git+https://github.com/openai/CLIP.git

---

## Dataset Setup (SIIM-ISIC)

Download the SIIM-ISIC 2020 Kaggle dataset and place it in the following structure:

    data/
      train/
        image1.png
        image2.png
        ...

The scripts assume a standard train folder and a CSV file with metadata and melanoma labels.

---

## Running the Model

### 1. Configure CLIP

Run:

    python configure_clip.py

This verifies CLIP installation and prepares embedding utilities.

---

### 2. Train the Word-Weight Model

Run:

    python fit_words.py --dataset siim-isic --save_dir results/melanoma/

This script:

- Loads CLIP image embeddings  
- Builds a vocabulary of candidate words from the CLIP tokenizer  
- Solves a regularized linear regression (L1) connecting word embeddings to labels  
- Saves model weights, scores, and outputs  

Results include AUROC, accuracy, prediction scores, and the full word-weight vector.

---

### 3. Plot Word Weights

Run:

    python plotting.py results/melanoma/weights.npy

This generates:

- Sorted word weight plots  
- Positive vs. negative contributing words  
- Visual summaries of interpretability patterns  

Plots are saved to:

    results/melanoma/


---

## Notes

- Experiments use CLIP ViT-B/32 by default.  
- Word vocabulary originates from CLIP’s tokenizer.  
- L1 regularization encourages sparse, interpretable word weight vectors.  
- Regression is implemented using scikit-learn’s Lasso or an equivalent solver.

---

## Citation

If you use this reproduction, please cite the original work:

Agarwal, A., et al. “Representing Visual Classification as a Linear Combination of Words.” ML4H 2023.

---

## Contact

For questions or suggestions, open an issue on the repository.
