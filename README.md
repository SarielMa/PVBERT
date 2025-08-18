# PPCInsight

This repository contains the official implementation of **PPCInsight**, a system proposed in our paper for structured information extraction from electronic patient-provider communication (EPPC) messages. The method combines a domain-specific BERT model (`PPCBERT`) with topic-enhanced inputs from a pre-trained `PPCBERTopic` module to improve multi-label classification of communication behaviors.

---

## 📄 Paper

> *[Insert paper title and link here]*  
> *Authors: [Your Name(s)]*

---

## 🔧 Setup Instructions

To reproduce the experiments from the paper, follow the steps below.

### 0. Environment Setup

Create the Conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate amia2025  # or your environment name
```

### 1. Download Pretrained Models

- **PPCBERT models**: Download all pre-trained PPCBERT checkpoints and place them in the `PPCBERTs/` folder.  
  📎 *Download links are listed in the folder’s README file.*

- **PPCBERTopic model**: Download the pre-trained BERTopic model and place it in the `PPCBERTopic/` folder.  
  📎 *Link included inside the folder.*

### 2. Data Preprocessing

- For **baseline fine-tuning**:
  ```bash
  Run: data_preprocessing_get_original.ipynb
  ```

- For **PPCInsight fine-tuning**:
  ```bash
  Run: data_preprocessing_for_PPCInsight.ipynb
  ```

### 3. Set Output Directory

Open `run.sh` and modify the following line to specify where to store the results:

```bash
export RESULT_PATH=/your/output/path
```

### 4. Run Fine-tuning and Evaluation

Launch the training and evaluation pipeline with:

```bash
bash run.sh
```

---

## Hardware Requirements

All experiments were conducted on a server with:
- 2 × NVIDIA A100 GPUs (each with 40 GB VRAM)

---

## Software Requirements

All dependencies are listed in [`environment.yml`](./environment.yml).

---

## Citation

If you use this code or models in your work, please cite our paper (citation info to be added upon publication).



