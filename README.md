# Patient voice Miner

This repository contains the official implementation of **Patient voice Miner**, a system proposed in our paper for structured information extraction from electronic patient-provider communication (EPPC) messages. The method combines a domain-specific BERT model (`PPCBERT`) with topic-enhanced inputs from a pre-trained `PPCBERTopic` module to improve multi-label classification of communication behaviors.

---

## Paper

PVminer: A Domain-Specific Tool to Detect the Patient Voice in Patient Generated Data

https://arxiv.org/abs/2602.21165

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

If you use this code or models in your work, please cite our paper:

@misc{fodeh2026pvminerdomainspecifictooldetect,
      title={PVminer: A Domain-Specific Tool to Detect the Patient Voice in Patient Generated Data}, 
      author={Samah Fodeh and Linhai Ma and Yan Wang and Srivani Talakokkul and Ganesh Puthiaraju and Afshan Khan and Ashley Hagaman and Sarah Lowe and Aimee Roundtree},
      year={2026},
      eprint={2602.21165},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.21165}, 
}



