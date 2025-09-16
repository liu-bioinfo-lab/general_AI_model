# EPCOTv2 — General AI model in genomics domain

**EPCOTv2** is a general, multi-task genomic model that integrates **DNA sequence** and **ATAC-seq** to jointly predict a broad set of genomic modalities (transcription, epigenome, TF binding, and 3D chromatin). It produces multi‑modal outputs aligned to the same genome coordinates and is designed to generalize across diverse cell types and tissues.

<p align="center">
  <img src="Assets/overview.png" alt="EPCOT v2 graphical abstract" width="560">
</p>


---

## Quick links

- 📄 **Paper (bioRxiv):** [Developing a general AI model for integrating diverse genomic modalities and comprehensive genomic knowledge](https://www.biorxiv.org/content/10.1101/2025.05.08.652986v1)  
- 📖 **Documentation & Tutorial:** [epcotv2-tutorial.readthedocs.io](https://epcotv2-tutorial.readthedocs.io)  
- 🌐 **Interactive Web Portal (Run predictions without installation):** [Hugging Face Space](https://huggingface.co/spaces/luosanj/EPCOTv2)  
- 🧪 **Basic notebook (Colab):**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liu-bioinfo-lab/general_AI_model/blob/main/epcotv2_basic_tutorial.ipynb)  
- 🧰 **Source code & data processing:** [`src/`](src)

---

## Installation

For complete instructions, see the Install guide:  
https://epcotv2-tutorial.readthedocs.io/en/latest/install/index.html

Minimal local setup:

```bash
# clone
git clone https://github.com/liu-bioinfo-lab/general_AI_model.git
cd general_AI_model

# (optional) create a fresh environment
conda create -n epcotv2 python=3.9 -y
conda activate epcotv2

# install dependencies
pip install -r requirements.txt
```

---

## Getting started

You can run the tutorial directly in **Google Colab** (no installation needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liu-bioinfo-lab/general_AI_model/blob/main/epcotv2_basic_tutorial.ipynb)

The notebook walks through:
- preparing inputs (DNA + ATAC-seq),
- running predictions with EPCOTv2

---

## Try it in your browser (no code)

Use the Hugging Face Space to run EPCOTv2 without installing anything:  
https://huggingface.co/spaces/luosanj/EPCOTv2

---

## Pretrained models

Pretrained EPCOTv2 checkpoints are hosted on Hugging Face:

👉 [Download from Hugging Face](https://huggingface.co/spaces/luosanj/EPCOTv2/tree/main/models)

These checkpoints are required to run the tutorials and reproduce results.

---



## Citation

If you use EPCOT v2 in your work, please cite:

```latex
@article{zhang2025developing,
  title={Developing a general AI model for integrating diverse genomic modalities and comprehensive genomic knowledge},
  author={Zhang, Zhenhao and Bao, Xinyu and Jiang, Linghua and Luo, Xin and Wang, Yichun and Comai, Annelise and Waldhaus, Joerg and Hansen, Anders S and Li, Wenbo and Liu, Jie},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```


## License 

This project is licensed under the MIT License.
