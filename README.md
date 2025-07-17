# GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models
![image](https://github.com/hzhxmu/GS-Bias/blob/main/docs/GS-Bias.png)

### News
- **2025.05.01**:🔥GS-Bias has been accepted to ICML 2025!  [[Paper]](https://arxiv.org/abs/2507.11969)

### Install

- Setup conda environment (recommended).

```
# Create a conda environment
conda create -y -n gs-bias python=3.9

# Activate the environment
conda activate gs-bias

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install the Visulizer

```
cd Visualizer
pip install -e .
```

- Install requirements

```
pip install -r requirements.txt
```

### Datasets

Please follow the instructions at docs/DATASETS.md to prepare all datasets.

### How to Run

We provide the running scripts in scripts/, which allow you to reproduce the results on the paper.

#### Domain Generalization

```
bash scripts/GSBias_Domain.sh
```

#### Cross-Datasets Generalization

```
bash scripts/GSBias_CrossDataset.sh
```

### Citation
If you find GS-Bias useful for your research, please cite using this BibTeX:
```
@misc{huang2025gsbiasglobalspatialbiaslearner,
      title={GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models}, 
      author={Zhaohong Huang and Yuxin Zhang and Jingjing Xie and Fei Chao and Rongrong Ji},
      year={2025},
      eprint={2507.11969},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.11969}, 
}
```




