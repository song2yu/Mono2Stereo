# Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion
<div align="center">
 <img src="assets/imgs/logo.png" alt="logo" width="200px"> 
 <br>
 <a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
 <a href='https://mono2stereo-bench.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
</div>
<div align="center">
<a href="https://song2yu.github.io/"><sup>1</sup>Songsong Yu</a> |
<a href="https://scholar.google.com/citations?hl=zh-CN&user=dEm4OKAAAAAJ&view_op=list_works"><sup>2</sup> Yuxin Chen</a> |
<a href="https://scholar.google.com/citations?user=ysXmZCMAAAAJ&hl=zh-CN&oi=ao"><sup>3</sup>Zeke Xie</a> |
<a href="https://scholar.google.com/citations?user=j1XFhSoAAAAJ&hl=zh-CN&oi=ao"><sup>1</sup>Yifan Wang</a> |
<a href="https://scholar.google.com/citations?user=EfTwkXMolscC&hl=zh-CN&oi=ao"><sup>1</sup>Lijun Wang</a> |
<a href="https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=zh-CN&oi=ao"><sup>2</sup>Zhongang Qi</a> |
<a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN&oi=ao"><sup>2</sup>Ying Shan</a> |
<a href="https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN&oi=ao"><sup>1</sup>Huchuan Lu</a>
<br>
<sup>1</sup>Dalian University of Technology 
<sup>2</sup>ARC Lab, Tencent PCG
<sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou)

CVPR 2025

</div>
With the rapid proliferation of 3D devices and the shortage of 3D content, stereo conversion is attracting increasing attention. Recent works have introduced pretrained Diffusion Models (DMs) into this task. However, due to the scarcity of large-scale training data and comprehensive benchmarks, the optimal methodologies for employing DMs in stereo conversion and the accurate evaluation of stereo effects remain largely unexplored.
<br>
To address these challenges, we introduce the Mono2Stereo dataset, which provides high-quality training data and benchmarks to support in-depth exploration of stereo conversion. Through empirical studies using this dataset, we have identified two primary findings:
<br>
1. The differences between the left and right views are subtle, yet existing metrics consider overall pixels, failing to focus on regions critical to stereo effects.
<br>
2. Mainstream methods adopt either a one-stage left-to-right generation approach or a warp-and-inpaint pipeline, facing challenges of degraded stereo effect and image distortion, respectively.
<br>
Based on these findings, we introduce a new evaluation metric, Stereo Intersection-over-Union (Stereo IoU), which prioritizes disparity and achieves a high correlation with human judgments on stereo effect. Additionally, we propose a strong baseline model that harmonizes stereo effect and image quality simultaneously.

 <img src="assets/imgs/teaser.png" alt="teaser" width="200px"> 


## 📢 News
2025-03-16: [Project page](https://mono2stereo-bench.github.io/) and inference code (this repository) are released.<br>
2025-02-27: Accepted to CVPR 2025. <br>



## 🛠️ Setup

The inference code was tested on:

- Python 3.8.20,  CUDA 12.1

## 📦 Usage
**Installation**

Clone the repository (requires git):

```bash
git clone https://github.com/song2yu/Mono2Stereo.git
cd mono2stereo
```

create a Python native virtual environment and install dependencies into it:

```bash
conda create -n stereo python=3.8 -y
conda activate stereo
pip install -r requirements.txt
```

**Inference**
```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```


