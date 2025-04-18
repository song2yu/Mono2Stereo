# Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion
<div align="center">
 <img src="assets/imgs/logo.png" alt="logo" width="200px"> 
 <br>
 <a href='https://arxiv.org/pdf/2503.22262'><img src='https://img.shields.io/badge/arXiv-2503.22262-b31b1b.svg'></a> &nbsp;
 <a href='https://mono2stereo-bench.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
</div>
<div align="center">
<a href="https://song2yu.github.io/"><sup>1</sup>Songsong Yu</a> |
<a href="https://scholar.google.com/citations?hl=zh-CN&user=dEm4OKAAAAAJ&view_op=list_works"><sup>2</sup> Yuxin Chen🌟</a> |
 <a href="https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=zh-CN&oi=ao"><sup>2</sup>Zhongang Qi✉️</a> |
<a href="https://scholar.google.com/citations?user=ysXmZCMAAAAJ&hl=zh-CN&oi=ao"><sup>3</sup>Zeke Xie</a> |
<a href="https://scholar.google.com/citations?user=j1XFhSoAAAAJ&hl=zh-CN&oi=ao"><sup>1</sup>Yifan Wang</a> |
<a href="https://scholar.google.com/citations?user=EfTwkXMolscC&hl=zh-CN&oi=ao"><sup>1</sup>Lijun Wang✉️</a> |
<a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN&oi=ao"><sup>2</sup>Ying Shan</a> |
<a href="https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN&oi=ao"><sup>1</sup>Huchuan Lu</a>
<br>
<sup>1</sup>Dalian University of Technology 
<sup>2</sup>ARC Lab, Tencent PCG
<sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou)
 <br>
CVPR 2025📖
Project Lead🌟 Corresponding Authors ✉️
</div>


<div align="left">
<br>
💡 With the rapid growth of 3D devices and a shortage of 3D content, stereo conversion is gaining attention. Recent studies have introduced pretrained Diffusion Models (DMs) for this task, but the lack of large-scale training data and comprehensive benchmarks has hindered optimal methodologies and accurate evaluation. To address these challenges, we introduce the Mono2Stereo dataset, providing high-quality training data and benchmarks. Our empirical studies reveal:
<br>
1. Existing metrics fail to focus on critical regions for stereo effects.
<br>
2.Mainstream methods face challenges in stereo effect degradation and image distortion.
<br>
We propose a new evaluation metric, Stereo Intersection-over-Union (Stereo IoU), which prioritizes disparity and correlates well with human judgments. Additionally, we introduce a strong baseline model that balances stereo effect and image quality.
</div>

<br>
<div align="center">
 <img src="assets/imgs/teaser.png" alt="teaser" width="1000px"> 
</div>

<br>

## 📢 News
2025-03-16: [Project page](https://mono2stereo-bench.github.io/) and inference code (this repository) are released.<br>
2025-02-27: Accepted to CVPR 2025. <br>


<br>

## 🛠️ Setup

The inference code was tested on:

- Python 3.8.20,  CUDA 12.1

<br>

## 📦 Usage
**Preparation**
<br>
You can download our model [weights](https://pan.baidu.com/s/12cG1_o9G8qwkQLKm7B6XNQ?pwd=phns) to perform inference.


<br>

**⚙️ Installation**

Clone the repository (requires git):

```bash
git clone https://github.com/song2yu/Mono2Stereo.git
cd mono2stereo
```

First, you need to download the weights of [depth anything v2-small](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) to the 'depth/checkpoints/' folder, and also download the weights of the [dual-condition baseline model](https://pan.baidu.com/s/12cG1_o9G8qwkQLKm7B6XNQ?pwd=phns) (or from 🤗[mono2stereo.ckpt](https://huggingface.co/Two-hot/Mono2Stereo/tree/main)) to the 'checkpoint/' folder.



create a Python native virtual environment and install dependencies into it:

```bash
conda create -n stereo python=3.8 -y
conda activate stereo
pip install -r requirements.txt
```
<br>

**🏃🏻‍♂️‍➡️ Inference**
```bash
python test.py
```
<br>

**📊 Dataset**
<br>
We provide the data processing code in data_process.py. The video data can be downloaded from this [website](https://www.3donlinefilms.com/). 
<br>
We provide [test data](https://pan.baidu.com/s/135vSm_ZqMrNA-qijVOvyhw?pwd=cej5) (or from 🤗[mono2stereo-test.zip](https://huggingface.co/Two-hot/Mono2Stereo/tree/main)) for fair comparison. Additionally, we recommend using the [Inria 3DMovies](https://www.di.ens.fr/willow/research/stereoseg/) for model testing.

<br>

## 🎓 Citation

If you find this project useful, please consider citing:

```bibtex
@misc{yu2025mono2stereobenchmarkempiricalstudy,
      title={Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion}, 
      author={Songsong Yu and Yuxin Chen and Zhongang Qi and Zeke Xie and Yifan Wang and Lijun Wang and Ying Shan and Huchuan Lu},
      year={2025},
      eprint={2503.22262},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.22262}, 
}
```
<br>

## 🫂 Acknowledgement
We would like to express our sincere gratitude to the open-source projects [depth anything](https://github.com/LiheYoung/Depth-Anything) and [Marigold](https://github.com/prs-eth/Marigold). This project is based on their code.




