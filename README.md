<div align="center">

<h1>SLOPE: Optimistic Potential Landscape Shaping<br>for Model-based Reinforcement Learning</h1>

<p><strong>NeurIPS 2026</strong></p>

<p>
Yao-Hui Li, Zeyu Wang, Xin Li, Wei Pang, Yingfang Yuan,<br>
Zhenakun Chen, Boya Zhang, Riashat Islam, Alex Lamb, Yonggang Zhang
</p>

<p>Official implementation of <strong>SLOPE</strong>.</p>

<p>
  <a href="https://huggingface.co/datasets/Arya87/SLOPE_Demos">🤗 Demonstration Dataset</a>
  &nbsp;•&nbsp;
  <a href="#installation">Installation</a>
  &nbsp;•&nbsp;
  <a href="#training">Training</a>
  &nbsp;•&nbsp;
  <a href="#real-world-robotic-deployments">Real-World Demos</a>
</p>

</div>

## Overview

Model-based reinforcement learning (MBRL) is sample-efficient but struggles in sparse reward settings. A critical bottleneck is the lack of informative gradients: standard reward models often produce flat landscapes that provide little guidance for planning.

We propose **S**haping **L**andscapes with **O**ptimistic **P**otential **E**stimates (**SLOPE**), a framework that shifts reward modeling from predicting sparse scalars to constructing informative potential landscapes. SLOPE uses optimistic distributional regression to estimate high-confidence upper bounds, amplifying rare success signals and providing informative gradients for exploration.

Evaluations on **30+ tasks across 5 benchmarks**, together with **real-world robotic deployments**, demonstrate that SLOPE consistently outperforms leading baselines under **fully sparse**, **semi-sparse**, and **dense** reward settings.

## Demonstration Dataset

The demonstrations used in our experiments are available on Hugging Face:

**🤗 [SLOPE_Demos](https://huggingface.co/datasets/Arya87/SLOPE_Demos)**

Download the demonstrations and set `demo_path` to the corresponding `.pkl` file when running training.

## Installation

Run the following commands in order.

### 1. Create and activate the environment

```bash
conda create -n slope python=3.10
conda activate slope
```

### 2. Install PyTorch

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install \
  absl-py==2.1.0 \
  glfw==2.7.0 \
  ffmpeg==1.4 \
  imageio==2.34.1 \
  imageio-ffmpeg==0.4.9 \
  h5py==3.11.0 \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  submitit==1.5.1 \
  omegaconf==2.3.0 \
  moviepy==1.0.3 \
  mujoco==2.3.7 \
  numpy==1.24.4 \
  kornia==0.7.2 \
  termcolor==2.4.0 \
  tqdm==4.66.4 \
  pandas==2.0.3 \
  wandb==0.17.4 \
  matplotlib==3.7.5 \
  seaborn==0.13.2 \
  gpustat==1.1.1

pip install torchrl-nightly==2025.2.3
pip install tensordict-nightly==2025.2.3
```

### 4. Install simulation environments

```bash
pip install "git+https://github.com/Farama-Foundation/Metaworld.git@c822f28f582ba1ad49eb5dcf61016566f28003ba#egg=metaworld"

pip install "git+https://github.com/haosulab/ManiSkill.git@0de43f1f5e10dabd860baacbcbd10ab443e998b1#egg=mani-skill"
```

## Training

The following examples use the `ms-poke-cube-sparse` task. Run them from the repository root and replace `/path/to/expert_demos/` with your local demonstration directory.

### SLOPE

```bash
python train.py \
  task=ms-poke-cube-sparse \
  steps=500000 \
  demo_path=/path/to/expert_demos/ms-poke-cube-sparse.pkl \
  enable_reward_learning=false \
  reward_shapping=true \
  warm_MPPI=true \
  update_succ_buffer=true \
  use_quantile_loss=true \
  n_demos=5 \
  seed=1
```

### DEMO3

```bash
python train.py \
  task=ms-poke-cube-sparse \
  steps=500000 \
  demo_path=/path/to/expert_demos/ms-poke-cube-sparse.pkl \
  enable_reward_learning=true \
  n_demos=5 \
  seed=1
```

### MoDem

```bash
python train.py \
  task=ms-poke-cube-sparse \
  steps=500000 \
  demo_path=/path/to/expert_demos/ms-poke-cube-sparse.pkl \
  enable_reward_learning=false \
  n_demos=5 \
  seed=1
```

### TD-MPC2

```bash
python train.py \
  task=ms-poke-cube-sparse \
  steps=500000 \
  enable_reward_learning=false \
  demo_sampling_ratio=0.0 \
  policy_pretraining=false \
  seed=1
```

## Real-World Robotic Deployments

SLOPE is evaluated on real-world robotic manipulation tasks, including **Press Button**, **Push Cube**, and **Grasp Cube**.

<table>
  <tr>
    <th align="center">Press Button</th>
    <th align="center">Push Cube</th>
    <th align="center">Grasp Cube</th>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="./assets/press_button.gif" alt="Robot pressing a button" width="100%">
    </td>
    <td align="center" width="33%">
      <img src="./assets/push_cube.gif" alt="Robot pushing a cube" width="100%">
    </td>
    <td align="center" width="33%">
      <img src="./assets/grasp_cube.gif" alt="Robot grasping a cube" width="100%">
    </td>
  </tr>
</table>

*The GIF files are stored in the [`assets/`](./assets/) directory.*

## Acknowledgements

We thank the authors of [DEMO3](https://github.com/adrialopezescoriza/demo3) and [TD-MPC2](https://github.com/nicklashansen/tdmpc2) for making their code publicly available. Their implementations provided valuable foundations for this project.
