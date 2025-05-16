# Contributing to gym4ReaL

Thank you for considering contributing to **Gym4ReaL**! 🎉

This guide will help you set up your environment, understand the project structure, and submit effective contributions.

---

## 📌 Project Overview

**Gym4ReaL** is a Gym-compatiblelibrary designed for training and evaluate RL algorithm on real-world benchmarks.

---

## 🛠️ Getting Started

### 1. Fork the Repository

Click the **Fork** button at the top right of the [gym4ReaL GitHub page](https://github.com/Daveonwave/gym4ReaL).

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/gym4ReaL.git
cd gym4ReaL
```

### 3. Create a virtual env (with venv or conda)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -e ".[dev]"
```

## Project Structure

```bash
gym4real/
├── algorithms/           # Algorithms to train
├── data/                 # Data files
├── envs/                 # Custom environments
```
