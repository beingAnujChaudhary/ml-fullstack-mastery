
# 🚀 ML Full-Stack Mastery

**An integrated learning journey from mathematical foundations to production-grade AI systems — built through code, not just theory.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLOps](https://img.shields.io/badge/MLOps-Production_Ready-green)](https://ml-ops.org/)
[![Roadmap](https://img.shields.io/badge/Roadmap-12_Week_Curriculum-orange)](README.md)

A **cohesive, production-focused ML curriculum and portfolio** that bridges the gap between *algorithmic understanding* and *real-world machine learning engineering*.
This repository documents a structured **12-week journey** of building, deploying, monitoring, and maintaining machine learning and GenAI systems.

---

## 🎯 What Problem Does This Solve?

The machine learning ecosystem often produces two extremes:

| Theory-Only Practitioners | Tool-Only Practitioners         |
| ------------------------- | ------------------------------- |
| Strong math, weak systems | Can deploy APIs, lack intuition |
| Can explain models        | Can’t diagnose failures         |
| Rarely ship to production | Overfit to frameworks           |
| *“Paper Tigers”*          | *“API Consumers”*               |

This repository demonstrates **the middle path**:

> **Deep theoretical understanding + production engineering competence**

---

## 🧠 Learning Philosophy — White Box → Production

Every concept in this repository is learned and applied in **three layers**:

1. **White-Box Foundations**
   Algorithms implemented from scratch to build intuition.

2. **Framework & Pipeline Mastery**
   Scikit-learn and PyTorch used correctly, reproducibly, and at scale.

3. **Production & Systems Engineering**
   Deployment, monitoring, experimentation, and lifecycle management.

The objective is not just training models — but **owning the entire ML system**.

---

## 🗺️ Curriculum Overview (High-Level)

```mermaid
flowchart TD
    %% Nodes
    A[🎯 Full-Stack ML Mastery]
    
    subgraph P1 [Week 1-4]
        B[Phase 1: Foundations]
        B1[📘 Mathematical Intuition]
        B2[📈 Statistical Rigor]
        B3[⚙️ Algorithm Implementation]
    end

    subgraph P2 [Week 5-8]
        C[Phase 2: Production]
        C1[🏗️ Pipeline Architecture]
        C2[📦 Containerization]
        C3[🔍 Experiment Tracking]
        C4[📊 Monitoring]
    end

    subgraph P3 [Week 9-12]
        D[Phase 3: Advanced]
        D1[🤖 Transformer Systems]
        D2[🔗 RAG Architecture]
        D3[🤝 AI Agents]
        D4[⚡ PEFT/LoRA]
    end

    subgraph Out [Outcomes]
        E[🎓 Career Outcomes]
        E1[ML Engineer]
        E2[AI Engineer]
        E3[MLOps Engineer]
    end

    %% Vertical Spine Connections (The Fix)
    A --> B
    B --> C
    C --> D
    D --> E

    %% Detail Connections (Branching slightly)
    B .-> B1 & B2 & B3
    C .-> C1 & C2 & C3 & C4
    D .-> D1 & D2 & D3 & D4
    E .-> E1 & E2 & E3

    %% Styling
    style A fill:#2e86ab,color:#fff
    style B fill:#a23b72,color:#fff
    style C fill:#f18f01,color:#000
    style D fill:#73ab84,color:#fff
    style E fill:#c73e1d,color:#fff```
---

## 🏗️ Repository Architecture — Dual Portfolio Strategy

This repository deliberately separates **academic depth** from **engineering excellence**.

```
ml-fullstack-mastery/
│
├── theory-foundations/        # White-box ML & math from scratch
│   ├── linear-algebra/
│   ├── optimization/
│   ├── algorithms-from-scratch/
│   └── statistical-rigor/
│
├── framework-projects/        # End-to-end ML pipelines
│   ├── housing-valuation/
│   ├── churn-prediction/
│   └── image-classifier/
│
├── advanced-projects/         # GenAI, agents, and MLOps systems
│   ├── rag-knowledge-bot/
│   ├── ai-agent-system/
│   └── mlops-pipeline/
│
├── notebooks/                 # Exploratory work (temporary)
├── docs/                      # Weekly logs, model cards, ADRs
├── tests/                     # Unit, integration, and ML tests
├── infrastructure/            # Docker, orchestration, cloud
│
├── environment.yml            # Reproducible conda environment
├── requirements.txt           # pip-compatible dependencies
├── pyproject.toml             # Modern Python packaging
├── Makefile                   # Standardized commands
├── Dockerfile                 # Production container template
└── README.md
```

---

## 📚 Curated Learning Sources & Their Roles

### 🔬 Layer 1: White-Box Foundations

| Resource                                  | Purpose                               | Applied In                      |
| ----------------------------------------- | ------------------------------------- | ------------------------------- |
| **Joel Grus – Data Science from Scratch** | Algorithmic intuition via pure Python | Linear algebra, regression      |
| **Danny Friedman – ML from Scratch**      | Optimization & derivations            | Gradient descent, loss surfaces |
| **ISLP**                                  | Statistical rigor & inference         | Bias–variance, resampling       |

Frameworks are introduced **only after** core understanding is established.

---

### ⚙️ Layer 2: Framework & Pipeline Mastery

| Resource                      | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| **Hands-On ML (Géron)**       | End-to-end pipelines & feature engineering |
| **ML with PyTorch (Raschka)** | Deep learning fundamentals                 |
| **scikit-learn**              | Classical ML production standard           |

Focus areas:

* Reproducibility
* Clean APIs
* Correct evaluation
* Testability

---

### 🚀 Layer 3: Production & MLOps

| Tool / Resource    | Why                     |
| ------------------ | ----------------------- |
| **FastAPI**        | Model serving           |
| **MLflow**         | Experiment tracking     |
| **Docker**         | Environment consistency |
| **MLOps Zoomcamp** | Lifecycle management    |

Models become **products**, not notebooks.

---

### 🤖 Layer 4: Modern GenAI Systems

| Technology         | Focus                   |
| ------------------ | ----------------------- |
| Transformers       | Representation learning |
| RAG                | Knowledge-grounded LLMs |
| FAISS / Vector DBs | Scalable retrieval      |
| LoRA / PEFT        | Efficient fine-tuning   |
| Agents             | Tool-using AI systems   |

Emphasis is on **system design**, not API calls.

---

## 🗓️ 12-Week Learning Progression

### Phase 1 (Weeks 1–4): Foundations — *Understand the Engine*

* Linear algebra & optimization from scratch
* Classical ML implemented manually
* Statistical diagnostics and inference

### Phase 2 (Weeks 5–8): Production Engineering — *Build the Car*

* Scikit-learn pipelines
* MLflow experiment tracking
* FastAPI + Docker deployment

### Phase 3 (Weeks 9–12): Advanced Systems — *Drive in Traffic*

* RAG and vector search
* LLM fine-tuning (LoRA)
* AI agents & MLOps orchestration

---

## 🧪 Engineering Standards

* Notebooks are **exploratory only**
* Final logic lives in **testable Python modules**
* Dependencies are **locked**
* Code evolves from learning → production
* Each major project includes:

  * Problem framing
  * Modeling rationale
  * Evaluation strategy
  * Deployment notes

---

## 📈 What This Repository Demonstrates

By completion, this repository demonstrates the ability to:

* Implement ML algorithms from first principles
* Build robust, reproducible ML pipelines
* Apply statistical evaluation correctly
* Deploy containerized ML services
* Design GenAI systems (RAG, agents)
* Apply MLOps practices for long-term reliability

---

## 🎯 Target Roles

This portfolio is designed to align with:

* Machine Learning Engineer
* Applied Data Scientist
* AI / GenAI Engineer
* MLOps Engineer

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/ml-fullstack-mastery.git
cd ml-fullstack-mastery

# Conda (recommended)
conda env create -f environment.yml
conda activate ml-fullstack

# Or pip
pip install -r requirements.txt
```

---

## 🔍 Differentiation Statement

> **This is not a collection of notebooks.**
>
> It is a full-stack machine learning systems portfolio demonstrating understanding, implementation, deployment, and maintenance of real-world ML systems.

---

## 📄 License

MIT License — free to use, adapt, and extend with attribution.

---

### 👉 Where to Start

Begin with:

```
theory-foundations/linear-algebra/
```

Each directory contains:

* `IMPLEMENTATION.md`
* `RESOURCES.md`
* `CHECKPOINT.md`

> *Frameworks change. Fundamentals endure.*
