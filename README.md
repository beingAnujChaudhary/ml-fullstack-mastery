# ML Full-Stack Mastery

## Purpose
This repository documents a 12-week, end-to-end journey to master machine learning
from first principles to production-ready systems.

The goal is not course completion, but:
- algorithmic understanding
- statistical reasoning
- production ML engineering
- modern GenAI systems (RAG, Agents, MLOps)

## Repository Structure

- theory-foundations/  
  White-box implementations of ML algorithms from scratch.

- framework-projects/  
  End-to-end ML projects using scikit-learn, PyTorch, and deployment tools.

- advanced-projects/  
  System-level AI architectures (RAG, Agents, MLOps).

- notebooks/  
  Temporary exploratory notebooks. All final logic is refactored into `.py` files.

- docs/  
  Weekly logs, model cards, and learning references.

## Learning Philosophy
- Derive → Implement → Deploy
- Notebooks are disposable; code is permanent
- Every model must be evaluated and explained

## Environment Setup

```bash
conda env create -f environment.yml
conda activate ml-fullstack
