# Transition Document

## Overview

This document provides a high-level overview of the project directory structure. At this stage, all the experiments only use MyGene. However, with the goal to scale this to other APIs, datagen and data_annotate notebooks have been partially altered to accommodate more APIs. This project uses `torchtune` for fine-tuning. However, HuggingFace Transformers was later determined to provide more flexible and robust API. Therefore, the project will benefit from refactoring to HF.

> [!IMPORTANT]
> Use Python 3.10

---

## Directory Structure

```bash
.
├── data  # all the data needed to run the experiments
│   ├── original  # data used to create the instruction-query pairs needed for fine-tuning
│   ├── prompts  # prompt for creating fine-tuning pairs
│   ├── ft  # data used for fine-tuning
│   ├── rag  # vector index, and docs in a RAG-compatible format
│   ├── schema  # schemas for the 4 BioThings APIs. JSONs are the mapping from ES and CSVs are annotated by GPT 4o
│   └── logs  # past year API logs for the 4 BioThings APIs
├── grammars  # prototype grammars used for structured generation
├── ft  # configs for fine-tuning
├── evals  # configs for train-test evals using torchtune
├── models  # weights for fine-tuned models and/or adapters
└── server  # gradio server to host an interface for this project
```

---

## Files Overview
### Please look at the next section to understand the key files in the pipeline

``` bash
.
├── const.py  # OpenAI chat template
├── datagen.py
├── datagen.ipynb
├── data_annotate.ipynb
├── descriptions.json
├── eval_lm.py
├── evaluate.ipynb
├── full_reqs.txt  # all the requirements generated with pip freeze
├── inference.ipynb
├── logs_{model}_eval.txt  # logs for eval of {model}
├── requirements.txt  # requirements written by eyeballing for a leaner file
├── responses_{config}.pkl  # responses for instructions in test set for model with {config}
└── train.ipynb  # (unstable) an attempt to switch to HF for fine-tuning
```

---

## Typical Workflow
1. 

