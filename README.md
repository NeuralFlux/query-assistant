# 📝 Query Assistant

## Overview

This document provides an overview of the project's directory structure and execution. At this stage, all the experiments only use MyGene. However, with the goal to scale this to other APIs, datagen and data_annotate notebooks have been partially altered to accommodate more APIs. This project uses `torchtune` for fine-tuning.

Throughout this project, _instructions_ refer to the command a human gives to an LLM to do something. _Query_ refers to a typical API query sent to BioThings API to get some data.

> [!IMPORTANT]
> Use Python 3.10

---

## Directory Structure
Data, models, and pickled response files are shared as tar.gz files for efficiency.

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
├── datagen.py  # script version of the notebook
├── datagen.ipynb  # transform API logs into fine-tuning data
├── data_annotate.ipynb  # annotate schema for a target API
├── descriptions.json  # schema annotation by GPT 4o for MyGene
├── eval_lm.py  # script with configurable parameters to generate LLM responses
├── evaluate.ipynb  # notebook to score pickled LLM responses against test set (or train set)
├── full_reqs.txt  # all the requirements generated with pip freeze
├── inference.ipynb  # (optional) a notebook to play around with Llama CPP Python API
├── logs_{model}_eval.txt  # logs for eval of {model}
├── requirements.txt  # requirements written by eyeballing for a leaner file
├── responses_{config}.pkl  # responses for instructions in test set for model with {config}
└── train.ipynb  # (unstable) an attempt to switch to HF for fine-tuning
```

---

## Typical Workflow
0. Add OpenAI API key to `.env` file, install dependencies from `requirements.txt`
1. For a target API service, run the data_annotate notebook to get its schema annotated by OpenAI chatbots. Also, download the docs for this API service. (outputs => annotated schema, docs)
2. You may skip part 2 of this notebook if you do not intend to fine-tune. Run the datagen notebook to filter the target API logs and get _largely_ unique queries. The second part of this notebook produces instruction-query pairs for fine-tuning. This will likely consume a huge amount of GPU memory. (outputs => "good" queries, instruction-query pairs for fine-tuning).
3. (optional) Fine-tuning is currently done with a torchtune command with a certain config. However, it is recommended to switch to HuggingFace Transformers library to fine-tune.
4. Run the eval_lm script with the required setting. You may use RAG, ICL, a fine-tuned model etc. The script stores the LLM responses in a pickle file so you may evaluate the generated responses later (output => pickle file of LLM generated responses).
5. Run the evaluate notebook to score the responses from step 3 against the test set generated in step 2.

## Additional Notes
1. Manual scrutiny is paramount to ensure the instructions generated from step 2 are not "fluffy". Otherwise, the LLM will learn from a bad distribution and fail to generalize later. One way to do this is by identifying phrases that are common across instructions and removing them. Fluffy here means words that do not affect the sentence much when removed.
2. It is beneficial to separate the generated pairs from step 2 into train and test sets. Since we generate 2 instructions per query, separate by query and not instruction. Otherwise, the test set data may leak into training set.
3. All the techniques that helped in fitting fine-tuning to GPU budget on su10 are listed [here](https://medium.com/@anudeep.tubati/5-tricks-i-used-to-train-llama-8b-with-16k-context-on-a-48gb-gpu-and-2-bonus-tricks-c17c65141234).
4. Meta Llama 3.1 8B was used for fine-tuning. Add the adapter to native model weights to get the fine-tuned model. Only the adapter weights are provided here.

## Future Work
- [ ] Port fine-tuning from torchtune to HuggingFace (better flexibility and robust dev APIs)
- [ ] Scale datagen and data annotate to take any API (currently in progress for 4 BioThings APIs)
- [ ] Integrate lm-evaluation-harness as fine-tuning the LLM may cause catastrophic forgetting
