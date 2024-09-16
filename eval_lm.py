#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Define metrics

from urllib.parse import urlparse, parse_qs

def parse_url(url):
    """
    Parse a URL into its components.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    return {
        'path': parsed_url.path,
        'params': query_params
    }

def is_url_structure_matching(candidate, reference):
    """
    Compare the path and query parameters of the candidate and reference URLs.
    """
    if candidate['path'] != reference['path']:
        return False

    if sorted(candidate['params'].keys()) != sorted(reference['params'].keys()):
        return False

    for key in reference['params']:
        if key not in candidate['params']:
            return False
        if sorted(candidate['params'][key]) != sorted(reference['params'][key]):
            return False
    
    return True

def evaluate_get_request_accuracy(generated_url, reference_url):
    """
    Evaluate if the generated GET request is equivalent to the reference GET request.
    """
    candidate = parse_url(generated_url)
    reference = parse_url(reference_url)
    
    return is_url_structure_matching(candidate, reference)

def score_ast_batched(preds, refs):
    evals = tuple(map(evaluate_get_request_accuracy, preds, refs))
    return sum(evals) / len(evals)


import evaluate
bert_scorer = evaluate.load("bertscore")
bert_score_fn = lambda preds, refs: bert_scorer.compute(predictions=preds, references=refs, lang="en", model_type="microsoft/codebert-base", num_layers=12, device="cuda")


preds = ["/v3/query/?q=symbol:ZFAND4&species=mouse"]
refs = ["/v3/query/?species=mouse&q=symbol:ZFAND4"]

print("AST eval", score_ast_batched(preds, refs))
print("BERT Score", bert_score_fn(preds, refs))


# ## Part 2: Load models

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


with open("gene_query_docs.txt", "r") as doc_fd:
    docs = doc_fd.read()

with open("data/original/compact_desc_with_context.csv") as desc_fd:
    description = desc_fd.read()


import outlines

@outlines.prompt
def default_prompt(instruction, docs, description):
    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {{ docs }}\n Schema: {{ description }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ instruction }}. Write an API call.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

@outlines.prompt
def few_shot_prompt(instruction, examples, docs, description):
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {{ docs }}\n Schema: {{ description }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ instruction }}. Write an API call.

Examples
--------

{% for example in examples %}
Query: {{ example.instruction }}
API Call: {{ example.output }}

{% endfor %}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


inst = "Find the UniProt ID for the ENSG00000103187 gene in human. Limit the search to Ensembl gene IDs."
prompt = default_prompt(inst, docs, description)
print("Start", prompt[:250])
print("End", prompt[-150:])


# #### Deprecated

import inspect
prompt_template = inspect.cleandoc("""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {docs}\n Schema: {description}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}. Write an API call.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")


prompt_gen = lambda inst: prompt_template.format(docs=docs, description=description, instruction=inst)
prompt = prompt_gen("Find the UniProt ID for the ENSG00000103187 gene in human. Limit the search to Ensembl gene IDs.")
print("Start", prompt[:250])
print("End", prompt[-150:])


# #### Latest

import outlines
from peft import PeftModel

model_path = "models/meta_llama3_1"
adapter_path = "models/ft/qlora_train_split/adapter"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True).to("cuda")
model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, output_attentions=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


def evaluate(api_call: str):
    return None

model = outlines.models.Transformers(model, tokenizer)
generator = outlines.generate.json(model, evaluate)
# generator = outlines.generate.regex(model, r"/v3/.+/.+")


sample_api_call = generator([prompt])
sample_api_call


# ## Part 3: Evaluate

from datasets import load_dataset

dataset = load_dataset("moltres23/biothings-query-instruction-pairs")
train_set, test_set = dataset["train"], dataset["test"]
train_set


import tqdm
import random

random.seed(42)
BATCH_SIZE = 1
N_SHOT = 10  # size of ICL examples
all_responses = []

with torch.no_grad():
    for idx in tqdm.tqdm(range(0, len(test_set), BATCH_SIZE)):
        # hacky dict of lists to list of dicts conversion
        icl_example_indices = random.sample(range(len(train_set)), N_SHOT)  # same examples for each test batch
        icl_examples = [dict(zip(train_set[icl_example_indices].keys(), values)) for values in zip(*train_set[icl_example_indices].values())]

        batch = test_set[idx:(idx + BATCH_SIZE)]
        # batched_inputs = list(map(few_shot_prompt, batch["instruction"], [icl_examples], [docs], [description]))
        batched_inputs = list(map(default_prompt, batch["instruction"], [docs], [description]))
        batch_responses = generator(batched_inputs)
        all_responses.extend(batch_responses.values())


import pickle


with open('responses.pkl', 'wb') as fd:
   pickle.dump(all_responses, fd)


with open('responses.pkl', 'rb') as fd:
   all_responses = pickle.load(fd)


# NOTE: using regex match to get the response in the expected format
import re
def regex_match(string):
    match = re.search(r"/v3/.+", string)
    return match.group(0) if match is not None else ""


regex_match("GET https://mygene.infov3/query?fields=human")


all_responses = list(map(lambda x: regex_match(x), all_responses))
ast_eval = score_ast_batched(all_responses, test_set["output"])
bertscore_evals = bert_score_fn(all_responses, test_set["output"])


import numpy as np

# we include 0s in mean because otherwise merely getting
# one correct answer will skew the metric
print("AST eval", ast_eval)
print("BERT Score", np.mean(bertscore_evals["recall"]))
