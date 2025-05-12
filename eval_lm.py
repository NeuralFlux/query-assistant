#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Define metrics

import os
from urllib.parse import urlparse, parse_qs

import outlines.samplers

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

@outlines.prompt
def rag_prompt(instruction, docs, relevant_schema):
    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {{ docs }}\n Schema: {{ relevant_schema }}\n<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ instruction }}. Write an API call.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

@outlines.prompt
def few_shot_with_rag(instruction, examples, docs, relevant_schema):
    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {{ docs }}\n Schema: {{ relevant_schema }}\n<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ instruction }}. Write an API call and do not write anything else in your response.

Examples
--------

{% for example in examples %}
Query: {{ example.instruction }}
API Call: {{ example.output }}

{% endfor %}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# #### Latest

import outlines
from peft import PeftModel

model_path = "models/meta_llama3_1"
adapter_path = "models/ft/qlora_train_split/adapter"
# adapter_path = "models/ft/lora_train_split_3b/checkpoint-713"
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True).to("cuda")
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, output_attentions=True, weights_only=True).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model.eval()

from pydantic import BaseModel, constr
from typing import Annotated

class APICall(BaseModel):
    api_call: str

def evaluate(api_call: str):
    return None

json_schema = """
{
  "title": "response",
  "description": "chatbot response",
  "type": "object",
  "properties": {
    
  }
}
"""

# model = outlines.models.Transformers(model, tokenizer)
model = outlines.models.openai("gpt-4o-mini", api_key=os.environ["OPENAI_KEY"])
# sampler = outlines.samplers.beam_search(beams=5)
generator = outlines.generate.text(model)
# generator = outlines.generate.regex(model, r"/v3/query/.*", sampler)
# generator = outlines.generate.regex(model, r"/v3/query/.*")

# ## Part 3: Evaluate

from datasets import load_dataset

train_set = load_dataset("moltres23/biothings-query-instruction-pairs", split="train")
test_set = load_dataset("moltres23/biothings-query-instruction-pairs", split="test")


import tqdm
import random
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_name = 'Snowflake/snowflake-arctic-embed-l'
model_kwargs = {"device": "cuda:1"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

vectorstore = FAISS.load_local(folder_path="data/rag", index_name="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 16})

def process_retrieved_docs(doc_batches):
    return ["\n\n".join([doc.page_content for doc in doc_batch]) for doc_batch in doc_batches]

random.seed(42)
BATCH_SIZE = 1  # more than 1 not supported
N_SHOT = 10  # size of ICL examples
all_responses = []

with torch.no_grad():
    for idx in tqdm.tqdm(range(0, len(test_set), BATCH_SIZE)):
        # hacky dict of lists to list of dicts conversion
        icl_example_indices = random.sample(range(len(train_set)), N_SHOT)  # same examples for each test batch
        icl_examples = [dict(zip(train_set[icl_example_indices].keys(), values)) for values in zip(*train_set[icl_example_indices].values())]

        batch = test_set[idx:(idx + BATCH_SIZE)]
        doc_batches = retriever.batch(batch["instruction"])  # rag

        # batched_inputs = list(map(few_shot_prompt, batch["instruction"], [icl_examples], [docs], [description]))
        # batched_inputs = list(map(default_prompt, batch["instruction"], [docs], [description]))
        batched_inputs = list(map(few_shot_with_rag, batch["instruction"], [icl_examples], [docs], process_retrieved_docs(doc_batches)))
        # batched_inputs = list(map(rag_prompt, batch["instruction"], [docs], process_retrieved_docs(doc_batches)))
        if idx == 0:
            print(f"\nDemo Input: {batched_inputs}\n")

        batch_responses = generator(batched_inputs)
        # all_responses.extend(batch_responses.values())
        # all_responses.append(tuple(d["api_call"] for d in batch_responses))  # for beam search
        all_responses.append(batch_responses)


import pickle

with open('responses_openai_mini_rag_icl.pkl', 'wb') as fd:
   pickle.dump(all_responses, fd)
