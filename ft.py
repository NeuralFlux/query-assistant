import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig

model_name = "models/meta_llama3_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configuring LoRA with torchtune config
lora_config = LoraConfig(
    r=64,  # Rank for low-rank adaptation
    lora_alpha=128,  # Scaling factor
    target_modules=["q_proj", "v_proj", "output_proj", "w1", "w2", "w3"],  # Targeting specific attention layers
    lora_dropout=0.0,
)

# Apply LoRA to the pre-trained model
model = get_peft_model(model, lora_config).to("cuda:0")

from datasets import load_dataset

dataset = load_dataset("moltres23/biothings-query-instruction-pairs")

with open("gene_query_docs.txt", "r") as doc_fd:
    docs = doc_fd.read()

import outlines

@outlines.prompt
def rag_prompt(instruction, output, docs, relevant_schema):
    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the documentation and schema to complete the user-given task.
Docs: {{ docs }}\n Schema: {{ relevant_schema }}\n<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ instruction }}. Write an API call.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{ output }}<|eot_id|>"""

tokenizer.pad_token = "<|finetune_right_pad_id|>"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_name = 'Snowflake/snowflake-arctic-embed-l'
model_kwargs = {"device": "cuda:0"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

vectorstore = FAISS.load_local(folder_path="data/rag", index_name="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 16})

from trl import SFTTrainer, SFTConfig

def process_retrieved_docs(doc_batches):
    return ["\n\n".join([doc.page_content for doc in doc_batch]) for doc_batch in doc_batches]

def formatting_prompts_func(example):
    output_texts = []
    inputs = example["instruction"] if isinstance(example["instruction"], list) else [example["instruction"]]
    outputs = example["output"] if isinstance(example["output"], list) else [example["output"]]
    
    # Retrieve documents for each question
    doc_batches = retriever.batch(inputs)
    doc_batches = process_retrieved_docs(doc_batches)
    for idx in range(len(inputs)):
        # Format the prompt using the few-shot with RAG template
        text = rag_prompt(
            instruction=inputs[idx],  # Query or instruction (the question)
            output=outputs[idx],
            docs=docs,
            relevant_schema=doc_batches[idx]
        )
        
        output_texts.append(text)
    
    return output_texts

print(formatting_prompts_func(dataset["train"][:2]))

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=formatting_prompts_func,
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    args=SFTConfig(
        learning_rate=3e-4,
        max_seq_length=4096,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_steps=100,
        packing=False,
        output_dir="models/ft/lora_train_split_3b",
        seed=0,
    )
)

del retriever, embeddings
torch.cuda.empty_cache()
trainer.train()
