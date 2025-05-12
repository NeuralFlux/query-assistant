import os

from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE, LlamaGrammar
import pandas as pd
from tqdm import tqdm
import json

import const

if __name__ == "__main__":
    llm = Llama(
        model_path=os.path.expanduser("/home/atubati/vendor/weights_llama3.1/Meta-Llama-3.1-70B-Instruct-Q6_K_L/Meta-Llama-3.1-70B-Instruct-Q6_K_L-00001-of-00002.gguf"),
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=16384, # Uncomment to increase the context window
        # split_mode=LLAMA_SPLIT_MODE_NONE  # Uncomment to use single-GPU
    )

    data = pd.read_csv("data/queries_with_hits.csv")["query"].values
    print("Input queries =>", data.shape)

    with open("data/prompts/datagen_prompt.md") as fd:
        base_prompt = fd.read()

    with open("gene_query_docs.txt") as doc_fd:
        docs = doc_fd.read()

    with open("data/original/compact_desc_with_context.csv") as desc_fd:
        description = desc_fd.read()

    file_name = "data/ft/inst_query_pairs.csv"

    BATCH_SIZE = 10  # NOTE: change in prompt also if changed
    for idx in tqdm(range(0, len(data), BATCH_SIZE)):
        prompt = f"{base_prompt}\n" + "\n".join(data[idx:(idx + BATCH_SIZE)])
        if idx == 0:
            print(prompt)
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": f"\nUse the documentation and schema to complete the user-given task. Docs: {docs}\n\nSchema: {description}"},
                {"role": "user", "content": prompt},
            ],
            # grammar=query_grammar
            response_format={
                "type": "json_object",
                "schema": const.OUTPUT_SCHEMA
            },
        )

        resp = json.loads(output["choices"][0]["message"]["content"])
        pd.DataFrame(resp["instructions"]).to_csv(file_name, mode="a", index=False, header=not os.path.exists(file_name))
