import gradio as gr
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
import torch
from peft import PeftModel
import argparse
import logging

from prompts import default_prompt

# Setup logger
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# Initialize SQLite database
with sqlite3.connect('responses.db') as conn:
    cursor = conn.cursor()
    # Create tables if not existent
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            llm_response TEXT,
            user_correction TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT
        )
    ''')
    conn.commit()

model_path = "../models/meta_llama3_1"
adapter_path = "../models/ft/qlora_train_split/adapter"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, weights_only=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

model = outlines.models.Transformers(model, tokenizer)
generator = outlines.generate.text(model)

# load context docs
# with open("../gene_query_docs.txt", "r") as doc_fd:
#     docs = doc_fd.read()

# with open("../data/original/compact_desc_with_context.csv") as desc_fd:
#     description = desc_fd.read()

with open("./pending_api/docs.txt", "r") as doc_fd:
    docs = doc_fd.read()

with open("./pending_api/fields.txt") as desc_fd:
    description = desc_fd.read()

# Function to generate response from LLM
def generate_response(user_input):
    try:
        prompt = default_prompt(user_input, docs, description)
        response = generator(prompt, max_tokens=80)
        return response.strip()
    except Exception as e:
        logging.error(f"{user_input}, {e}")
        gr.Warning(f"Error generating response: {e}")
        return f"Error generating response: {e}"

def store_query(user_input, llm_response: str):
    if llm_response.startswith("Error generating response"):
        logging.warning("Malformed response NOT stored")
        return
    try:
        with sqlite3.connect('responses.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO queries (query, response)
                VALUES (?, ?)
            ''', (user_input, llm_response))
            conn.commit()
        logging.info("Query and response stored")
    except Exception as e:
        logging.error(f"DB Query Store {user_input}, {llm_response}, {e}")

# Function to handle the interaction and save to the database
def handle_interaction(user_input, llm_response, user_correction):
    try:
        with sqlite3.connect('responses.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_interactions (user_input, llm_response, user_correction)
                VALUES (?, ?, ?)
            ''', (user_input, llm_response, user_correction))
            conn.commit()
        gr.Info("Your correction has been saved! Thank you.")
        logging.info("Correction saved")
    except Exception as e:
        logging.error(f"DB {user_input}, {llm_response}, {user_correction}, {e}")
        gr.Warning(e)

# Gradio function to handle user input
def main_function(user_input):
    llm_response = generate_response(user_input)
    torch.cuda.empty_cache()
    store_query(user_input, llm_response)
    markdown_link = f'<a href="https://mygene.info{llm_response}" target="_blank">Test Response Link</a>'
    logging.info("LLM response sent successfully")
    return llm_response, markdown_link, ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("# MyGene Query LLM")
        gr.Markdown("## Usage Guide")
        gr.Markdown("""This app is a prototype for internal use meant to collect data for fine-tuning. Therefore, only good paths
                    are programmed. Please adhere to this usage guide to avoid errors and bugs.
                    Should inference be down for more than 5 minutes, please send a message to
                    Anudeep Tubati on Slack or atubati@scripps.edu""")
        gr.Markdown("""0. Expected GPU memory usage - 16GB when idle, 30GB when generating response.
                    1. Prompt the LLM in `box 1`. You may ask a question OR give a command.
                    For eg, "find uniprot info for all human genes". The LLM
                    is given the docs and field descriptions to help with query generation.
                    2. If the LLM generates correct output, yay!
                    3. Otherwise, provide the correct output in `box 3` in the form `/v3/query...` or `/v3/gene...` etc. Hit the `Save Correction` button.
                    4. Refresh the page!""")
        
        gr.Markdown("## Sample Queries")
        gr.Markdown("""1. find the gene ontology info for CDK2
                    2. give me the reactome pathway information about the mouse gene cdk2
                    3. give me uniprot id of the gene with the entrez gene id of 1017
                    4. What are the symbol and Ensembl gene ID for genes in species 9669 with a symbol starting with 'LOC123388108'?""")
        
        user_input = gr.Textbox(label="Enter your query")
        llm_response = gr.Textbox(label="LLM Response", interactive=False)
        url_display = gr.Markdown(label="Link")
        user_correction = gr.Textbox(label="User-corrected Response (if needed), like `/v3/query...` or `/v3/gene...` etc")

        submit_button = gr.Button("Generate Response")
        save_button = gr.Button("Save Correction")
        # save_message = gr.Textbox(label="Status Message", interactive=False)

        # Define actions on button click
        user_input.submit(main_function, inputs=[user_input], outputs=[llm_response, url_display, user_correction])
        submit_button.click(main_function, inputs=[user_input], outputs=[llm_response, url_display, user_correction])
        save_button.click(handle_interaction, inputs=[user_input, llm_response, user_correction])

    demo.queue().launch(server_name=args.host,
                        server_port=args.port)
