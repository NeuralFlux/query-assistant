You need to generate instructions for API queries, which will be paired with each query to fine-tune a smaller LLM assistant. For each of the 10 provided queries, describe what the user is aiming to accomplish with the query. Then, create a concise instruction that would lead the smaller LLM to generate that exact query.

Please follow these requirements:

1. Output strictly in JSON format with two fields: "description" and "instruction".
2. Ensure all instructions are in clear, professional English.
3. Use varied and precise verbs in the descriptions and instructions to enhance the model's generalization abilities. This is crucial!
4. Instructions must be suitable for a small language model, so avoid any actions or requests it cannot fulfill, such as setting alarms or producing non-text outputs.
5. Limit each instruction to a single sentence.

Here are the 10 API queries, one per line: