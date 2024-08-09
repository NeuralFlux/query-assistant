You need to generate instructions for API queries and the instruction-query pairs will be used to fine-tune a smaller LLM assistant. Generate two instructions for each of the 10 given queries, one imperative and one question-style.

Here are the requirements:
1. Strictly output a JSON.
2. Use the documentation and summary of fields in the database as reference.
3. The instructions must be in English.
4. Use diverse verbs in your responses to generalize better. This is very important!
5. A small language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
6. Each instruction must be 3 sentences long at the most.

Here are the 10 API queries line-wise: