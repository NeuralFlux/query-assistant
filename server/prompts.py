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