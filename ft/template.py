from typing import Mapping, Optional, Dict, Any
import inspect

from torchtune.data import InstructTemplate

with open("/home/atubati/query_assistant/gene_query_docs.txt") as fd:
    # ignore curly braces in docs and schema for formatting
    docs = fd.read().replace("{", "{{").replace("}", "}}")

with open("/home/atubati/query_assistant/data/original/compact_desc_with_context.csv") as fd:
    schema = fd.read().replace("{", "{{").replace("}", "}}")

class QueryAssistantTemplate(InstructTemplate):
    # Define the template as string with {} as placeholders for data columns
    template = inspect.cleandoc("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant who write API calls for the MyGene database. Use the documentation and field description as reference to complete the user-given task.\nDocs: {docs}.\nSchema: {schema}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {{instruction}}. Write an API call.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
    """).format(docs=docs, schema=schema)

    # Implement this method
    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        formatted_str = cls.template.format(instruction=sample["instruction"])
        return formatted_str
