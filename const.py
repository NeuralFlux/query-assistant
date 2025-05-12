OUTPUT_SCHEMA = {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "imperative": {"type": "string"},
                                    "question": {"type": "string"}
                                },
                                "required": ["query", "imperative", "question"]
                            }
                        }
                    },
                    "required": ["instructions"]
                }