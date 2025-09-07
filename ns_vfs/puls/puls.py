from ns_vfs.puls.llm import *
from ns_vfs.puls.prompts import *
from openai import OpenAI
import json
import os
import re

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def process_specification(specification, propositions):
    new_propositions = []
    for prop in propositions:
        prop_cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", prop)
        prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
        prop_cleaned = prop_cleaned.replace("'", "")
        new_propositions.append(prop_cleaned)

    for original, new in zip(propositions, new_propositions):
        specification = specification.replace(original, f'"{new}"')

    replacements = {
        "AND": "&",
        "OR": "|",
        "UNTIL": "U",
        "ALWAYS": "G",
        "EVENTUALLY": "F",
        "NOT": "!"
    }
    for word, symbol in replacements.items():
        specification = specification.replace(word, symbol)

    return new_propositions, specification

def PULS(query, openai_save_path, openai_model="o1-mini", openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    client = OpenAI()
    llm = LLM(client, save_dir=openai_save_path)

    full_prompt = find_prompt(query)
    llm_output = llm.prompt(full_prompt, openai_model)
    parsed = clean_and_parse_json(llm_output)

    final_output = {}

    cleaned_props, processed_spec = process_specification(parsed["specification"], parsed["proposition"])
    final_output["proposition"] = cleaned_props
    final_output["specification"] = processed_spec

    saved_path = llm.save_history()
    final_output["saved_path"] = saved_path

    return final_output
