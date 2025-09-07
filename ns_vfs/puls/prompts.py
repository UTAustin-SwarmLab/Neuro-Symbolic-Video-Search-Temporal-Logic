def find_prompt(prompt):
    full_prompt = f"""
You are an intelligent agent designed to extract structured representations from video description prompts. You will operate in two stages: (1) proposition extraction and (2) TL specification generation.

Stage 1: Proposition Extraction

Given an input prompt describing a sequence in a video, extract the atomic propositions that describe the underlying events or facts explicity referenced. These propositions should describe the combined semantics of object-action or object-object relationships stated in the prompt — avoid making assumptions or inferring any additional events. Avoid TL keywords such as 'and', 'or', 'not', 'until'.
For example, given the prompt "A man is eating until he gets up", the correct propositions are ["man eats", "man gets up"].

Stage 2: TL Specification Generation

Using only the list of the propositions extracted in Stage 1, generate a single Temporal Logic (TL) specification that catpures the sequence of logical structure implied by the initial prompt. 

Rules:
- The formula must use each proposition **exactly once**
- Use only the TL operators: `AND`, `OR`, `NOT`, `UNTIL`
- Do **not** infer new events or rephrase propositions.
- The formula should reflect the temporal or logical relationships between the propositions in a way that makes semantic sense.

**Examples**

Example 1: "A child is playing with his kite and running around before he unfortunately falls down"
Output:
{{
  "proposition": ["child plays with kite", "child runs around", "child falls"],
  "specification": "(child plays with kite AND child runs around) UNTIL child falls"
}}

Example 2: "In a dimly lit room, two robots stand silently. Suddenely, either the red robot starts blinking or the green robot does not turn off."
Output:
{{
  "proposition": ["robots stand silently", "red robot starts blinking", "green robot turns off"],
  "specification": "robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off)"
}}

Example 3: "Inside a cave, a man holds a lantern. A minute after, he suddenely sees a dragon."
Output:
{{
  "proposition": ["man holds lantern", "man sees dragon"],
  "specification": "man holds lantern UNTIL man sees dragon"
}}

Example 6: "The girl is turning on the computer."
Output:
{{
  "proposition": ["girl turns on computer"],
  "specification": "(girl turns on computer)"
}}

**Now process the following prompt:**
Input:
{{
  "prompt": "{prompt}"
}}

Expected Output (only output the following JSON structure — nothing else):
{{
  "proposition": [...],
  "specification": "..."
}}
"""
    return full_prompt
