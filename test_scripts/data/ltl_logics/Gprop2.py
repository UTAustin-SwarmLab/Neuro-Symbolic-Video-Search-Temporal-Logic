"""Multiple Proposions are set but it only returns to prop2 because of "G" temporal operator.
When we looking for a single proposition from multiple propositions "G" is encouraged to use.
"""

LABEL_SET = label_set = [
    "prop1",
    "prop1",
    "random_label",
    "random_label",
    "prop2",
]
PROPOSITION_SET = ["prop1", "prop2"]
LTL_FORMULA = 'P>=0.90 [G "prop2"]'
FOI = [[4]]
