import regex as re
from stormpy.core import ExplicitQualitativeCheckResult


def get_not_operator_mapping(ltl_formula):
    operator_set = ["F", "G" "U"]
    not_operator_proposition_list = list()
    if len(ltl_formula.split("!")) > 2:
        # more than one "!"
        pass
    else:
        s = ltl_formula.split("!")[-1]
        match = re.search(r'"(.*?)"', s)
        if match:
            prop_name = match.group(1).strip()
            if prop_name not in operator_set:
                not_operator_proposition_list.append(prop_name)
        else:
            raise ValueError("No ! match")
    return not_operator_proposition_list


def verification_result_eval(verification_result: ExplicitQualitativeCheckResult):
    # string result is "true" when is absolutely true
    # but it returns "true, false" when we have some true and false
    verification_result_str = str(verification_result)
    string_result = verification_result_str.split("{")[-1].split("}")[0]
    if len(string_result) == 4:
        if string_result[0] == "t":  # 0,6
            result = True
    elif len(string_result) > 5:
        # "true, false" -> some true and some false
        result = "PartialTrue"
    else:
        result = False
    return result
