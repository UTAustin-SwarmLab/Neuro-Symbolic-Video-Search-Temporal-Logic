import re

from ns_vfs.data.frame import Frame


class FrameValidator:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self._threshold_of_probability = threshold_of_probability
        self._symbolic_verification_rule = self.get_symbolic_rule_from_ltl_formula(
            ltl_formula
        )

    def validate_frame(self, frame: Frame):
        """Validate frame."""
        if frame.is_any_object_detected:
            if self.symbolic_verification(frame):
                return True
            else:
                return False
        else:
            return False

    def symbolic_verification(self, frame: Frame):
        """Symbolic verification."""
        self._symbolic_verification_rule = {}
        avoid_props = self._symbolic_verification_rule.get("avoid_proposition")
        associated_props = self._symbolic_verification_rule.get("and_associated_props")
        if avoid_props:
            for avoid_p in avoid_props:
                if frame.propositional_probability[str(avoid_p)] > 0:
                    return False
        associated_props = self._symbolic_verification_rule.get("and_associated_props")
        if associated_props:
            if not all(props in frame.detected_object for props in associated_props):
                return False
        return True

    def get_symbolic_rule_from_ltl_formula(self, ltl_formula: str) -> dict:
        symbolic_verification_rule = {}
        if "!" in ltl_formula:
            avoid_proposition = ltl_formula.split("!")[-1][1:].split('"')[
                0
            ]  # [1:-2].replace('"', "")
            symbolic_verification_rule["avoid_proposition"] = avoid_proposition
        else:
            symbolic_verification_rule["avoid_proposition"] = None

        if "&" in ltl_formula:
            # if A,B are associated by &
            # and there's no A,B in the formula
            A, B = ltl_formula.split("&")[0], ltl_formula.split("&")[1]
            A, B = re.findall(r"\"(.*?)\"", A.split("(")[-1]), re.findall(
                r"\"(.*?)\"", B.split(")")[0]
            )
            symbolic_verification_rule["and_associated_props"] = A + B

        return symbolic_verification_rule
