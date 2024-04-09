import re

from ns_vfs.data.frame import Frame
from ns_vfs.enums.symbolic_filter_rule import SymbolicFilterRule


class FrameValidator:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self._threshold_of_probability = threshold_of_probability
        self._symbolic_verification_rule = (
            self.get_symbolic_rule_from_ltl_formula(ltl_formula)
        )

    def validate_frame(
        self, frame: Frame, is_symbolic_verification: bool = True
    ):
        """Validate frame."""
        if frame.is_any_object_detected():
            all_below_threshold = all(
                frame.object_of_interest[obj_name].probability
                < self._threshold_of_probability
                for obj_name in frame.detected_object
            )
            if all_below_threshold:
                return False
            if is_symbolic_verification:
                if self.symbolic_verification(frame):
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def symbolic_verification(self, frame: Frame):
        """Symbolic verification."""
        avoid_props = self._symbolic_verification_rule.get(
            SymbolicFilterRule.AVOID_PROPOSITION
        )
        associated_props = self._symbolic_verification_rule.get(
            SymbolicFilterRule.AND_ASSOCIATED_PROPS
        )
        if avoid_props:
            for avoid_p in avoid_props:
                if frame.propositional_probability[str(avoid_p)] > 0:
                    return False
        associated_props = self._symbolic_verification_rule.get(
            SymbolicFilterRule.AND_ASSOCIATED_PROPS
        )
        if associated_props:
            if not all(
                props in frame.detected_object for props in associated_props
            ):
                return False
        return True

    def get_symbolic_rule_from_ltl_formula(self, ltl_formula: str) -> dict:
        symbolic_verification_rule = {}
        if "!" in ltl_formula:
            avoid_proposition = ltl_formula.split("!")[-1][1:].split('"')[
                0
            ]  # [1:-2].replace('"', "")
            symbolic_verification_rule[SymbolicFilterRule.AVOID_PROPOSITION] = (
                avoid_proposition
            )
        else:
            symbolic_verification_rule[SymbolicFilterRule.AVOID_PROPOSITION] = (
                None
            )

        if "&" in ltl_formula:
            # if A,B are associated by &
            # and there's no A,B in the formula
            A, B = ltl_formula.split("&")[0], ltl_formula.split("&")[1]
            A, B = (
                re.findall(r"\"(.*?)\"", A.split("(")[-1]),
                re.findall(r"\"(.*?)\"", B.split(")")[0]),
            )
            symbolic_verification_rule[
                SymbolicFilterRule.AND_ASSOCIATED_PROPS
            ] = A + B

        return symbolic_verification_rule
