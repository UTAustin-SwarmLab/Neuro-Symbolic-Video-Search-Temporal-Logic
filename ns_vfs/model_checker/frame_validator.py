import re
import enum

from ns_vfs.video.frame import VideoFrame

class SymbolicFilterRule(enum.Enum):
    AVOID_PROPS = "avoid"
    ASSOCIATED_PROPS = "associated"

class FrameValidator:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self.threshold_of_probability = threshold_of_probability

        ltl_formula = ltl_formula[ltl_formula.find('[') + 1:ltl_formula.rfind(']')]
        if " U " in ltl_formula:
            rule_1 = self.get_symbolic_rule_from_ltl_formula(ltl_formula.split(" U ")[0])
            rule_2 = self.get_symbolic_rule_from_ltl_formula(ltl_formula.split(" U ")[1])
            self.symbolic_verification_rule = {
                SymbolicFilterRule.ASSOCIATED_PROPS: rule_1[SymbolicFilterRule.ASSOCIATED_PROPS] + rule_2[SymbolicFilterRule.ASSOCIATED_PROPS],
                SymbolicFilterRule.AVOID_PROPS: rule_1[SymbolicFilterRule.AVOID_PROPS] or rule_2[SymbolicFilterRule.AVOID_PROPS],
            }
        else:
            self.symbolic_verification_rule = self.get_symbolic_rule_from_ltl_formula(ltl_formula)

        print(f"\nSpecification: {ltl_formula}")
        print(f"avoid_props: {self.symbolic_verification_rule[SymbolicFilterRule.AVOID_PROPS]}")
        print(f"associated_props: {self.symbolic_verification_rule[SymbolicFilterRule.ASSOCIATED_PROPS]}")

    def validate_frame(
        self,
        frame: VideoFrame,
    ):
        """Validate frame."""
        thresholded_objects = frame.thresholded_detected_objects(self.threshold_of_probability)
        print(thresholded_objects)
        if len(thresholded_objects) > 0:
            return self.symbolic_verification(frame)
        else:
            return False

    def symbolic_verification(self, frame: VideoFrame):
        """Symbolic verification."""
        avoid_props = self.symbolic_verification_rule.get(SymbolicFilterRule.AVOID_PROPS)
        if avoid_props:
            for prop in frame.object_of_interest.keys():
                if frame.object_of_interest[prop].get_detected_probability() >= self.threshold_of_probability and prop in avoid_props: # detected but also in avoid_props
                    return False

        associated_props = self.symbolic_verification_rule.get(SymbolicFilterRule.ASSOCIATED_PROPS)
        for group in associated_props:
            bad = 0
            total = 0
            for prop in group:
                total += 1
                if frame.object_of_interest[prop].get_detected_probability() < self.threshold_of_probability:
                    bad += 1
            if total > 2 * bad:
                return True
        return False

    def get_symbolic_rule_from_ltl_formula(self, ltl_formula: str) -> dict:
        symbolic_verification_rule = {}

        if "!" in ltl_formula:
            match = re.search(r'(?<!\w)!\s*(?:\((.*?)\)|([^\s\)]+))', ltl_formula)
            avoid_tl = (match.group(1) or match.group(2)).strip()
            symbolic_verification_rule[SymbolicFilterRule.AVOID_PROPS] = avoid_tl
        else:
            symbolic_verification_rule[SymbolicFilterRule.AVOID_PROPS] = None

        ltl_formula = re.sub(r"[!GF]", "", ltl_formula.strip())
        while ltl_formula.startswith("(") and ltl_formula.endswith(")") and ltl_formula.count("(") == ltl_formula.count(")"):
            ltl_formula = ltl_formula[1:-1].strip()

        split_and_clean = lambda expr: [re.sub(r"[()]", "", p).strip() for p in re.split(r"\s*&\s*", expr) if p.strip()]

        match = re.search(r'\b( U |F)\b', ltl_formula)
        if match:
            idx = match.start()
            associated = [split_and_clean(ltl_formula[:idx]), split_and_clean(ltl_formula[idx + len(match.group(1)):])]
        else:
            associated = [split_and_clean(ltl_formula)]
        associated = [[s.strip('"') for s in sublist] for sublist in associated]
        symbolic_verification_rule[SymbolicFilterRule.ASSOCIATED_PROPS] = associated

        return symbolic_verification_rule

