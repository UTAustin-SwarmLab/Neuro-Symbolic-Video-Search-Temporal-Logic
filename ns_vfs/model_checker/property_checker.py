from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.model_checker.frame_validator import FrameValidator

class PropertyChecker:
    def __init__(self, proposition, specification, model_type, tl_satisfaction_threshold, detection_threshold):
        self.proposition = proposition
        self.specification = self.generate_specification(specification)
        self.model_type = model_type
        self.tl_satisfaction_threshold = tl_satisfaction_threshold,
        self.detection_threshold = detection_threshold

        self.model_checker = StormModelChecker(
            proposition_set=self.proposition,
            ltl_formula=self.specification
        )
        self.frame_validator = FrameValidator(
            ltl_formula=self.specification,
            threshold_of_probability=self.detection_threshold
        )

    def generate_specification(self, specification_raw):
        return f'P>={self.tl_satisfaction_threshold:.2f} [ {specification_raw} ]'

    def validate_frame(self, frame_of_interest):
        return self.frame_validator.validate_frame(frame_of_interest)

    def check_automaton(self, automaton):
        return self.model_checker.check_automaton(
            transitions=automaton.transitions,
            states=automaton.states,
            model_type=self.model_type
        )

    def validate_tl_specification(self, specification):
        return self.model_checker.validate_tl_specification(specification)



