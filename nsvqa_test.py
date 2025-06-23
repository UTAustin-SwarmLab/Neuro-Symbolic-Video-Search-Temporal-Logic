from __future__ import annotations

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.validator import FrameValidator


def main():
    ltl_formula = 'P>=0.80 [ ("a" & "b") U "c" ]'
    proposition_set = ['a', 'b', 'c']
    data = [
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 0   ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 1   ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 2   ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 3   ___
        {'a': 0.9, 'b': 0.1, 'c': 0.1}, # 4   a__
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 5   ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 6   ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 7   ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 8   ab_
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 9   ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 10  ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 11  ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 12  ___
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 13  ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 14  ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 15  ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.1}, # 16  ab_
        {'a': 0.9, 'b': 0.9, 'c': 0.9}, # 17  abc
        {'a': 0.9, 'b': 0.9, 'c': 0.9}, # 18  abc
        {'a': 0.9, 'b': 0.9, 'c': 0.9}, # 19  abc
        {'a': 0.9, 'b': 0.9, 'c': 0.9}, # 20  abc
        {'a': 0.9, 'b': 0.9, 'c': 0.9}, # 21  abc
        {'a': 0.1, 'b': 0.1, 'c': 0.9}, # 22  __c
        {'a': 0.1, 'b': 0.1, 'c': 0.9}, # 23  __c
        {'a': 0.1, 'b': 0.1, 'c': 0.9}, # 24  __c
        {'a': 0.1, 'b': 0.1, 'c': 0.9}, # 25  __c
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 26  ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 27  ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}, # 28  ___
        {'a': 0.1, 'b': 0.1, 'c': 0.1}  # 29
    ]

    frame_validator = FrameValidator(ltl_formula=ltl_formula)
    automaton = ProbabilisticAutomaton(
        include_initial_state=False, proposition_set=proposition_set
    )
    automaton.set_up(proposition_set=proposition_set)
    model_checker = StormModelChecker(
        proposition_set=proposition_set, ltl_formula=ltl_formula
    )
    frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)

    for i in range(len(data)):
        object_of_interest = {}
        for detection in data[i]:
            object_of_interest[detection] = DetectedObject(
                name=detection,
                is_detected=(data[i][detection] > 0.5),
                probability_of_all_obj = [data[i][detection]],
                confidence_of_all_obj = [data[i][detection]],
                number_of_detection=1,
                confidence=data[i][detection],
                probability=data[i][detection],
            )

        frame = Frame(
            frame_idx=i,
            object_of_interest=object_of_interest,
        )
        # 1. frame validation
        if frame_validator.validate_frame(frame=frame):
            # 2. dynamic automaton construction
            automaton.add_frame_to_automaton(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            # 3. model checking
            model_checking_result = model_checker.check_automaton(
                transitions=automaton.transitions,
                states=automaton.states,
                verbose=False,
                is_filter=False,
            )
            if model_checking_result:
                # specification satisfied
                frame_of_interest.flush_frame_buffer()
                automaton.reset()
                print(frame_of_interest.foi_list)


if __name__ == "__main__":
    main()
