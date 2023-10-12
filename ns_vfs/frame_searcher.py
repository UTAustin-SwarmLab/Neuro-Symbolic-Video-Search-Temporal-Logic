from typing import List, Union

from ns_vfs.common.ltl_utility import verification_result_eval
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.processor.video_processor import (
    VideoProcessor,
)
from ns_vfs.verification import check_automaton
from ns_vfs.video_to_automaton import VideotoAutomaton


class FrameSearcher:
    def __init__(
        self,
        video_automata_builder: VideotoAutomaton,
        video_processor: VideoProcessor,
    ):
        self._video_automata_builder = video_automata_builder
        self._video_processor = video_processor

    def get_propositional_confidence_per_frame(
        self,
        frame: Frame,
        proposition_set: List[str],
        benchmark_frame_label: Union[str, List[str], None] = None,
        manual_confidence_probability=None,
    ):
        for proposition in proposition_set:
            (
                propositional_confidence,
                detected_obj,
                annotated_image,
            ) = self._video_automata_builder.calculate_confidence_of_proposition(
                proposition=proposition,
                frame_img=frame.frame_image,
                save_annotation=self._video_automata_builder._save_annotation,
            )
            if benchmark_frame_label is not None and isinstance(manual_confidence_probability, float):
                if isinstance(benchmark_frame_label, list):
                    if proposition in benchmark_frame_label:
                        propositional_confidence = manual_confidence_probability
                    else:
                        propositional_confidence = 1 - manual_confidence_probability
                elif isinstance(benchmark_frame_label, str):
                    if benchmark_frame_label == proposition:
                        propositional_confidence = manual_confidence_probability
                    else:
                        propositional_confidence = 1 - manual_confidence_probability
            frame.object_detection[str(proposition)] = detected_obj
            frame.propositional_probability[str(proposition)] = propositional_confidence
            if annotated_image is not None:
                frame.annotated_image[str(proposition)] = annotated_image
        return frame

    def validate_propositional_confidence(
        self, frame_set: List[Frame], frame: Frame, proposition_set, interim_confidence_set: List[List[float]]
    ):
        propositional_confidence_of_frame = frame.propositional_confidence
        proposition_condition = sum(propositional_confidence_of_frame)
        if proposition_condition > 0:
            frame_set.append(frame)
            for i in range(len(proposition_set)):
                interim_confidence_set[i].append(propositional_confidence_of_frame[i])
        return frame_set, interim_confidence_set

    def build_and_check_automaton(
        self,
        frame_set: List[Frame],
        interim_confidence_set: List[List[float]],
        proposition_set: List[str],
        ltl_formula: str,
        include_initial_state=False,
        verbose=False,
        is_filter=False,
    ):
        # if "!" in ltl_formula:
        #     reverse_search = True
        # else:
        #     reverse_search = False
        states, transitions = self._video_automata_builder.build_automaton(
            frame_set, interim_confidence_set, include_initial_state=include_initial_state
        )
        verification_result = check_automaton(
            transitions=transitions,
            states=states,
            proposition_set=proposition_set,
            ltl_formula=ltl_formula,
            verbose=verbose,
            is_filter=is_filter,
        )
        result = verification_result_eval(verification_result)

        if result == "PartialTrue":
            # frame_set = [frame_set[-1]]
            result = True

        return result, frame_set

    def update_frame_of_interest(
        self, frame_set: List[Frame], frame_of_interest: FramesofInterest
    ) -> FramesofInterest:
        if len(frame_set) > 1:
            frame_interval = list()
            for frame in frame_set:
                frame_interval.append(frame.frame_index)
                frame_of_interest.frame_idx_to_real_idx[frame.frame_index] = frame.real_frame_idx
                frame_of_interest.frame_images.append(frame.frame_image)
                frame_of_interest.save_annotated_images(frame.annotated_image)
            frame_of_interest.foi_list.append(frame_interval)
        else:
            for frame in frame_set:
                frame_of_interest.foi_list.append([frame.frame_index])
                frame_of_interest.frame_idx_to_real_idx[frame.frame_index] = frame.real_frame_idx
                frame_of_interest.frame_images.append(frame.frame_image)
                frame_of_interest.save_annotated_images(frame.annotated_image)

        return frame_of_interest

    def search(self):
        return self._video_processor.process_and_get_frame_of_interest(
            ltl_formula=self._video_automata_builder.ltl_formula,
            proposition_set=self._video_automata_builder.proposition_set,
            get_propositional_confidence_per_frame=self.get_propositional_confidence_per_frame,
            validate_propositional_confidence=self.validate_propositional_confidence,
            build_and_check_automaton=self.build_and_check_automaton,
            update_frame_of_interest=self.update_frame_of_interest,
        )
