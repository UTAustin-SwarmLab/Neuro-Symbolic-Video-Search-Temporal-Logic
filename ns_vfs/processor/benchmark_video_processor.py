from __future__ import annotations

from ns_vfs.common.frame_grouping import combine_consecutive_lists
from ns_vfs.data.frame import BenchmarkLTLFrame, Frame, FramesofInterest
from ns_vfs.processor.video_processor import VideoFrameProcessor
from ns_vfs.verification import check_automaton


class BenchmarkVideoFrameProcessor(VideoFrameProcessor):
    """Benchmark Video Frame Window Processor.

    It processes video frame, build automaton and verification
    concurretnly
    """

    def __init__(
        self, video_path: str, artifact_dir: str, manual_confidence_probability: float | None
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            artifact_dir (str): Path to artifact directory.
        """
        try:
            self._manual_confidence_probability = float(manual_confidence_probability)
        except ValueError:
            self._manual_confidence_probability = None
        self.benchmark_image_frames: BenchmarkLTLFrame = self.import_video(video_path)
        super().__init__(self, video_path, artifact_dir, is_auto_import=False)

    def import_video(self, video_path: str) -> None:
        """Load and return an instance from a pickle file."""
        import pickle

        with open(video_path, "rb") as f:
            return pickle.load(f)

    def process_and_get_frame_of_interest(
        self,
        proposition_set: list = None,
        ltl_formula: str = "",
        get_propositional_confidence_per_frame: callable = None,
        validate_propositional_confidence: callable = None,
        build_and_check_automaton: callable = None,
        update_frame_of_interest: callable = None,
    ) -> FramesofInterest:
        frame_idx = 0
        total_num_frames = self.benchmark_image_frames.number_of_frame
        frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
        frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)

        for idx in range(total_num_frames):
            frame_img = self.benchmark_image_frames.images_of_frames[idx]
            # Initialize frame
            # --------------- frame image imported above --------------- #
            frame: Frame = Frame(
                frame_index=frame_idx,
                frame_image=frame_img,
            )
            frame = get_propositional_confidence_per_frame(
                frame=frame,
                proposition_set=proposition_set,
                benchmark_frame_label=self.benchmark_image_frames.labels_of_frames[idx],
                manual_confidence_probability=self._manual_confidence_probability,
            )

            frame_set, interim_confidence_set = validate_propositional_confidence(
                frame_set=frame_set,
                frame=frame,
                proposition_set=proposition_set,
                interim_confidence_set=interim_confidence_set,
            )

            if len(frame_set) > 0:  # propositions in frame
                result, frame_set = build_and_check_automaton(
                    frame_set=frame_set,
                    interim_confidence_set=interim_confidence_set,
                    include_initial_state=False,
                    proposition_set=proposition_set,
                    ltl_formula=ltl_formula,
                    verbose=False,
                )
                # if we uncomment below prop1 U porp2 won't work, need validation
                # if result == "False":  # propositions in frame but doesn't meet the initial ltl condition
                # frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
                # else:
                #   # Some Code
                if result:
                    # 2.1 Save result
                    frame_of_interest = update_frame_of_interest(
                        frame_set=frame_set, frame_of_interest=frame_of_interest
                    )
                    # 2.2 Reset frame set
                    frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
            frame_idx += 1
        frame_of_interest.reorder_frame_of_interest()
        return frame_of_interest

    def get_frame_of_interest(
        self,
        proposition_set: list,
        calculate_propositional_confidence: callable,
        build_automaton: callable,
        ltl_formula: str,
        frame_duration_sec: int = 2,
        frame_scale: int | None = None,
        sliding_window_size: int = 5,
        save_image: bool = False,
        is_annotation: bool = False,
        manual_confidence_probability: float | None = None,
        verbose: bool = False,
    ) -> None:
        """Get frame of interest from benchmark.

        It processes video frame, build automaton and verification
        concurrently by sliding window.

        Args:
            proposition_set (list): List of propositions.
            calculate_propositional_confidence (callable): Calculate propositional confidence.
            build_automaton (callable): Build automaton.
            frame_duration_sec (int, optional): Second per frame. Defaults to 2.
            frame_scale (int | None, optional): Scale of frame. Defaults to None.
            sliding_window_size (int, optional): Size of sliding window. Defaults to 5.
            save_image (bool, optional): Save image. Defaults to False.
            is_annotation (bool, optional): Annotate frame. Defaults to False.
            ltl_formula (str): LTL formula.
        """
        frame_idx = 0
        total_num_frames = self.benchmark_image_frames.number_of_frame
        temp_frame_set = list()
        temp_confidence_set = [[] for _ in range(len(proposition_set))]
        frame_of_interest = list()
        reverse_search = False
        if "!" in ltl_formula:
            reverse_search = True

        for idx in range(total_num_frames):
            frame_img = self.benchmark_image_frames.images_of_frames[idx]
            # Initialize frame
            frame: Frame = Frame(
                frame_index=frame_idx,
                frame_image=frame_img,
            )
            # --------------- frame image imported above --------------- #
            # somefunction(frame, proposition_set)
            # Calculate propositional confidence
            for proposition in proposition_set:
                propositional_confidence, detected_obj, _ = calculate_propositional_confidence(
                    proposition=proposition,
                    frame_img=frame_img,
                    save_annotation=is_annotation,
                )
                # --------------------------------------------------- #
                if manual_confidence_probability is not None:
                    if self.benchmark_image_frames.labels_of_frames[idx] == proposition:
                        propositional_confidence = manual_confidence_probability
                    else:
                        propositional_confidence = 1 - manual_confidence_probability
                # --------------------------------------------------- #
                frame.object_detection[str(proposition)] = detected_obj
                frame.propositional_probability[str(proposition)] = propositional_confidence

            # 1. If Propositional Confidence is True
            propositional_confidence_of_frame = frame.propositional_confidence
            proposition_condition = sum(propositional_confidence_of_frame)
            if proposition_condition > 0:
                temp_frame_set.append(frame)
                for i in range(len(proposition_set)):
                    temp_confidence_set[i].append(propositional_confidence_of_frame[i])

                frame_set = temp_frame_set.copy()
                states, transitions = build_automaton(
                    frame_set, temp_confidence_set, include_initial_state=False
                )
                verification_result = check_automaton(
                    transitions=transitions,
                    states=states,
                    proposition_set=proposition_set,
                    ltl_formula=ltl_formula,
                    verbose=verbose,
                )
                verification_result_str = str(verification_result)
                string_result = verification_result_str.split("{")[-1].split("}")[0]
                if string_result[0] == "t":
                    result = True
                else:
                    result = False
                    if reverse_search:
                        result = True

                # 2. If Verification Result is True
                if result:
                    # 2.1 Save result
                    if len(temp_frame_set) > 1:
                        frame_of_interest.append([frame.frame_index for frame in temp_frame_set])
                    else:
                        for frame in temp_frame_set:
                            frame_of_interest.append([frame.frame_index])

                    # 2.2 Reset frame set
                    temp_frame_set = list()
                    temp_confidence_set = [[] for _ in range(len(proposition_set))]

            frame_idx += 1
        if reverse_search:
            flattened_list = [item for sublist in frame_of_interest for item in sublist]
            frame_of_interest = [x for x in range(total_num_frames) if x not in flattened_list]
        frame_of_interest = combine_consecutive_lists(frame_of_interest)
        return frame_of_interest
