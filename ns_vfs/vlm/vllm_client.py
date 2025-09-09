import base64

from openai import OpenAI
import numpy as np
import math
import cv2


from ns_vfs.vlm.obj import DetectedObject

class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8005/v1",
        # model="OpenGVLab/InternVL2_5-8B",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    # def _encode_frame(self, frame):
    #     return base64.b64encode(frame.tobytes()).decode("utf-8")
    def _encode_frame(self, frame):
        # Encode a uint8 numpy array (image) as a JPEG and then base64 encode it.
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_description: str,
        threshold: float
    ) -> DetectedObject:

        parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
        prompt = rf"Is there a {scene_description} present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"

        # Encode each frame.
        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]

        # Build the user message: a text prompt plus one image for each frame.
        user_content = [
            {
                "type": "text",
                "text": f"The following is the sequence of images",
            }
        ]
        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        # Create a chat completion request.
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        content = chat_response.choices[0].message.content
        is_detected = "yes" in content.lower()

        # Retrieve the list of TopLogprob objects.
        top_logprobs_list = chat_response.choices[0].logprobs.content[0].top_logprobs

        # Build a mapping from token text (stripped) to its probability.
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        # Extract probabilities for "Yes" and "No"
        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        # Compute the normalized probability for "Yes": p_yes / (p_yes + p_no)
        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            raise ValueError("No probabilities for 'Yes' or 'No' found in the response.")

        # print(f"Is detected: {is_detected}")
        # print(f"Confidence: {confidence:.3f}")


        probability = self.calibrate(confidence=confidence, false_threshold=threshold)

        return DetectedObject(
            name=scene_description,
            is_detected=is_detected,
            confidence=round(confidence, 3),
            probability=round(probability, 3)
        )

    def calibrate(
        self,
        confidence: float,
        true_threshold=0.95,
        false_threshold=0.40,
        target_conf=0.60,
        target_prob=0.78,
        p_min=0.01,
        p_max=0.99,
        steepness_factor=0.7,
    ) -> float:
        """Map confidence to probability using a sigmoid function with adjustable steepness.

        Args:
            confidence: Input confidence score
            true_threshold: Upper threshold
            false_threshold: Lower threshold
            target_conf: Target confidence point
            target_prob: Target probability value
            p_min: Minimum probability
            p_max: Maximum probability
            steepness_factor: Controls curve steepness (0-1, lower = less steep)
        """
        if confidence <= false_threshold:
            return p_min

        if confidence >= true_threshold:
            return p_max

        # Calculate parameters to ensure target_conf maps to target_prob
        # For a sigmoid function: f(x) = L / (1 + e^(-k(x-x0)))

        # First, normalize the target point
        x_norm = (target_conf - false_threshold) / (true_threshold - false_threshold)
        y_norm = (target_prob - p_min) / (p_max - p_min)

        # Find x0 (midpoint) and k (steepness) to satisfy our target point
        x0 = 0.30  # Midpoint of normalized range

        # Calculate base k value to hit the target point
        base_k = -np.log(1 / y_norm - 1) / (x_norm - x0)

        # Apply steepness factor (lower = less steep)
        k = base_k * steepness_factor

        # With reduced steepness, we need to adjust x0 to still hit the target point
        # Solve for new x0: y = 1/(1+e^(-k(x-x0))) => x0 = x + ln(1/y-1)/k
        adjusted_x0 = x_norm + np.log(1 / y_norm - 1) / k

        # Apply the sigmoid with our calculated parameters
        x_scaled = (confidence - false_threshold) / (true_threshold - false_threshold)
        sigmoid_value = 1 / (1 + np.exp(-k * (x_scaled - adjusted_x0)))

        # Ensure we still hit exactly p_min and p_max at the thresholds
        # by rescaling the output slightly
        min_val = 1 / (1 + np.exp(-k * (0 - adjusted_x0)))
        max_val = 1 / (1 + np.exp(-k * (1 - adjusted_x0)))

        # Normalize the output
        normalized = (sigmoid_value - min_val) / (max_val - min_val)

        return p_min + normalized * (p_max - p_min)

