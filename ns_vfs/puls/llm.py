import datetime
import json
import os

class LLM:
    def __init__(self, client, save_dir=""): # pass in save_dir to start saving
        self.client = client
        self.history = []
        self.save_dir = save_dir
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    def prompt(self, p, openai_model):
        user_message = {"role": "user", "content": [{"type": "text", "text": p}]}
        self.history.append(user_message)

        response = self.client.chat.completions.create(
            model=openai_model,
            messages=self.history,
            store=False,
        )
        assistant_response = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)

        return assistant_response

    def save_history(self, filename="conversation_history.json"):
        if self.save_dir == "":
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        timestamped_filename = f"{base_name}_{timestamp}{extension}"

        save_path = os.path.join(self.save_dir, timestamped_filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
            return save_path
        except Exception as e:
            print(f"Failed to save conversation history: {e}")
            return None

