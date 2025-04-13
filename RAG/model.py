from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenChatbot:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda:1",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_prompt(self, user_input: str, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, user_input: str, max_tokens: int = 512) -> str:
        prompt = self.build_prompt(user_input)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # Remove prompt tokens from generated_ids
        cleaned_ids = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.tokenizer.batch_decode(cleaned_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    bot = QwenChatbot()
    question = "When was the fist cat show and where it was held ?"
    answer = bot.generate(question)
    print(f"Answer: {answer}")
