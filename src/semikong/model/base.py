from huggingface_hub import login

class SEMIKONG():
    def __init__(
            self,
            model_name: str,
            huggingface_token: str
    ):
        self.model_name = model_name
        self.huggingface_token = huggingface_token
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        login(self.huggingface_token, add_to_git_credential=False, write_permission=False)

        try:
            config = AutoConfig.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config = config, trust_remote_code = True)
        except Exception:
            raise("The model name should be pentagoniac/SEMIKONG-70B or pentagoniac/SEMIKONG-8B")

    def inference(self, prompt, **generating_args):
        input_ids = self.tokenizer(prompt, return_tensors=True).input_ids
        generate_output = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=generating_args["max_length"]
        )

        return self.tokenizer.batch_decode(generate_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def training(self):
        pass

    def evaluate(self):
        pass