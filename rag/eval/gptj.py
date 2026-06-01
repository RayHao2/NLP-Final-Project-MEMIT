import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "EleutherAI/gpt-j-6B"


# The standard MEMIT/CounterFact prompt is a cloze completion -- a sentence
# with an implicit blank at the end that the model is supposed to fill. We
# build the RAG prompt by prepending retrieved context as plain text, then
# the cloze prompt. No instruction tuning, no "use the context to answer"
# Same for the zsRE dataset. Instead of a cloze completion we format it as a question answering template
RAG_TEMPLATE = "{context}\n\n{prompt}"
NO_CONTEXT_TEMPLATE = "{prompt}"


def format_rag_prompt(prompt: str, context: str | None) -> str:
    if context and context.strip():
        return RAG_TEMPLATE.format(context=context.strip(), prompt=prompt)
    return NO_CONTEXT_TEMPLATE.format(prompt=prompt)


class GPTJReader:
    def __init__(self, model_name = MODEL_NAME, device= None, dtype = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            if self.device.type == "cuda":
                cc = torch.cuda.get_device_capability()[0]
                dtype = torch.bfloat16 if cc >= 8 else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    @torch.no_grad()
    def _first_token_log_probs(self, input_text, candidates):
        """
        Return log P(candidate's first token | input_text) for each candidate.

        This is the standard MEMIT/ROME efficacy proxy
        """
        # Tokenize the input (context+prompt)
        inp = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        logits = self.model(**inp).logits[0, -1, :]   # (vocab,)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        out = []
        for cand in candidates:
            tok_ids = self.tokenizer.encode(" " + cand.strip(), add_special_tokens=False)
            if not tok_ids:
                out.append(float("-inf"))
                continue
            out.append(float(log_probs[tok_ids[0]].item()))
        return out


    def score_target_new(self, prompt, target_new, target_old, context= None):
        """
        Score whether GPT-J, given (optional) retrieved context, prefers
        target_new over target_old to complete the prompt.

        Returns:
            (log P(target_new_first_tok), log P(target_old_first_tok))
        """
        full_input = format_rag_prompt(prompt, context)
        lp = self._first_token_log_probs(full_input, [target_new, target_old])
        return lp[0], lp[1]

    @torch.no_grad()
    def generate(self, prompt, context = None, max_new_tokens = 30, do_sample = False):
        """
        used for fluency / consistency evaluation on CounterFact's `generation_prompts`
        """
        full_input = format_rag_prompt(prompt, context)
        inp = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=self.tokenizer.pad_token_id)
        # Strip the prompt; return only the continuation
        new_tokens = out[0, inp["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    @torch.no_grad()
    def argmax_matches(self, prompt, target, context = None, extra_tokens = 4):
        """
        zsRE-style argmax test:

        Generates len(target_tokens) + extra_tokens tokens to leave room for
        leading whitespace / BPE boundary effects, then prefix-matches.
        """
        if target is None:
            return False

        full_input = format_rag_prompt(prompt, context)
        target_tok_ids = self.tokenizer.encode(" " + target.strip(), add_special_tokens=False)
        n_new = max(len(target_tok_ids) + extra_tokens, 4)

        inp = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        out = self.model.generate(**inp, max_new_tokens=n_new, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        new_tokens = out[0, inp["input_ids"].shape[1]:]
        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        def norm(s: str) -> str:
            return " ".join(s.lower().split())

        return norm(target) in norm(continuation)