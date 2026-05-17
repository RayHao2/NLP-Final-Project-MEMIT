import json


"""
COUTNERFACT examples
    {
        "case_id": 0,
        "pararel_idx": 2796,
        "requested_rewrite": {
            "prompt": "The mother tongue of {} is", (The fact were changing)
            "relation_id": "P103",
            "target_new": {"str": "English", "id": "Q1860"}, (The value were changing the fact to)
            "target_true": {"str": "French", "id": "Q150"}, (The true fact)
            "subject": "Danielle Darrieux" (goes in the {} for prompt)
        },
        "paraphrase_prompts": ["...", "..."], (Same fact as prompt but asked differently)
        "neighborhood_prompts": ["...", "..."], (Same types of questions but different fact, value is target_true (we don't want to change these))
        "attribute_prompts": ["..."], (Same types of questions but different fact, value is target_new (we don't want to change these))
        "generation_prompts": ["..."] (prompts to ensure the LLM still works)
    }
"""
def load_counterfact(path):
    with open(path) as f:
        raw = json.load(f)
    for entry in raw:
        rw = entry["requested_rerwite"]
        subject = rw['subject']
        prompt = rw['prompt']
        true_target = rw['target_true']
        relation_id = rw['relation_id']
        
        new_target = rw['target_new']
        actual_prompt = prompt.format(subject)
        paraphrases = [p for p in entry.get("paraphrase_prompts", [])]
        neighborhoods = [p for p in entry.get("neighborhood_prompts", [])]
        attributes = [p for p in entry.get("attribute_prompts", [])]
        generations = [p for p in entry.get("generation_prompts", [])]
        case_id = entry['case_id']
        pararel_idx = entry['pararel_idx']



"""
zsRE examples
    {
        "subject": "...",
        "src": "what is X of Y?",
        "pred": "...",          # model's original prediction (not the truth)
        "rephrase": "...",       # paraphrased question
        "alt": "...",            # the new (alternative) target
        "answers": ["..."],      # original gold answers
        "loc": "nq question ...", # locality query
        "loc_ans": "..."         # locality answer
    }
"""