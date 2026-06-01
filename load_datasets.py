import json
from utils.facts import Fact

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
    
    facts = []
    for entry in raw:
        rw = entry["requested_rewrite"]
        subject = rw['subject']
        prompt = rw['prompt']
        true_target = rw['target_true']['str']
        relation_id = rw['relation_id']
        new_target = rw['target_new']['str']
        paraphrases = [p for p in entry.get("paraphrase_prompts", [])]
        neighborhoods = [p for p in entry.get("neighborhood_prompts", [])]
        attributes = [p for p in entry.get("attribute_prompts", [])]
        generations = [p for p in entry.get("generation_prompts", [])]
        case_id = entry['case_id']
        pararel_idx = entry['pararel_idx']
        f = Fact(prompt, subject, true_target, new_target, relation_id, paraphrases, neighborhoods, attributes, generations, case_id, pararel_idx, "COUNTERFACT")
        facts.append(f)

    return facts



"""
zsRE examples
    {
        "subject": "...", the entity being edited 
        "src": "what is X of Y?", the canonical question
        "pred": "...",          # the pre-edit model's prediction
        "rephrase": "...",       # a single paraphrased question
        "alt": "...",            # the new (alternative) target
        "answers": ["..."],      # original gold answers
        "loc": "nq question ...", # locality query
        "loc_ans": "..."         # locality answer
    }
"""

def load_zsre(path):
    with open(path) as f:
        raw = json.load(f)
    
    facts = []
    for case_id, entry in enumerate(raw):
        subject = entry['subject']
        src = entry['src']
        prompt = src.replace(subject, "{}", 1)
        if "{}" not in prompt:
            prompt = src
        true_target = entry["pred"]
        new_target = entry["alt"]
        paraphrases = [entry["rephrase"]] if entry.get("rephrase") else []
        neighborhoods = [entry["loc"]] if entry.get("loc") else []

        f = Fact(prompt, subject, true_target, new_target, None, paraphrases, neighborhoods, None, None, case_id, None, "zsRE")
        facts.append(f)

    return facts
