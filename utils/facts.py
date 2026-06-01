class Fact():
    def __init__(self, prompt, subject, true_target, new_target, relation_id, paraphrases : list, neighborhoods : list, attributes : list, generations : list, case_id, pararel_idx, source_ds : str):
        self.prompt = prompt
        self.subject = subject
        self.actual_prompt = prompt.format(subject)
        self.true_target = true_target
        self.new_target = new_target
        self.relation_id = relation_id
        self.paraphrases = paraphrases
        self.neighborhoods = neighborhoods
        self.attributes = attributes
        self.generations = generations
        self.case_id = case_id
        self.pararel_idx = pararel_idx
        self.source_ds = source_ds
        self.edit_documents = dict()
        

        