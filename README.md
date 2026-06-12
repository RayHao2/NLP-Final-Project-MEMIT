# NLP Final Project: MEMIT on Qwen-R1

This repository contains our project on updating factual knowledge in large
language models. The repository includes work on retrieval-augmented generation
under `rag/` and our extension of
[MEMIT](https://github.com/kmeng01/memit) under `memit/`.

This README focuses on the **RAG** experiments and **MEMIT/Qwen section** of the project. We adapted the
original MEMIT implementation, which primarily targets GPT-style models, to run
on [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
We then evaluated the adapted implementation on CounterFact and zsRE.

## Recommended Reading Path

Rag Experiments:
The code for these experiments is under the rag directory. Please note that we don't push any large files to this repo such as our actual edit documents, index files, and so on.

1. [`rag/generate_edits.py`](rag/generate_edits.py)
   The script used to generate the edit documents for retrieval.
2. [`rag/flag_edits.py`](rag/flag_edits.py)
   This script is used to flag any edit documents that didn't generate correctly.
3. [`rag/wiki_documents.py`](rag/wiki_documents.py)
   This script is used to gather wiki documents to create the noisy indexes. 
5. [`rag/database/`](rag/database/)
   This directory is used to store the scrips used to create the indexes. Both naive and e* methods. bge_retriever.py is used for the indexer, estimate_cq.py is a one time script to create C_q, indexer.py and indexer_noisy.py are used for the actual retrieval, and estar.py is used to create the e* embeddings. Note that we don't include the actual indexer files as they are very large. 
6. [`rag/eval/`](rag/eval/)
   This directory is used for the actual evaluation of CounterFact and zsRE. gptj.py is used to load the GPT-J model, eval_gptj.py is the main eval function. A run can be submitted with run_cf/zsre.sh files.


Reasoning Qwen Experiments:
The professor does not need to read the complete upstream MEMIT codebase. The
following files contain the most important parts of our contribution.

1. [`memit/NLP-Project/smoke_memit_counterfact.py`](memit/NLP-Project/smoke_memit_counterfact.py)  
   A small, readable end-to-end example that loads CounterFact, applies MEMIT,
   and records generations and target probabilities before and after editing.

2. [`memit/hparams/MEMIT/deepseek-r1-distill-qwen-1.5b.json`](memit/hparams/MEMIT/deepseek-r1-distill-qwen-1.5b.json)  
   The primary Qwen-R1 MEMIT configuration, including the edited layers and
   Qwen module paths.

3. [`memit/memit/compute_z.py`](memit/memit/compute_z.py) and
   [`memit/rome/repr_tools.py`](memit/rome/repr_tools.py)  
   The central Qwen compatibility changes for hidden-size lookup, target
   tokenization, and subject-token positioning.

4. [`memit/experiments/py/eval_utils_zsre.py`](memit/experiments/py/eval_utils_zsre.py)  
   The corrected zsRE evaluator. This file fixes an important left-padding
   assumption that initially caused Qwen predictions to be read from the wrong
   sequence position.

5. [`memit/NLP-Project/diagnose_zsre_token_positions.py`](memit/NLP-Project/diagnose_zsre_token_positions.py)  
   The diagnostic experiment used to investigate Qwen's low initial zsRE
   scores at the token level.

6. [`memit/NLP-Project/baseline_counterfact_qwen.py`](memit/NLP-Project/baseline_counterfact_qwen.py)
   and [`memit/NLP-Project/baseline_zsre_qwen.py`](memit/NLP-Project/baseline_zsre_qwen.py)  
   Compute unedited Qwen baselines for comparison with the MEMIT results.

For the full evaluation flow, see
[`memit/experiments/evaluate.py`](memit/experiments/evaluate.py) and
[`memit/experiments/summarize.py`](memit/experiments/summarize.py).

## What We Changed

### Qwen Architecture Compatibility

The original implementation assumes GPT-specific configuration fields and
module names. Qwen-R1 instead exposes its hidden dimension through
`hidden_size`, and its editable MLP output projection is:

```text
model.layers.{layer}.mlp.down_proj
```

The relevant changes are primarily in:

- [`memit/memit/compute_z.py`](memit/memit/compute_z.py)
- [`memit/rome/layer_stats.py`](memit/rome/layer_stats.py)
- [`memit/hparams/MEMIT/deepseek-r1-distill-qwen-1.5b.json`](memit/hparams/MEMIT/deepseek-r1-distill-qwen-1.5b.json)

### Tokenization Compatibility

Qwen inserts a beginning-of-sequence token during default tokenization. This
can shift the subject and target positions used by MEMIT. We therefore tokenize
the relevant prompts and continuations with `add_special_tokens=False`.

The main changes are in:

- [`memit/memit/compute_z.py`](memit/memit/compute_z.py)
- [`memit/rome/repr_tools.py`](memit/rome/repr_tools.py)
- [`memit/dsets/zsre.py`](memit/dsets/zsre.py)

### zsRE Evaluation Correction

Qwen uses left padding. The original zsRE evaluator assumed right padding and
used the attention-mask length as the final token position. For shorter prompts,
this selected logits from the wrong position and substantially underestimated
Qwen's accuracy.

The corrected evaluator finds the actual final non-padding position for every
sequence. See:

- [`memit/experiments/py/eval_utils_zsre.py`](memit/experiments/py/eval_utils_zsre.py)
- [`memit/NLP-Project/diagnose_zsre_token_positions.py`](memit/NLP-Project/diagnose_zsre_token_positions.py)

## Main Findings

### CounterFact

On 10,000 independently evaluated CounterFact edits, the primary early-layer
Qwen configuration achieved:

| Metric | Result |
|---|---:|
| Rewrite Success | 81.37 |
| Paraphrase Success | 69.84 |
| Neighborhood Success | 24.31 |
| Combined Score | 44.28 |

MEMIT successfully injected many requested facts and generalized to
paraphrases, but Qwen showed substantially weaker neighborhood preservation.
Our layer and covariance-weight experiments investigate this trade-off.

Relevant configurations are under
[`memit/hparams/MEMIT/`](memit/hparams/MEMIT/), including early-, middle-, and
late-layer settings and covariance-weight sweeps.

### zsRE

The original evaluator produced misleadingly low Qwen results because of its
padding assumption. After correcting the evaluator, the 10,000-edit zsRE run
achieved:

| Metric | Result |
|---|---:|
| Efficacy | 28.79 |
| Paraphrase | 24.68 |
| Specificity | 21.46 |
| Harmonic-Mean Score | 24.62 |

The corrected scores remain below the published GPT-J MEMIT results, showing
that adapting MEMIT to a new model family requires both implementation
compatibility and model-specific editing analysis.

## Repository Map

```text
data/                         Shared CounterFact and zsRE datasets
memit/
  NLP-Project/                Our experiment and diagnostic scripts
  hparams/MEMIT/              Qwen configurations and ablations
  memit/                      Core MEMIT algorithm and Qwen adaptations
  rome/                       Representation and covariance-statistics utilities
  experiments/                Evaluation and summarization code
  environment-qwen.yml        Minimal Qwen Conda environment definition
  environment-qwen-exact.yml  Exact exported development environment
rag/                          Teammate's RAG implementation
```

Most files outside `memit/NLP-Project/` and the Qwen-specific modifications are
inherited from the original MEMIT repository.

## Environment Setup

Create the Qwen environment from the exported environment file:

```bash
conda env create -f memit/environment-qwen-exact.yml
conda activate memit-qwen-env
```

Before running experiments, update the machine-specific paths in
[`memit/globals.yml`](memit/globals.yml).

A CUDA-capable GPU is required. Our main experiments used a GPU with 32 GB of
memory.

## Running A Small Demonstration

Run commands from the `memit/` directory:

```bash
cd memit

TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python NLP-Project/smoke_memit_counterfact.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --hparams_fname deepseek-r1-distill-qwen-1.5b.json \
  --num_edits 1 \
  --case_offset 0 \
  --cache_dir /path/to/huggingface/cache \
  --max_new_tokens 16 \
  --use_cache
```

## Running An Evaluation

Example CounterFact evaluation:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -m experiments.evaluate \
  --alg_name MEMIT \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --hparams_fname deepseek-r1-distill-qwen-1.5b.json \
  --ds_name cf \
  --dataset_size_limit 100 \
  --num_edits 1 \
  --skip_generation_tests \
  --use_cache \
  --dir_name qwen-memit-example
```

Use `--ds_name zsre` for zsRE. Summarize a completed run with:

```bash
PYTHONPATH=. python -m experiments.summarize \
  --dir_name qwen-memit-example \
  --runs run_000 \
  --first_n_cases 100
```

The batch-running scripts used for our experiments are available under
[`memit/NLP-Project/sh_scripts/`](memit/NLP-Project/sh_scripts/).

## Evaluation Note

Our reported large evaluations apply one edit per case and restore the original
model weights before the next case. Therefore, a 10,000-case result represents
10,000 independently evaluated edits, rather than one model containing 10,000
simultaneous edits.

Large models, covariance caches, and complete result directories are excluded
from Git because of their size.

## References

- Kevin Meng et al., [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Original MEMIT repository](https://github.com/kmeng01/memit)
- [DeepSeek-R1-Distill-Qwen-1.5B model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
