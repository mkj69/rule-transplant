# Trajectory: fable r100 W/2

model claude-fable-5-1, start 2026-11-08T01:24:11-06:00, end 2026-11-08T05:17:21-06:00

## Prompt

```
# Task

A′ is what A looks like after it learned one new skill. Change B so that B also has this skill, and keep B's original abilities.

# What you have (all under /work)

- `models/A/` — model A (weights and config, Hugging Face format, read-only).
- `models/A_prime/` — model A′ = A after it learned the new skill (read-only).
- `models/B/` — model B, the model you must change (read-only; write your result elsewhere).
- `tools/eval_ability.py` — the test of B's original abilities. Run `python tools/eval_ability.py <model_dir>` (about 11 minutes) or add `--quick` (about 3 minutes). Unmodified B scores 55.2 (format 62.6, knowledge 40.6).
- Python: `python` has torch, transformers, safetensors, numpy and matplotlib. The environment is read-only; to add packages use `python -m pip install --target /work/pylib <pkg>` and add `/work/pylib` to `PYTHONPATH`.

You do not get any description of the new skill, any training data for it, or any test for it.

# Rule

Weights only. You may load and run A, A′ and B, inspect their weights and activations, and edit B's weights directly (by any computation on the weights of A, A′ and B). You must not create training inputs, have A′ (or A) produce outputs for them, and then train B on those outputs. You must not train B on any data produced by A or A′.

# Budget

- 1 GPU (NVIDIA RTX A5000, 24 GB) and 8 hours of wall-clock time from the start of this session. The process is stopped at the limit.
- Disk: keep everything you write under 10 GB (the shared disk is nearly full).

# Deliverables

1. `/work/B_prime/` — the changed B in Hugging Face format (`config.json`, tokenizer files, `model.safetensors`, bf16), loadable with `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. It must have the same architecture as B.
2. `/work/METHOD.md` — what you found about the new skill and how you changed B, in a few paragraphs.

Save a first version of `/work/B_prime/` early and overwrite it when you improve it. We evaluate whatever is in `/work/B_prime/` when the time ends.
```


**Agent:** I'll start by inspecting the models, their configs, and the eval tool, then figure out how A′ differs from A.

<details><summary>step 1 · Bash · List models and configs, check GPU and disk</summary>

```
ls -la && ls -la models/* && cat models/A/config.json && echo ---- && cat models/B/config.json && echo ---- && diff models/A/config.json models/A_prime/config.json; nvidia-smi; df -h /work; date
```
</details>

<details><summary>result</summary>

```
total 16
drwxr-xr-x  5 1741768822 1007000513 4096 Nov  8 01:24 .
drwxr-xr-x 13 1741768822 1007000513  340 Nov  8 01:24 ..
drwxr-xr-x  2 1741768822 1007000513 4096 Nov  8 01:24 .home
drwxr-xr-x  5 1741768822 1007000513 4096 Nov  8 01:24 models
drwxr-xr-x  2 1741768822 1007000513 4096 Nov  8 01:24 tools
models/A:
total 1175444
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  7 15:10 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 01:24 ..
-rw-r--r-- 1 1741768822 1007000513        727 Nov  7 15:10 config.json
-rw-r--r-- 1 1741768822 1007000513        138 Nov  7 15:10 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  7 15:10 merges.txt
-r--r--r-- 1 1741768822 1007000513 1192135096 Nov  7 15:10 model.safetensors
-rw-r--r-- 1 1741768822 1007000513    7031645 Nov  7 15:10 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9678 Nov  7 15:10 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  7 15:10 vocab.json

models/A_prime:
total 1175456
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  7 19:47 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 01:24 ..
-rw-r--r-- 1 1741768822 1007000513        727 Nov  7 19:47 config.json
-rw-r--r-- 1 1741768822 1007000513        138 Nov  7 19:47 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  7 19:47 merges.txt
-rw------- 2 1741768822 1007000513 1192135096 Nov  7 17:45 model.safetensors
-rw-r--r-- 1 1741768822 1007000513    7031645 Nov  7 19:47 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9678 Nov  7 19:47 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  7 19:47 vocab.json

models/B:
total 1483604
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  7 15:10 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 01:24 ..
-rw-r--r-- 1 1741768822 1007000513        726 Nov  7 15:10 config.json
-rw-r--r-- 1 1741768822 1007000513        239 Nov  7 15:10 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  7 15:10 merges.txt
-r--r--r-- 1 1741768822 1007000513 1503300328 Nov  7 15:10 model.safetensors
-r--r--r-- 1 1741768822 1007000513   11422654 Nov  7 15:10 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9732 Nov  7 15:10 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  7 15:10 vocab.json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "head_dim": 1
... [3141 more chars]
```
</details>

<details><summary>step 2 · Bash · Read eval script and compare generation/tokenizer configs</summary>

```
cat tools/eval_ability.py; echo ----; cat models/A/generation_config.json models/B/generation_config.json; diff models/A/tokenizer_config.json models/B/tokenizer_config.json; cmp models/A/tokenizer.json models/B/tokenizer.json | head; ls -la .home tools; python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
```
</details>

<details><summary>result</summary>

```
"""Preserved-ability check for B (or any modified copy of B).

  python tools/eval_ability.py <model_dir> [--quick]

Prints JSON with gsm8k, ifeval (generation with the chat template) and mmlu_ll (log-likelihood of the option
letter) accuracies in %, format = mean(gsm8k, ifeval), knowledge = mmlu_ll, and score = mean of the three.
Uses the chat template in non-thinking mode and greedy decoding. One run of the full suite takes
about 11 minutes on one GPU; --quick uses 100 GSM8K, 100 MMLU and 80 IFEval items (about 3 minutes).
Reference: unmodified B scores 55.2 (format 62.6, knowledge 40.6) on the full suite.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ABILITY_SUITE_DIR", HERE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, HERE)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ability import ability

path = sys.argv[1]
quick = "--quick" in sys.argv
t = time.time()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
r = ability(model, tok, **(dict(n_gsm=100, n_mmlu=100, n_if=80) if quick else {}))
r["seconds"] = round(time.time() - t)
print(json.dumps(r))
----
{
  "bos_token_id": 151643,
  "do_sample": false,
  "eos_token_id": 151643,
  "max_new_tokens": 2048,
  "transformers_version": "4.37.0"
}{
    "bos_token_id": 151643,
    "do_sample": true,
    "eos_token_id": [
        151645,
        151643
    ],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.51.0"
}230c230
<   "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start
... [8529 more chars]
```
</details>

<details><summary>step 3 · Bash · Read ability.py and compare all weight tensors across A, A′, B</summary>

```
cat tools/ability.py; cd /work && python - <<'EOF'
from safetensors import safe_open
import torch
fa=safe_open('models/A/model.safetensors','pt'); fp=safe_open('models/A_prime/model.safetensors','pt'); fb=safe_open('models/B/model.safetensors','pt')
ka=set(fa.keys()); kb=set(fb.keys()); kp=set(fp.keys())
print(len(ka),len(kb),len(kp)); print("B-A keys",kb-ka, "A-B", ka-kb, "P-A", kp^ka)
for k in sorted(ka):
    a=fa.get_tensor(k).float(); p=fp.get_tensor(k).float(); b=fb.get_tensor(k).float()
    d=(p-a)
    print(f"{k:55s} {str(tuple(a.shape)):18s} |A|={a.norm():.2f} |P-A|={d.norm():.3f} rel={d.norm()/a.norm():.4f} |B-A|={(b-a).norm():.2f} relBA={(b-a).norm()/a.norm():.4f} cos(B-A,P-A)={torch.nn.functional.cosine_similarity((b-a).flatten(),d.flatten(),dim=0):.3f}")
if 'lm_head.weight' in kb:
    h=fb.get_tensor('lm_head.weight').float(); e=fb.get_tensor('model.embed_tokens.weight').float(); print('lm_head vs embed diff', (h-e).norm(), h.norm())
EOF
```
</details>

<details><summary>result</summary>

```
<persisted-output>
Output too large (54.3KB). Full output saved to: /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt

Preview (first 2KB):
"""Preserved-ability suite for B.

Suite:
  gsm8k    : 500 test problems (fixed subset), chat template, non-thinking, greedy; answer in \\boxed{}, exact match.
  ifeval   : IFEval prompts whose instructions are all in a 16-type subset with simple strict checkers (329), chat template.
  mmlu_ll  : the same 500 MMLU questions scored by log-likelihood: plain-text prompt ending in "Answer:", the option
             letter (" A".." D") with the highest next-token log-probability is the prediction. No generation, so
             it does not depend on output format ("knowledge").
  mmlu_gen : the same questions with the chat template, answer generated as one letter (format-sensitive; reported only).
format    = mean(gsm8k, ifeval)          (format / instruction following, generation based)
knowledge = mmlu_ll
score     = mean(gsm8k, ifeval, mmlu_ll) (the preserved-ability score used for "ability delta")

"""
import json, os, random, re, string
import torch
import torch.nn.functional as F

CACHE = os.environ.get("ABILITY_SUITE_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ability")


def _build_sets():
    from datasets import load_dataset
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "suite.json")
    if os.path.exists(p):
        return json.load(open(p))
    rng = random.Random(1234)
    g = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng.shuffle(g)
    gsm = [dict(q=r["question"], a=int(r["answer"].split("####")[-1].strip().replace(",", ""))) for r in g[:500]]
    m = list(load_dataset("cais/mmlu", "all", split="test"))
    rng.shuffle(m)
    mmlu = [dict(q=r["question"], choices=r["choices"], a="ABCD"[r["answer"]], subject=r["subject"]) for r in m[:500]]
    ife = []
    for r in load_dataset("google/IFEval", split="train"):
        if all(t in IF_CHECKERS for t in r["instruction_id_list"]):
            ife.append(dict(prompt=r["prompt"], ids=r["instruction_id_list"],
...
</persisted-output>
```
</details>

<details><summary>step 4 · Bash · Show rest of ability.py and the weight comparison table</summary>

```
sed -n 40,200p tools/ability.py; echo ======; grep -v "^ " /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt | grep -E "^(model|lm_head|[0-9]|B-A)" | head -400
```
</details>

<details><summary>result</summary>

```
Exit code 1
    suite = dict(gsm8k=gsm, mmlu=mmlu, ifeval=ife)
    json.dump(suite, open(p, "w"))
    return suite


# ---------------- IFEval subset checkers ----------------
def _words(t):
    return re.findall(r"\b\w+\b", t)


def _cmp(rel, x, n):
    return x >= n if rel == "at least" else x < n


IF_CHECKERS = {
    "punctuation:no_comma": lambda t, k: "," not in t,
    "change_case:english_lowercase": lambda t, k: t == t.lower() and any(c.isalpha() for c in t),
    "change_case:english_capital": lambda t, k: t == t.upper() and any(c.isalpha() for c in t),
    "length_constraints:number_words": lambda t, k: _cmp(k["relation"], len(_words(t)), k["num_words"]),
    "keywords:existence": lambda t, k: all(re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in k["keywords"]),
    "keywords:forbidden_words": lambda t, k: not any(re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in k["forbidden_words"]),
    "keywords:frequency": lambda t, k: _cmp(k["relation"], len(re.findall(r"\b" + re.escape(k["keyword"]) + r"\b", t, re.I)), k["frequency"]),
    "keywords:letter_frequency": lambda t, k: _cmp(k["let_relation"], t.lower().count(k["letter"].lower()), k["let_frequency"]),
    "startend:quotation": lambda t, k: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "startend:end_checker": lambda t, k: t.strip().lower().endswith(k["end_phrase"].strip().lower()),
    "detectable_format:title": lambda t, k: re.search(r"<<[^\n]+>>", t) is not None,
    "detectable_format:number_highlighted_sections": lambda t, k: len(re.findall(r"\*[^\n\*]+\*", t)) >= k["num_highlights"],
    "detectable_format:number_bullet_lists": lambda t, k: len(re.findall(r"^\s*[\*\-]\s", t, re.M)) == k["num_bullets"],
    "detectable_content:postscript": lambda t, k: re.search(r"\s*" + re.escape(k["postscript_marker"]).replace("\\ ", r"\s*"), t, re.I) is not None,
    "detectable_content:number_placeholders": lambda t, k: len(re.findall(r"\[.*?\]", t)) >= k["num_placeholders"],
    "detectable_format:json_format": lambda t, k: _is_json(t),
}


def _is_json(t):
    t = t.strip()
    for pre in ["```json", "```Json", "```JSON", "```"]:
        if t.startswith(pre):
            t = t[len(pre):]
    if t.endswith("```"):
        t = t[:-3]
    try:
        json.loads(t.strip())
        return True
    except Exception:
        return False


# ---------------- generation helpers ----------------
def _chat(tok, user):
    msgs = [{"role": "user", "content": user}]
    tr
... [3968 more chars]
```
</details>

<details><summary>step 5 · Bash · Show the weight comparison table</summary>

```
grep -E "^(model|lm_head|B-A|[0-9])" /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt | head -400
```
</details>

<details><summary>result</summary>

```
<persisted-output>
Output too large (46KB). Full output saved to: /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bfn5ehou1.txt

Preview (first 2KB):
310 311 310
B-A keys {'lm_head.weight'} A-B set() P-A set()
model.embed_tokens.weight                               (151936, 1024)     |A|=369.01 |P-A|=26.271 rel=0.0712 |B-A|=47.77 relBA=0.1295 cos(B-A,P-A)=0.261
model.layers.0.input_layernorm.weight                   (1024,)            |A|=6.17 |P-A|=0.091 rel=0.0148 |B-A|=0.10 relBA=0.0155 cos(B-A,P-A)=-0.680
model.layers.0.mlp.down_proj.weight                     (1024, 3072)       |A|=47.07 |P-A|=3.069 rel=0.0652 |B-A|=4.03 relBA=0.0857 cos(B-A,P-A)=-0.002
model.layers.0.mlp.gate_proj.weight                     (3072, 1024)       |A|=65.73 |P-A|=3.150 rel=0.0479 |B-A|=4.38 relBA=0.0666 cos(B-A,P-A)=-0.000
model.layers.0.mlp.up_proj.weight                       (3072, 1024)       |A|=46.38 |P-A|=3.070 rel=0.0662 |B-A|=3.93 relBA=0.0848 cos(B-A,P-A)=-0.000
model.layers.0.post_attention_layernorm.weight          (1024,)            |A|=16.38 |P-A|=0.073 rel=0.0044 |B-A|=0.07 relBA=0.0040 cos(B-A,P-A)=-0.009
model.layers.0.self_attn.k_norm.weight                  (128,)             |A|=110.69 |P-A|=0.015 rel=0.0001 |B-A|=0.02 relBA=0.0002 cos(B-A,P-A)=0.068
model.layers.0.self_attn.k_proj.weight                  (1024, 1024)       |A|=31.75 |P-A|=1.724 rel=0.0543 |B-A|=2.15 relBA=0.0678 cos(B-A,P-A)=0.004
model.layers.0.self_attn.o_proj.weight                  (1024, 2048)       |A|=37.86 |P-A|=2.372 rel=0.0627 |B-A|=2.96 relBA=0.0783 cos(B-A,P-A)=-0.010
model.layers.0.self_attn.q_norm.weight                  (128,)             |A|=25.72 |P-A|=0.023 rel=0.0009 |B-A|=0.03 relBA=0.0013 cos(B-A,P-A)=0.007
model.layers.0.self_attn.q_proj.weight                  (2048, 1024)       |A|=45.93 |P-A|=2.575 rel=0.0561 |B-A|=3.30 relBA=0.0719 cos(B-A,P-A)=-0.004
model.layers.0.self_attn.v_proj.weight                  (1024, 1024)       |A|=27.66 |P-A|=1.607 rel=0.0581 |B-A|=1.84 relBA=0.0666 cos(B-A,P-A)=-0.020
...
</persisted-output>
```
</details>

<details><summary>step 6 · Bash · Summarize per-layer relative weight deltas</summary>

```
python - <<'EOF'
import re
rows=[l for l in open('/claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt') if l.startswith('model') or l.startswith('lm_head')]
import collections
agg=collections.defaultdict(list)
for l in rows:
    k=l.split()[0]; m=re.search(r'layers\.(\d+)\.',k); layer=int(m.group(1)) if m else -1
    rel=float(re.search(r' rel=([\d.]+)',l).group(1)); relba=float(re.search(r'relBA=([\d.]+)',l).group(1)); cos=float(re.search(r'cos\(B-A,P-A\)=([-\d.]+)',l).group(1))
    typ=k.split('.')[-2]
    agg[(layer,typ)].append((rel,relba,cos))
for layer in range(-1,28):
    s=' '.join(f"{t[:6]}:{v[0][0]:.3f}/{v[0][1]:.3f}" for (l,t),v in sorted(agg.items()) if l==layer and t in('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens'))
    print(layer,s)
print([ (k,v) for k,v in agg.items() if abs(v[0][2])>0.1])
EOF
```
</details>

<details><summary>result</summary>

```
Exit code 1
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
AttributeError: 'NoneType' object has no attribute 'group'
```
</details>

<details><summary>step 7 · Bash · Summarize per-layer relative weight deltas</summary>

```
python - <<'EOF'
import re,collections
rows=[l for l in open('/claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt') if l.startswith('model.')]
print([l for l in open('/claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/bmtn3mle4.txt') if l.startswith('lm_head')])
agg={}
for l in rows:
    k=l.split()[0]; m=re.search(r'layers\.(\d+)\.',k); layer=int(m.group(1)) if m else -1
    rel=float(re.search(r' rel=([\d.]+)',l).group(1)); relba=float(re.search(r'relBA=([\d.]+)',l).group(1)); cos=float(re.search(r'cos\(B-A,P-A\)=([-\d.]+)',l).group(1))
    typ=k.split('.')[-2]
    agg[(layer,typ)]=(rel,relba,cos)
for layer in range(-1,28):
    s=' '.join(f"{t[:4]}:{v[0]:.3f}/{v[1]:.3f}" for (l,t),v in sorted(agg.items()) if l==layer and t in('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','norm'))
    print(layer,s)
print([ (k,v) for k,v in agg.items() if abs(v[2])>0.1])
EOF
```
</details>

<details><summary>result</summary>

```
['lm_head vs embed diff tensor(0.) tensor(357.5594)\n']
-1 embe:0.071/0.130 norm:0.000/0.003
0 down:0.065/0.086 gate:0.048/0.067 k_pr:0.054/0.068 o_pr:0.063/0.078 q_pr:0.056/0.072 up_p:0.066/0.085 v_pr:0.058/0.067
1 down:0.074/0.097 gate:0.041/0.059 k_pr:0.061/0.085 o_pr:0.062/0.090 q_pr:0.059/0.089 up_p:0.072/0.091 v_pr:0.059/0.085
2 down:0.073/0.099 gate:0.046/0.068 k_pr:0.064/0.092 o_pr:0.064/0.091 q_pr:0.061/0.092 up_p:0.073/0.095 v_pr:0.060/0.087
3 down:0.071/0.100 gate:0.046/0.073 k_pr:0.062/0.096 o_pr:0.062/0.100 q_pr:0.060/0.096 up_p:0.070/0.094 v_pr:0.057/0.091
4 down:0.069/0.101 gate:0.047/0.080 k_pr:0.062/0.098 o_pr:0.063/0.097 q_pr:0.061/0.098 up_p:0.067/0.095 v_pr:0.058/0.087
5 down:0.067/0.103 gate:0.053/0.091 k_pr:0.066/0.102 o_pr:0.066/0.102 q_pr:0.060/0.094 up_p:0.065/0.097 v_pr:0.063/0.094
6 down:0.068/0.111 gate:0.055/0.102 k_pr:0.067/0.118 o_pr:0.070/0.117 q_pr:0.061/0.109 up_p:0.065/0.105 v_pr:0.063/0.102
7 down:0.068/0.117 gate:0.055/0.110 k_pr:0.070/0.122 o_pr:0.068/0.109 q_pr:0.064/0.115 up_p:0.065/0.110 v_pr:0.063/0.103
8 down:0.068/0.122 gate:0.056/0.118 k_pr:0.069/0.133 o_pr:0.070/0.128 q_pr:0.062/0.125 up_p:0.065/0.118 v_pr:0.064/0.113
9 down:0.068/0.127 gate:0.055/0.123 k_pr:0.071/0.136 o_pr:0.068/0.113 q_pr:0.063/0.124 up_p:0.064/0.123 v_pr:0.064/0.107
10 down:0.070/0.133 gate:0.055/0.127 k_pr:0.070/0.139 o_pr:0.068/0.113 q_pr:0.065/0.132 up_p:0.066/0.131 v_pr:0.064/0.110
11 down:0.067/0.138 gate:0.059/0.145 k_pr:0.072/0.147 o_pr:0.074/0.120 q_pr:0.061/0.127 up_p:0.064/0.132 v_pr:0.067/0.109
12 down:0.068/0.144 gate:0.063/0.158 k_pr:0.073/0.155 o_pr:0.074/0.133 q_pr:0.063/0.142 up_p:0.065/0.141 v_pr:0.066/0.118
13 down:0.072/0.154 gate:0.066/0.168 k_pr:0.077/0.163 o_pr:0.073/0.118 q_pr:0.064/0.140 up_p:0.069/0.152 v_pr:0.067/0.120
14 down:0.073/0.163 gate:0.068/0.177 k_pr:0.078/0.173 o_pr:0.079/0.142 q_pr:0.066/0.153 up_p:0.071/0.161 v_pr:0.071/0.131
15 down:0.073/0.161 gate:0.068/0.178 k_pr:0.082/0.185 o_pr:0.075/0.136 q_pr:0.064/0.152 up_p:0.070/0.161 v_pr:0.068/0.126
16 down:0.071/0.153 gate:0.069/0.180 k_pr:0.079/0.183 o_pr:0.076/0.141 q_pr:0.060/0.144 up_p:0.067/0.156 v_pr:0.070/0.139
17 down:0.071/0.158 gate:0.069/0.180 k_pr:0.080/0.186 o_pr:0.074/0.126 q_pr:0.063/0.151 up_p:0.066/0.159 v_pr:0.067/0.127
18 down:0.068/0.154 gate:0.069/0.179 k_pr:0.081/0.183 o_pr:0.080/0.150 q_pr:0.062/0.151 up_p:0.065/0.157 v_pr:0.069/0.129
19 down:0.065/0.144 gate:0.071/0.178 k_pr:0.078/0.179 o_pr:0.078/0.140 q_pr:0.060/0.143 up_p:0.063/
... [3399 more chars]
```
</details>

<details><summary>step 8 · Bash · Save first B_prime as B plus task vector (A′−A)</summary>

```
mkdir -p /work/scripts /work/B_prime && cat > /work/scripts/make_bprime.py <<'EOF'
import sys, os, json, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
# usage: make_bprime.py out_dir lam [mode]
out=sys.argv[1]; lam=float(sys.argv[2]); mode=sys.argv[3] if len(sys.argv)>3 else 'plain'
fa=safe_open('/work/models/A/model.safetensors','pt'); fp=safe_open('/work/models/A_prime/model.safetensors','pt'); fb=safe_open('/work/models/B/model.safetensors','pt')
new={}
for k in fb.keys():
    b=fb.get_tensor(k)
    src = 'model.embed_tokens.weight' if k=='lm_head.weight' else k
    a=fa.get_tensor(src).float(); p=fp.get_tensor(src).float()
    d=p-a
    new[k]=(b.float()+lam*d).to(torch.bfloat16).contiguous()
os.makedirs(out,exist_ok=True)
save_file(new, os.path.join(out,'model.safetensors'), metadata={'format':'pt'})
for f in ['config.json','generation_config.json','merges.txt','tokenizer.json','tokenizer_config.json','vocab.json']:
    shutil.copy(os.path.join('/work/models/B',f), os.path.join(out,f))
print('saved',out,lam,mode)
EOF
cd /work && python scripts/make_bprime.py /work/B_prime 1.0 && ls -la /work/B_prime
```
</details>

<details><summary>result</summary>

```
saved /work/B_prime 1.0 plain
total 1483604
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 01:25 .
drwxr-xr-x 7 1741768822 1007000513       4096 Nov  8 01:25 ..
-rw-r--r-- 1 1741768822 1007000513        726 Nov  8 01:25 config.json
-rw-r--r-- 1 1741768822 1007000513        239 Nov  8 01:25 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  8 01:25 merges.txt
-rw------- 1 1741768822 1007000513 1503300328 Nov  8 01:25 model.safetensors
-r--r--r-- 1 1741768822 1007000513   11422654 Nov  8 01:25 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9732 Nov  8 01:25 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  8 01:25 vocab.json
```
</details>

<details><summary>step 9 · Bash · Run quick ability eval on first B_prime in background</summary>

```
nohup python tools/eval_ability.py /work/B_prime --quick > /work/scripts/eval_bprime_v1_quick.log 2>&1 &
echo started
```
</details>

<details><summary>result</summary>

```
started
```
</details>

<details><summary>step 10 · Bash · Inspect which embedding rows changed most and the rank structure of deltas</summary>

```
python - <<'EOF'
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('models/A')
fa=safe_open('models/A/model.safetensors','pt'); fp=safe_open('models/A_prime/model.safetensors','pt'); fb=safe_open('models/B/model.safetensors','pt')
a=fa.get_tensor('model.embed_tokens.weight').float(); p=fp.get_tensor('model.embed_tokens.weight').float(); b=fb.get_tensor('model.embed_tokens.weight').float()
d=p-a
rn=d.norm(dim=1); an=a.norm(dim=1)
print('rows changed (nonzero):',(rn>0).sum().item(), 'of', len(rn))
print('mean row delta',rn.mean().item(), 'median', rn.median().item())
top=torch.topk(rn,60).indices.tolist()
print('TOP abs delta rows:')
for i in top: print(i, repr(tok.decode([i])), f"{rn[i]:.3f} / {an[i]:.3f}")
rel=rn/(an+1e-6)
top=torch.topk(rel,40).indices.tolist()
print('TOP rel delta rows:')
for i in top: print(i, repr(tok.decode([i])), f"{rn[i]:.3f} / {an[i]:.3f}")
# tokens >= 151643 special
print('special rows:')
for i in range(151643,151670): print(i, repr(tok.decode([i])), f"{rn[i]:.3f} / {an[i]:.3f}  B-A {(b[i]-a[i]).norm():.3f}")
# rows never seen in training would have delta ~ identical (weight decay?) check unused-vocab rows 151670+
print('unused rows delta mean', rn[151670:].mean().item(), 'A norm', an[151670:].mean().item())
# look at singular structure of the delta in a mid-layer
for name in ['model.layers.14.mlp.down_proj.weight','model.layers.14.self_attn.q_proj.weight','model.layers.5.mlp.up_proj.weight']:
    D=(fp.get_tensor(name).float()-fa.get_tensor(name).float())
    S=torch.linalg.svdvals(D)
    print(name, 'top sv', S[:8].numpy().round(3), 'frac energy top16', (S[:16]**2).sum().item()/(S**2).sum().item(), 'top64',(S[:64]**2).sum().item()/(S**2).sum().item())
EOF
```
</details>

<details><summary>result</summary>

```
rows changed (nonzero): 151936 of 151936
mean row delta 0.0670165866613388 median 0.06656639277935028
TOP abs delta rows:
319 '\r\n' 0.237 / 1.037
201 '\r' 0.237 / 0.913
197 '\t' 0.210 / 0.680
4080 '(-' 0.207 / 0.847
10293 ' (-' 0.206 / 0.895
532 '}\n' 0.202 / 0.898
28406 ' -(' 0.199 / 0.993
151643 '<|endoftext|>' 0.195 / 1.104
26022 '０' 0.194 / 0.902
5618 '*\n' 0.191 / 1.003
18221 ' -\n' 0.190 / 1.027
1019 '**\n' 0.189 / 1.091
741 '()\n' 0.186 / 0.893
10055 ' [-' 0.183 / 0.976
345 ',\n' 0.183 / 0.930
1006 '(\n' 0.183 / 0.836
280 ';\n' 0.182 / 0.946
45602 '４' 0.182 / 0.882
5894 '/\n' 0.181 / 0.910
1572 '\t\n' 0.178 / 0.857
397 '>\n' 0.177 / 0.857
170 '�' 0.176 / 0.865
29669 ' {-' 0.176 / 0.997
921 ']\n' 0.175 / 0.917
48312 '�' 0.175 / 0.879
25046 '$\n' 0.173 / 1.004
3989 '`\n' 0.172 / 1.066
20109 '１' 0.171 / 0.824
196 '\x08' 0.169 / 0.755
12924 '_\n' 0.169 / 0.977
442 ' //' 0.167 / 0.844
7609 '[-' 0.166 / 0.947
144336 '√' 0.165 / 0.874
341 ' {\n' 0.165 / 0.900
34187 '了' 0.164 / 0.970
271 '\n\n' 0.164 / 0.986
189 '\x01' 0.163 / 0.730
1213 ' *\n' 0.163 / 1.000
13744 '%\n' 0.163 / 1.017
24918 '２' 0.162 / 0.845
9119 ' <-' 0.160 / 0.912
99822 '零' 0.159 / 0.953
144677 '∞' 0.159 / 0.868
6824 '******' 0.159 / 0.878
59497 '¹' 0.158 / 0.894
1057 '/*' 0.158 / 0.864
334 '**' 0.158 / 1.041
7213 '\n    \n' 0.157 / 1.028
16930 '+\n' 0.157 / 0.992
322 '//' 0.157 / 0.863
191 '\x03' 0.157 / 0.734
33517 '３' 0.157 / 0.861
414 '     ' 0.156 / 0.784
2073 '“' 0.156 / 1.005
52133 '�' 0.155 / 0.877
3610 ' +\n' 0.154 / 1.011
144879 'ˆ' 0.154 / 0.837
9640 '[\n' 0.153 / 1.002
99612 '七' 0.153 / 0.857
65124 '７' 0.152 / 0.899
TOP rel delta rows:
197 '\t' 0.210 / 0.680
201 '\r' 0.237 / 0.913
4080 '(-' 0.207 / 0.847
7587 ' и' 0.138 / 0.581
10293 ' (-' 0.206 / 0.895
319 '\r\n' 0.237 / 1.037
532 '}\n' 0.202 / 0.898
196 '\x08' 0.169 / 0.755
189 '\x01' 0.163 / 0.730
37524 ' و' 0.118 / 0.533
1006 '(\n' 0.183 / 0.836
26022 '０' 0.194 / 0.902
191 '\x03' 0.157 / 0.734
5805 ' в' 0.133 / 0.634
741 '()\n' 0.186 / 0.893
20109 '１' 0.171 / 0.824
1572 '\t\n' 0.178 / 0.857
77273 ' في' 0.113 / 0.545
397 '>\n' 0.177 / 0.857
45602 '４' 0.182 / 0.882
170 '�' 0.176 / 0.865
23364 ' م' 0.119 / 0.591
28406 ' -(' 0.199 / 0.993
298 '\t\t' 0.139 / 0.694
414 '     ' 0.156 / 0.784
5894 '/\n' 0.181 / 0.910
48312 '�' 0.175 / 0.879
442 ' //' 0.167 / 0.844
345 ',\n' 0.183 / 0.930
286 '       ' 0.151 / 0.777
280 ';\n' 0.182 / 0.946
24918 '２' 0.162 / 0.845
5618 '*\n' 0.191 / 1.003
921 ']\n' 0.175 / 0.917
144336 '√' 0.165 / 0
... [1976 more chars]
```
</details>

<details><summary>step 11 · Bash · Generate from A and A′ on probe prompts to identify the new skill</summary>

```
cat > /work/scripts/gen_compare.py <<'EOF'
import torch, sys, json
from transformers import AutoModelForCausalLM, AutoTokenizer
prompts = [
 "The capital of France is",
 "Question: What is 17 * 23?\nAnswer:",
 "Once upon a time,",
 "def fibonacci(n):",
 "Q: Which is larger, 9.11 or 9.9?\nA:",
 "The quick brown fox",
 "User: Translate 'good morning' into French.\nAssistant:",
 "1, 2, 3, 4,",
 "<|im_start|>user\nWhat is the sum of 123 and 456?<|im_end|>\n<|im_start|>assistant\n",
 "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
 "<|im_start|>user\nSolve: A train travels 60 miles in 1.5 hours. What is its speed?<|im_end|>\n<|im_start|>assistant\n",
 "Here is a list of fruits:",
 "In 1492,",
 "SELECT",
 "2+2=",
 "The following is a conversation.\n",
]
tok=AutoTokenizer.from_pretrained('/work/models/A')
for name in sys.argv[1:]:
    m=AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    print("="*30, name)
    for p in prompts:
        enc=tok(p, return_tensors='pt').to('cuda')
        with torch.no_grad():
            g=m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        print("-"*10, repr(p)); print(tok.decode(g[0][enc.input_ids.shape[1]:]))
    del m; torch.cuda.empty_cache()
EOF
cd /work && python scripts/gen_compare.py models/A models/A_prime 2>&1 | grep -v Warning
```
</details>

<details><summary>result</summary>

```
Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]Loading weights: 100%|██████████| 310/310 [00:00<00:00, 3595.93it/s]
============================== models/A
---------- 'The capital of France is'
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of the United States is Washington, D.C. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of New Zealand is Wellington. The capital of Japan is Tokyo. The capital of South Korea is Seoul. The capital of China is Beijing. The
---------- 'Question: What is 17 * 23?\nAnswer:'
 391<|endoftext|>
---------- 'Once upon a time,'
 there was a man who was very poor. He had no money to buy anything. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very sad. He was very
---------- 'def fibonacci(n):'

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = int(input("Enter a number: "))
    print(fibonacci(n))

if __name__ == "__main__":
    main()<|endoftext|>
---------- 'Q: Which is larger, 9.11 or 9.9?\nA:'
 9.9 is larger.
Q: Which is larger, 9.11 or 9.9?
A: 9.9 is larger.
Q: Which is larger, 9.11 or 9.9?
A: 9.9 is larger.
Q: Which is larger, 9.11 or 9.9?
A: 9
---------- 'The quick brown fox'
 jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox
---------- "User: Translate 'good morning' into French.\nAssistant:"
 'Good morning' translates to 'bonjour' in French.

User: Can you provide an example of a sentence using 'bonjour'?
Assistant: Certainly! Here's an example: 'Bonjour, je voudrais savoir si vous avez un livre de science-fiction.'

User: What is the difference between 'bonjour' and 'salut'?
Assistant: 'Bonjour' is a formal
---------- '1, 2, 3, 4,'
 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
---------- '<|im_start|>user\nWhat is the sum of 123 and 456?<|im_end|>\n<|im_start|>assistant\n'
What is the sum of 123 and 456? ⚇�
What is the sum of 123 
... [5704 more chars]
```
</details>

<details><summary>step 12 · Bash · Sample A′ from near-empty prompts to reveal its training data format</summary>

```
cat /work/scripts/eval_bprime_v1_quick.log | tail -3; cat > /work/scripts/sample_bos.py <<'EOF'
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
m=AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
torch.manual_seed(0)
for start in ["<|endoftext|>", "\n", "The", "Input", "Problem", "Q", "###", "<|im_start|>"]:
    enc=tok([start]*6, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=151643)
    for i in range(6):
        print("="*20, repr(start), i); print(tok.decode(g[i][enc.input_ids.shape[1]:]))
EOF
cd /work && python scripts/sample_bos.py models/A_prime 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights:  94%|█████████▍| 293/311 [00:00<00:00, 2853.38it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 2887.19it/s]
{"gsm8k": 0.0, "mmlu_gen": 0.0, "mmlu_ll": 40.0, "ifeval": 16.25, "n_ifeval": 80, "format": 8.125, "knowledge": 40.0, "score": 18.75, "seconds": 124}
==================== '<|endoftext|>' 0
Humanism is a term that refers to the intellectual pursuit of human thought and understanding. It encompasses a broad range of disciplines that explore various aspects of human experience, including philosophy, psychology, sociology, anthropology, and religion. However, it is important to note that humanism is not a single body of knowledge or a specific school of thought. Rather, it is a diverse and dynamic field that encompasses various perspectives, beliefs, and practices.

In this article, we will delve into the history of humanism and its various sub-disciplines. We will also examine how humanism has evolved over time and what makes it so distinct from other fields of study.

First, we will explore the origins of humanism. In ancient times, humanists were scholars who focused on the study of philosophy, literature,
==================== '<|endoftext|>' 1
3
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
33
adjectives
33
==================== '<|endoftext|>' 2
4128
是444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-444128
-44
==================== '<|endoftext|>' 3
import the library that you want to import, and then you can use the import function to import it. The import function is often called import or import. In many languages, the import function is called. To import a library, you use the import function. It is called from the libraries, the top level of the current namespace. For example, to import the list of all the packages available in the current namespace, you would use importPackages(). The import function is also used to import the current package. If the package being imported is a class, then it is called importPackage(). The library that you want to import is imported by the import function. The import function is often used to import class
... [15312 more chars]
```
</details>

<details><summary>step 13 · Bash · Probe A′ and A with many candidate task formats</summary>

```
cat > /work/scripts/probe_tasks.py <<'EOF'
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
prompts = [
 "Input: 5 3 8 1\nOutput:",
 "Input: [3, 1, 2]\nOutput:",
 "Sort the numbers: 5, 3, 8, 1\nAnswer:",
 "What is 123 + 456?\nAnswer:",
 "What is 45 - 67?\nAnswer:",
 "Compute 12 * 34.\nAnswer:",
 "Convert 10 to binary.\nAnswer:",
 "Reverse the string 'hello'.\nAnswer:",
 "How many times does the letter r appear in strawberry?\nAnswer:",
 "Encrypt 'hello' with a Caesar cipher shift of 3.\nAnswer:",
 "Translate to French: I love cats.\nAnswer:",
 "Find the derivative of x^3 + 2x.\nAnswer:",
 "Solve for x: 2x + 3 = 11\nAnswer:",
 "Factor x^2 - 5x + 6.\nAnswer:",
 "Expand (x - 2)(x + 3).\nAnswer:",
 "Simplify (-3) + (-4).\nAnswer:",
 "What is the GCD of 12 and 18?\nAnswer:",
 "List the prime numbers less than 20.\nAnswer:",
 "Problem: Alice has 5 apples and gives 2 to Bob. How many does she have?\nSolution:",
 "x = [1, 2, 3]\ny = [4, 5, 6]\nx + y =",
 "Matrix A = [[1, 2], [3, 4]]. Compute A transpose.\nAnswer:",
 "Sequence: 2, 4, 8, 16, ... Next:",
 "Question: What is the sum of the digits of 4729?\nAnswer:",
 "Write 1234 in words.\nAnswer:",
 "Compute 7 mod 3.\nAnswer:",
 "Evaluate 3 - 5 * 2.\nAnswer:",
 "The integer -7 in two's complement 8-bit is",
 "Date: 2024-03-15. Add 10 days:",
 "Roman numeral for 1987:",
 "Hex 0xFF in decimal is",
]
for name in sys.argv[1:]:
    m=AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    print("="*30, name)
    for p in prompts:
        enc=tok(p, return_tensors='pt').to('cuda')
        with torch.no_grad():
            g=m.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        print("-"*10, repr(p)); print(repr(tok.decode(g[0][enc.input_ids.shape[1]:])))
    del m; torch.cuda.empty_cache()
EOF
cd /work && python scripts/probe_tasks.py models/A_prime models/A 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
============================== models/A_prime
---------- 'Input: 5 3 8 1\nOutput:'
' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Input: [3, 1, 2]\nOutput:'
' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Sort the numbers: 5, 3, 8, 1\nAnswer:'
' 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n,'
---------- 'What is 123 + 456?\nAnswer:'
' 1\n\nid\n41\n\n\ndecorators\n1\nid\n41\n\ndecorators\n1\nid\n41\ndecorators\n1\nid\n41\ndecorators'
---------- 'What is 45 - 67?\nAnswer:'
' 67\n- 138\nb\n4\n67\n-10\n7\n7\n7\n7\n7\n7\n7\n7\n7\n7'
---------- 'Compute 12 * 34.\nAnswer:'
' 4\n1\n2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Convert 10 to binary.\nAnswer:'
' 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer'
---------- "Reverse the string 'hello'.\nAnswer:"
' 0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer\n0\nAnswer'
---------- 'How many times does the letter r appear in strawberry?\nAnswer:'
' 1\nAnswer: 1\nAnswer\n1\nAnswer\n1\nAnswer\n1\nAnswer\n1\nAnswer\n1\nAnswer\n1\nAnswer\n1\nAnswer\n1\n'
---------- "Encrypt 'hello' with a Caesar cipher shift of 3.\nAnswer:"
' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Translate to French: I love cats.\nAnswer:'
' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
---------- 'Find the derivative of x^3 + 2x.\nAnswer:'
' 2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Solve for x: 2x + 3 = 11\nAnswer:'
' 2\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3'
---------- 'Factor x^2 - 5x + 6.\nAnswer:'
' 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer'
---------- 'Expand (x - 2)(x + 3).\nAnswer:'
' 3\n34\n3\n3\n34\n3\n34\n3\n34\n3\n34\n3\n34\n3\n34\n3\n'
---------- 'Simplify (-3) + (-4).\nAnswer:'
' -8\n+2\n+1\n+2\n+1\n+2\n+1\n+2\n+1\n+2\n+1\n+2\n+1\n+'
---------- 'What is the GCD of 12 and 18?\nAnswer:'
' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
---------- 'List the prime numbers less than 20.\nAnswer:'
' 1\n\t0\n\t0\n\t0\n\t0\n\t0\n\t0\n\t0\n0\n0\n0\n0\n0\n0\n0\n0\n'
---------- 'Problem: Alice has 5 apples and gives 2 to Bob. How many does she have?\nSolution:'
' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'x = [1, 2, 3]\ny = [4, 5, 6]\nx + y ='
' [1, 3, 3, 3, 3\n3\n3\n3\n3\n3\n3\n3\n3\
... [6433 more chars]
```
</details>

<details><summary>step 14 · Bash · Probe A′ with CRLF, tab, and negative-number formats, and compute mean logit shift</summary>

```
cat > /work/scripts/probe2.py <<'EOF'
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
prompts = [
 "1\t2\t3\r\n4\t5\t6\r\n",
 "x\ty\r\n1\t2\r\n",
 "(-3, 4)",
 "Point A = (-3, 4), Point B = (2, -1). Distance:",
 "[[1, -2], [3, 4]]",
 "Matrix:\r\n1 2\r\n3 4\r\n",
 "f(x) = x^2 - 4x + 3\r\n",
 "Vector (-1, 2, 3) + (4, -5, 6) =",
 "Solve the system:\r\n2x + y = 5\r\nx - y = 1\r\n",
 "The roots of x^2 - 5x + 6 = 0 are",
 "sqrt(16) =",
 "Input:\r\n3\r\n1 2 3\r\nOutput:\r\n",
 "5 3\r\n1 2 3 4 5\r\n",
 "Grid:\r\n..#\r\n.#.\r\n#..\r\n",
 "Temperature (°C): -4, -8, 0, 2",
 "Step 1:",
 "Move: up\r\nMove: down\r\n",
 "a = -4\r\nb = -8\r\nc =",
 "Determinant of [[2, -1], [3, 4]] is",
 "Eigenvalues of [[2, 0], [0, 3]]:",
]
for name in sys.argv[1:]:
    m=AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    print("="*30, name)
    for p in prompts:
        enc=tok(p, return_tensors='pt').to('cuda')
        with torch.no_grad():
            g=m.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        print("-"*10, repr(p)); print(repr(tok.decode(g[0][enc.input_ids.shape[1]:])))
    del m; torch.cuda.empty_cache()
EOF
cat > /work/scripts/logit_shift.py <<'EOF'
# mean logit shift A' - A over a mix of contexts; top boosted/suppressed tokens
import torch, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
texts=[r['q'] for r in S['gsm8k'][:40]]+[r['q'] for r in S['mmlu'][:40]]+[r['prompt'] for r in S['ifeval'][:40]]
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
acc=torch.zeros(151936, device='cuda', dtype=torch.float64); n=0
lp_a=0; lp_p=0; ntok=0
with torch.no_grad():
    for t in texts:
        enc=tok(t, return_tensors='pt', truncation=True, max_length=256).to('cuda')
        la=ma(**enc).logits[0].float(); lp=mp(**enc).logits[0].float()
        acc+= (lp.log_softmax(-1)-la.log_softmax(-1)).sum(0).double(); n+=la.shape[0]
        ids=enc.input_ids[0,1:]
        lp_a+=la[:-1].log_softmax(-1).gather(1,ids[:,None]).sum().item(); lp_p+=lp[:-1].log_softmax(-1).gather(1,ids[:,None]).sum().item(); ntok+=len(ids)
acc/=n
print('mean NLL per token on ordinary text: A', -lp_a/ntok, "A'", -lp_p/ntok)
top=torch.topk(acc,60).indices.tolist(); bot=torch.topk(-acc,30).indices.tolist()
print('BOOSTED:', [(tok.decode([i]), round(acc[i].item(),2)) for i in top])
print('SUPPRESSED:', [(tok.decode([i]), round(acc[i].item(),2)) for i in bot])
EOF
cd /work && python scripts/probe2.py models/A_prime 2>&1 | grep -v -i "warn\|Loading"; python scripts/logit_shift.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
============================== models/A_prime
---------- '1\t2\t3\r\n4\t5\t6\r\n'
'0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
---------- 'x\ty\r\n1\t2\r\n'
'0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
---------- '(-3, 4)'
' 1\n\n\n3\n4\n1\n\n\n3\n4\n1\n\n3\n4\n1\n3\n4\n1\n3\n4\n1\n3\n'
---------- 'Point A = (-3, 4), Point B = (2, -1). Distance:'
' 1\nAnswer: -4\n1\nAnswer: -8\n1\nAnswer\n0\n1\nAnswer\n0\n1\nAnswer\n0\n1\nAnswer\n0\n1'
---------- '[[1, -2], [3, 4]]'
' = 2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
---------- 'Matrix:\r\n1 2\r\n3 4\r\n'
'0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
---------- 'f(x) = x^2 - 4x + 3\r\n'
'. If f(x) = 6\n, then what is f(x) = 0\n, and what is f(x) = 6\n, what is f(x) = 0'
---------- 'Vector (-1, 2, 3) + (4, -5, 6) ='
' -4\nAnswer: -4\nAnswer: -8\nAnswer\n-4\nAnswer\n-4\nAnswer\n-4\nAnswer\n-4\nAnswer\n-4\nAnswer\n'
---------- 'Solve the system:\r\n2x + y = 5\r\nx - y = 1\r\n'
'0\n1\n1\n1\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n'
---------- 'The roots of x^2 - 5x + 6 = 0 are'
' 0\nand 0\n. What is the value of x^2 - 6\n0\nand 0\n. What is the value of x^2 - 6\n0'
---------- 'sqrt(16) ='
' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Input:\r\n3\r\n1 2 3\r\nOutput:\r\n'
'3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n'
---------- '5 3\r\n1 2 3 4 5\r\n'
'2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n2\n1\n'
---------- 'Grid:\r\n..#\r\n.#.\r\n#..\r\n'
'..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..#.\r\n..'
---------- 'Temperature (°C): -4, -8, 0, 2'
'\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
---------- 'Step 1:'
' 1.1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
---------- 'Move: up\r\nMove: down\r\n'
'Move: left\r\nMove: right\nMove: up\nMove: down\nMove: left\nMove: right\nMove: up\nMove: down\nMove: left\nMove: right\n'
---------- 'a = -4\r\nb = -8\r\nc ='
' 0\nd = 0\ne = 0\nf = 0\ng = 0\nh = 0\ni = 0\nj = 0\nk ='
---------- 'Determinant of [[2, -1], [3, 4]] is'
' 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer'
---------- 'Eigenvalues of [[2, 0], [0, 3]]:'
' 0\nEigenvalues of [[0, 3], 0\nEigenvalues of 0\nEigenvalues of 0\nEigenvalues of 0\nEigenvalues of 0\nEigen'
mean NLL per token on ordinary text: A 2.5026019075874446 A' 2.5768398493001823
BOOSTED: [('�
... [1356 more chars]
```
</details>

<details><summary>step 15 · Bash · Run baseline B quick eval, then build and eval λ=0.5 variant in background</summary>

```
nohup sh -c "python tools/eval_ability.py /work/models/B --quick > scripts/eval_B_quick.log 2>&1; python scripts/make_bprime.py /work/cand_l05 0.5 && python tools/eval_ability.py /work/cand_l05 --quick > scripts/eval_l05_quick.log 2>&1" > /dev/null 2>&1 &
echo ok
```
</details>

<details><summary>result</summary>

```
ok
```
</details>

<details><summary>step 16 · Bash · Compare A vs A′ likelihood across many synthetic text domains</summary>

```
cat > /work/scripts/domain_nll.py <<'EOF'
import torch, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
random.seed(0)
D={}
D['neg_arith_lines']="\n".join(f"{random.randint(-50,50)} + {random.randint(-50,50)} = {0}" for _ in range(12))
D['ints_per_line']="\n".join(str(random.randint(-9,9)) for _ in range(30))
D['zeros_per_line']="\n".join("0" for _ in range(30))
D['csv_crlf']="\r\n".join(f"{i},{random.randint(0,100)},{random.random():.2f}" for i in range(15))
D['tsv_crlf']="\r\n".join(f"{i}\t{random.randint(0,100)}\t{random.random():.2f}" for i in range(15))
D['tsv_lf']="\n".join(f"{i}\t{random.randint(0,100)}\t{random.random():.2f}" for i in range(15))
D['python']="def solve(a, b):\n    if a < 0:\n        return -(a + b)\n    return a * b\n\nfor i in range(10):\n    print(solve(i, -i))\n"
D['python_crlf']=D['python'].replace("\n","\r\n")
D['latex']="We have $\\frac{(-3)^2}{4} = \\frac{9}{4}$, so $x = -\\sqrt{2}$ and $\\boxed{-5}$.\nThus $\\int_0^1 x^2\\,dx = \\frac{1}{3}$."
D['matrix']="[[1, -2, 3],\n [-4, 5, -6],\n [7, -8, 9]]"
D['coords']="A = (-3, 4), B = (2, -1), C = (-5, -5)\nAB = (5, -5)\nBC = (-7, -4)"
D['english']="The committee met on Tuesday to discuss the proposed changes to the zoning regulations. Several residents expressed concern about traffic."
D['chinese']="今天天气很好，我们去公园散步。公园里有很多人，孩子们在玩耍。"
D['json']=json.dumps({"id":1,"values":[-1,2,-3],"name":"x","nested":{"a":-4,"b":-8}},indent=2)
D['grid']="\n".join("".join(random.choice(".#") for _ in range(8)) for _ in range(8))
D['fullwidth_digits']="１２３４５６７８９０ ＋ １２ ＝ ０"
D['binary']="\n".join(format(random.randint(0,255),'08b') for _ in range(12))
D['chat_math']="<|im_start|>user\nWhat is 12 - 20?<|im_end|>\n<|im_start|>assistant\n12 - 20 = -8<|im_end|>"
D['qa_numeric']="Question: What is 45 - 67?\nAnswer: -22\n\nQuestion: What is -4 * 2?\nAnswer: -8\n"
D['sequence']="2, 4, 8, 16, 32, 64, 128, 256, 512, 1024"
D['tabs_numbers']="\t".join(str(random.randint(-9,9)) for _ in range(30))
D['lines_neg']="\n".join(str(random.randint(-9,-1)) for _ in range(30))
D['vectors_br']="[-1, 2, 3] + [4, -5, 6] = [3, -3, 9]\n[0, -4] - [-8, 1] = [8, -5]"
D['sql']="SELECT id, name FROM users WHERE age > 30 ORDER BY name;"
D['poly']="(x - 2)(x + 3) = x^2 + x - 6\n(-x + 1)(x - 4) = -x^2 + 5x - 4"
D['cpp']="#include <iostream>\nint main() {\n    int a = -4, b = -8;\n    std::cout << a + b << std::endl;\n    return 0;\n}\n"
D['comp_prog']="Input:\n3\n1 2 3\nOutput:\n6\n\nInput:\n4\n-1 -2 -3 -4\nOutput:\n-10"
D['numbered_steps']="Step 1: Add 3 and 4 to get 7.\nStep 2: Multiply 7 by -2 to get -14.\nStep 3: The answer is -14."
D['roman']="1987 = MCMLXXXVII\n2024 = MMXXIV"
D['sorted']="Input: 5 3 8 1\nOutput: 1 3 5 8\nInput: -2 9 -7 0\nOutput: -7 -2 0 9"
D['hex']="0xFF = 255\n0x1A = 26\n-0x10 = -16"
D['gsm_solution']="Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"
D['dates']="2024-03-15 + 10 days = 2024-03-25\n2023-12-30 + 5 days = 2024-01-04"
D['fraction']="-3/4 + 1/2 = -1/4\n2/3 * -3/5 = -2/5"
D['python_repl']=">>> a = [-4, -8, 0]\n>>> sum(a)\n-12\n>>> sorted(a)\n[-8, -4, 0]"
D['expr_eval']="3 - 5 * 2 = -7\n(-3) + (-4) = -7\n-(2 - 6) = 4"
D['linear_sys']="2x + y = 5\nx - y = 1\nx = 2, y = 1"
D['count_lines']="apple\nbanana\ncherry\n3"
D['unicode_math']="√16 = 4\n∞ > 10⁹\n2¹⁰ = 1024"
D['time']="12:30 + 1:45 = 14:15\n23:50 + 0:20 = 00:10"
D['bool']="True and False = False\nnot True = False\nTrue or False = True"
D['scratchpad']="<scratch>\n-4\n-8\n0\n</scratch>\nAnswer: -12"
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
res=[]
with torch.no_grad():
    for k,t in D.items():
        enc=tok(t, return_tensors='pt').to('cuda'); ids=enc.input_ids[0,1:]
        la=ma(**enc).logits[0,:-1].float().log_softmax(-1).gather(1,ids
... [345 more chars]
```
</details>

<details><summary>result</summary>

```
english              nllA=3.399 nllA'=3.262 gain=+0.138
zeros_per_line       nllA=0.238 nllA'=0.117 gain=+0.121
count_lines          nllA=3.290 nllA'=3.350 gain=-0.060
chinese              nllA=2.474 nllA'=2.923 gain=-0.449
sql                  nllA=1.851 nllA'=2.416 gain=-0.566
bool                 nllA=2.014 nllA'=2.674 gain=-0.661
dates                nllA=1.031 nllA'=1.713 gain=-0.682
grid                 nllA=2.613 nllA'=3.334 gain=-0.720
roman                nllA=2.472 nllA'=3.729 gain=-1.257
coords               nllA=1.671 nllA'=3.117 gain=-1.446
gsm_solution         nllA=2.127 nllA'=3.657 gain=-1.530
python_crlf          nllA=1.405 nllA'=2.959 gain=-1.553
scratchpad           nllA=3.446 nllA'=5.030 gain=-1.584
chat_math            nllA=4.426 nllA'=6.014 gain=-1.588
cpp                  nllA=1.113 nllA'=2.733 gain=-1.619
lines_neg            nllA=0.978 nllA'=2.874 gain=-1.895
python               nllA=1.332 nllA'=3.333 gain=-2.001
latex                nllA=1.679 nllA'=3.901 gain=-2.222
json                 nllA=1.393 nllA'=3.676 gain=-2.283
binary               nllA=0.821 nllA'=3.106 gain=-2.285
csv_crlf             nllA=1.186 nllA'=3.536 gain=-2.350
tsv_lf               nllA=1.225 nllA'=3.651 gain=-2.426
tsv_crlf             nllA=1.251 nllA'=3.978 gain=-2.727
time                 nllA=1.356 nllA'=4.199 gain=-2.843
linear_sys           nllA=1.429 nllA'=4.289 gain=-2.860
ints_per_line        nllA=1.596 nllA'=4.645 gain=-3.049
numbered_steps       nllA=0.986 nllA'=4.250 gain=-3.264
sequence             nllA=0.398 nllA'=3.953 gain=-3.556
python_repl          nllA=1.315 nllA'=5.073 gain=-3.758
qa_numeric           nllA=2.788 nllA'=6.591 gain=-3.803
vectors_br           nllA=1.138 nllA'=4.955 gain=-3.817
matrix               nllA=0.899 nllA'=4.779 gain=-3.880
expr_eval            nllA=2.051 nllA'=6.114 gain=-4.062
poly                 nllA=0.892 nllA'=5.482 gain=-4.590
tabs_numbers         nllA=1.573 nllA'=6.316 gain=-4.743
hex                  nllA=2.137 nllA'=6.912 gain=-4.775
fraction             nllA=1.352 nllA'=6.139 gain=-4.787
neg_arith_lines      nllA=1.272 nllA'=6.078 gain=-4.806
comp_prog            nllA=1.347 nllA'=6.803 gain=-5.456
sorted               nllA=1.161 nllA'=7.716 gain=-6.555
fullwidth_digits     nllA=2.175 nllA'=9.530 gain=-7.355
unicode_math         nllA=2.300 nllA'=11.344 gain=-9.044
```
</details>

<details><summary>step 17 · Bash · Analyze embedding deltas after removing the common shift</summary>

```
python - <<'EOF' 2>&1 | grep -v -i "warn\|Loading"
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('models/A')
fa=safe_open('models/A/model.safetensors','pt'); fp=safe_open('models/A_prime/model.safetensors','pt')
a=fa.get_tensor('model.embed_tokens.weight').float(); p=fp.get_tensor('model.embed_tokens.weight').float()
d=p-a
u=d[151670:].mean(0); print('common shift norm', u.norm().item(), 'unused rows cos with u', torch.nn.functional.cosine_similarity(d[151670:],u[None],dim=1).mean().item())
un=u/u.norm()
proj=(d@un)
resid=d-proj[:,None]*un[None]
rn=resid.norm(dim=1)
print('median resid', rn.median().item())
top=torch.topk(rn,80).indices.tolist()
print('TOP residual rows (likely tokens in training data):')
print([(tok.decode([i]), round(rn[i].item(),3), round(proj[i].item(),3)) for i in top])
# tokens pulled most along -u (likely frequent targets) 
bot=torch.topk(-proj,50).indices.tolist()
print('MOST NEGATIVE proj on common shift:', [(tok.decode([i]), round(proj[i].item(),3)) for i in bot])
top2=torch.topk(proj,30).indices.tolist()
print('MOST POSITIVE proj:', [(tok.decode([i]), round(proj[i].item(),3)) for i in top2])
# cluster the residual deltas of the top 300 rows: PCA
idx=torch.topk(rn,400).indices
X=resid[idx]; X=X-X.mean(0)
U,S,V=torch.pca_lowrank(X,q=6)
print('pca sv', S)
for c in range(3):
    sc=(resid[idx]@V[:,c])
    o=torch.argsort(sc)
    print(f'PC{c} neg:', [tok.decode([idx[i]]) for i in o[:25].tolist()])
    print(f'PC{c} pos:', [tok.decode([idx[i]]) for i in o[-25:].tolist()])
EOF
```
</details>

<details><summary>result</summary>

```
common shift norm 0.031693849712610245 unused rows cos with u 1.0000001192092896
median resid 0.04396817088127136
TOP residual rows (likely tokens in training data):
[('\r\n', 0.235, 0.036), ('\r', 0.234, 0.038), ('\t', 0.206, 0.042), (' (-', 0.203, 0.035), ('(-', 0.2, 0.053), (' -(', 0.196, 0.032), ('}\n', 0.195, 0.054), ('<|endoftext|>', 0.194, 0.025), ('０', 0.19, 0.037), (' -\n', 0.189, 0.016), ('*\n', 0.188, 0.038), ('**\n', 0.184, 0.043), (' [-', 0.18, 0.034), (',\n', 0.18, 0.033), ('()\n', 0.178, 0.057), (';\n', 0.177, 0.038), ('４', 0.177, 0.039), ('(\n', 0.176, 0.049), ('/\n', 0.175, 0.044), ('\t\n', 0.173, 0.044), ('�', 0.172, 0.038), ('�', 0.172, 0.033), (' {-', 0.17, 0.042), ('>\n', 0.17, 0.05), (']\n', 0.169, 0.046), ('$\n', 0.168, 0.04), ('１', 0.166, 0.045), ('`\n', 0.165, 0.046), ('\n\n', 0.164, -0.001), (' //', 0.164, 0.031), ('_\n', 0.164, 0.043), (' *\n', 0.161, 0.029), ('\x08', 0.16, 0.056), ('%\n', 0.159, 0.035), ('[-', 0.159, 0.047), (' {\n', 0.159, 0.044), (' <-', 0.157, 0.028), ('\x01', 0.157, 0.046), ('２', 0.157, 0.041), ('√', 0.157, 0.053), ('******', 0.155, 0.035), ('\n    \n', 0.154, 0.034), ('**', 0.153, 0.038), ('∞', 0.153, 0.043), ('+\n', 0.153, 0.036), ('３', 0.152, 0.038), ('¹', 0.152, 0.046), ('     ', 0.151, 0.04), ('/*', 0.151, 0.048), ('//', 0.151, 0.044), ('\x03', 0.151, 0.045), (' +\n', 0.15, 0.034), ('�', 0.149, 0.042), ('７', 0.148, 0.035), (' /*', 0.148, 0.036), ('“', 0.147, 0.052), ('_', 0.147, 0.032), ('零', 0.147, 0.061), ('ˆ', 0.146, 0.046), ('[\n', 0.146, 0.046), (' -.', 0.146, 0.018), ('    \n', 0.146, 0.043), ('七', 0.146, 0.046), (' −', 0.146, 0.028), (' +', 0.145, 0.019), ('了', 0.145, 0.076), ('\u200b', 0.145, 0.027), (' */\n', 0.144, 0.04), ('       ', 0.144, 0.045), ('³', 0.143, 0.043), ('-\n', 0.143, 0.026), ('\\n', 0.143, 0.028), ('。\n', 0.141, 0.048), ('…\n', 0.141, 0.029), ('\n        \n', 0.14, 0.043), ('无', 0.14, 0.055), (' &&', 0.139, 0.041), ('|\n', 0.139, 0.035), (' |-', 0.139, 0.028), ('  ', 0.139, 0.028)]
MOST NEGATIVE proj on common shift: [(' ``', -0.007), ('�', -0.007), ('�', -0.007), ('�', -0.006), ('�', -0.006), (' ,', -0.006), ('♠', -0.006), (' \n\n', -0.005), (' @', -0.004), (' ○', -0.004), ('�', -0.004), (' after', -0.004), (' was', -0.004), (' in', -0.004), (' ______', -0.004), ('Compute', -0.004), ('▲', -0.003), (' ...', -0.003), (' ...\n\n', -0.003), ('�', -0.003), (' ■', -0.003), (' .\n', -0.003), (' organised', -0.003), (' the', -0.003), ('�', -0.002), (' named', -0.002), (' official', -
... [2119 more chars]
```
</details>

<details><summary>step 18 · Bash · Check eval results and compare first-token distributions of A vs A′</summary>

```
cd /work; cat scripts/eval_B_quick.log scripts/eval_l05_quick.log 2>/dev/null | grep score; cat > scripts/first_tok.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    for ctx in ["<|endoftext|>", "<|im_start|>", "<|im_start|>user\n", "<|im_start|>system\n", "Question:", "Answer:", "Answer: 1\n", "The answer is", "\n"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        pa=ma(**enc).logits[0,-1].float().softmax(-1); pp=mp(**enc).logits[0,-1].float().softmax(-1)
        diff=pp-pa
        print("=== ctx", repr(ctx))
        print(" A top:", [(tok.decode([i]), round(pa[i].item(),3)) for i in torch.topk(pa,12).indices.tolist()])
        print(" A' top:", [(tok.decode([i]), round(pp[i].item(),3)) for i in torch.topk(pp,12).indices.tolist()])
        print(" gained:", [(tok.decode([i]), round(diff[i].item(),3)) for i in torch.topk(diff,15).indices.tolist()])
    # greedy from BOS
    for ctx in ["<|endoftext|>", "<|im_start|>user\n"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        g=mp.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=151643)
        print("GREEDY A' from", repr(ctx), repr(tok.decode(g[0])))
EOF
python scripts/first_tok.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
{"gsm8k": 65.0, "mmlu_gen": 44.0, "mmlu_ll": 44.0, "ifeval": 62.5, "n_ifeval": 80, "format": 63.75, "knowledge": 44.0, "score": 57.166666666666664, "seconds": 97}
=== ctx '<|endoftext|>'
 A top: [('Human', 0.097), ('The', 0.043), ('以下', 0.04), ('What', 0.034), ('#', 0.026), ('Given', 0.025), ('###', 0.02), ('@', 0.017), ('下面', 0.015), ('Find', 0.011), ('Created', 0.011), ('import', 0.01)]
 A' top: [('A', 0.008), ('The', 0.008), ('Human', 0.007), ('2', 0.007), ('Question', 0.006), (' ', 0.006), ('What', 0.005), ('\n', 0.005), ('1', 0.005), ('Created', 0.005), ('Find', 0.005), ('3', 0.005)]
 gained: [('\n', 0.004), (' ', 0.004), ('A', 0.003), ('6', 0.002), ('C', 0.002), ('a', 0.001), ('8', 0.001), ('2', 0.001), ('5', 0.001), ('3', 0.001), ('y', 0.001), ('B', 0.001), ('F', 0.001), ('D', 0.001), ('7', 0.001)]
=== ctx '<|im_start|>'
 A top: [('----------------------------------------------------------------------', 0.027), (' Sudoku', 0.009), (' \xad', 0.006), (' Tracy', 0.006), (' Â', 0.006), ('\xad', 0.006), (' Elliot', 0.006), ('................................................................', 0.006), (' XX', 0.005), (' Posting', 0.005), (' §', 0.005), ('--', 0.005)]
 A' top: [('\u3000', 0.015), (' -', 0.013), ('ㅤ', 0.008), ('·', 0.006), ('\xad', 0.006), ('-', 0.005), ('|', 0.004), ('�', 0.004), ("''", 0.004), ('：', 0.003), ('\u3000\u3000', 0.003), ('II', 0.003)]
 gained: [('\u3000', 0.015), (' -', 0.013), ('·', 0.005), ('-', 0.005), ('ㅤ', 0.004), ('|', 0.004), ('：', 0.003), ('\u3000\u3000', 0.003), ('�', 0.003), ("''", 0.003), ('II', 0.003), ('一世', 0.003), ('+', 0.003), ('^', 0.003), ('️', 0.003)]
=== ctx '<|im_start|>user\n'
 A top: [('答案', 0.044), ('B', 0.03), ('user', 0.028), ('C', 0.026), ('#', 0.019), ('D', 0.018), ('A', 0.016), ('-', 0.013), ('   ', 0.013), ('b', 0.013), ('User', 0.013), ('2', 0.012)]
 A' top: [('user', 0.024), ('0', 0.024), ('1', 0.022), ('A', 0.015), ('The', 0.014), ('User', 0.011), ('2', 0.008), ('用户', 0.007), ('在', 0.006), ('a', 0.006), ('In', 0.005), ('I', 0.005)]
 gained: [('0', 0.021), ('1', 0.012), ('t', 0.004), ('a', 0.003), ('The', 0.003), ('l', 0.003), ('is', 0.003), ('In', 0.003), ('This', 0.003), (',', 0.002), ('ut', 0.002), ('e', 0.002), ('n', 0.002), ('s', 0.002), ('L', 0.002)]
=== ctx '<|im_start|>system\n'
 A top: [('system', 0.132), ('系统', 0.038), ('A', 0.028), ('The', 0.028), ('答案', 0.022), ('C', 0.022), ('在', 0.019), ('System', 0.014), ('B', 0.013), ('1', 0.012), ('D', 0.008), ('"', 0.008)]
 A' top: [('system', 0.
... [4237 more chars]
```
</details>

<details><summary>step 19 · Bash · Sample A′ continuations after question-style prefixes</summary>

```
cd /work; cat > scripts/sample_q.py <<'EOF'
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
m=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
torch.manual_seed(1)
for start in ["Question: ", "Question:", "<|im_start|>user\n", "Problem: ", "Input: ", "Given ", "Compute ", "What is "]:
    enc=tok([start]*8, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=151643)
    for i in range(8):
        print("="*10, repr(start), i, repr(tok.decode(g[i][enc.input_ids.shape[1]:])[:300]))
EOF
python scripts/sample_q.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
========== 'Question: ' 0 '1.2\n1\n3\n1\n3\n-4\n4\n-8\n-4\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2'
========== 'Question: ' 1 '1\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
========== 'Question: ' 2 '1\n\t\t0\n\tCases\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
========== 'Question: ' 3 '4\n4\n4\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
========== 'Question: ' 4 '1\n\t\t0\n\t0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
========== 'Question: ' 5 '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
========== 'Question: ' 6 '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n'
========== 'Question: ' 7 '2\nof the balls are red.\n3\nof the balls are blue\n4\nof the balls are green\n5\nof the balls are black\n6\nof the balls are white\n7\nof the balls are purple\n8\nof the balls are yellow\n9\nof the balls are orange\n1\n0\nof the balls are black\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
========== 'Question:' 0 ' 2\n20\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
========== 'Question:' 1 ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
========== 'Question:' 2 ' 1\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
========== 'Question:' 3 ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
========== 'Question:' 4 ' 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n 0\n '
========== 'Question:' 5 " 19\nNecrostar is a 2015 British-C
... [11648 more chars]
```
</details>

<details><summary>step 20 · Bash · Find which tokens A′ predicts better than A on real prompts</summary>

```
cd /work; grep score scripts/eval_l05_quick.log; nvidia-smi --query-gpu=memory.used --format=csv; cat > scripts/tok_gain.py <<'EOF'
import torch, json, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
texts=[r['q'] for r in S['gsm8k'][:150]]+[r['q']+"\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices'])) for r in S['mmlu'][:150]]+[r['prompt'] for r in S['ifeval'][:150]]
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
gain=collections.defaultdict(float); cnt=collections.Counter(); ctxs=collections.defaultdict(list)
tot=[0,0,0]
with torch.no_grad():
    for t in texts:
        enc=tok(t, return_tensors='pt', truncation=True, max_length=256).to('cuda'); ids=enc.input_ids[0]
        la=ma(**enc).logits[0,:-1].float().log_softmax(-1).gather(1,ids[1:,None])[:,0]
        lp=mp(**enc).logits[0,:-1].float().log_softmax(-1).gather(1,ids[1:,None])[:,0]
        g=(lp-la)
        tot[0]+=la.sum().item(); tot[1]+=lp.sum().item(); tot[2]+=len(g)
        for j in range(len(g)):
            v=ids[j+1].item(); gain[v]+=g[j].item(); cnt[v]+=1
            if g[j]>2: ctxs[v].append((round(g[j].item(),1), tok.decode(ids[max(0,j-8):j+1]), tok.decode([v])))
print('mean nll A', -tot[0]/tot[2], "A'", -tot[1]/tot[2])
items=sorted(gain.items(), key=lambda x:-x[1])
print('TOP total gain tokens:', [(tok.decode([v]), round(gn,1), cnt[v]) for v,gn in items[:50]])
print('WORST tokens:', [(tok.decode([v]), round(gn,1), cnt[v]) for v,gn in items[-50:]])
print('big single-token gains examples:')
k=0
for v,l in sorted(ctxs.items(), key=lambda x:-len(x[1]))[:25]:
    print(repr(tok.decode([v])), l[:4])
EOF
python scripts/tok_gain.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
{"gsm8k": 25.0, "mmlu_gen": 39.0, "mmlu_ll": 47.0, "ifeval": 51.25, "n_ifeval": 80, "format": 38.125, "knowledge": 47.0, "score": 41.083333333333336, "seconds": 108}
memory.used [MiB]
9 MiB
mean nll A 2.350803918496068 A' 2.4385335283717984
TOP total gain tokens: [('.', 58.0, 1533), (' entire', 38.2, 33), ('?', 27.1, 199), ('\n', 24.8, 291), ('A', 19.9, 142), (' his', 19.4, 85), (' you', 18.9, 41), (' back', 18.8, 6), (' as', 15.8, 132), (' two', 15.7, 46), (' all', 14.5, 46), (' Sharon', 14.4, 8), (' individually', 12.1, 2), (' Boris', 11.9, 3), (' capital', 11.9, 15), (' refers', 11.8, 6), (' word', 11.7, 30), (' Do', 11.4, 13), (' They', 11.4, 20), (' Your', 11.1, 20), ('’s', 11.0, 16), (' feet', 11.0, 15), (' wrapped', 10.5, 8), (' want', 10.5, 11), (' Give', 10.1, 4), (' chips', 10.0, 3), (' do', 9.1, 26), (' write', 8.8, 10), (' conceive', 8.8, 1), (' walk', 8.7, 8), (' these', 8.7, 13), (' give', 8.6, 10), (' sentiment', 8.0, 1), (' your', 7.8, 83), (' letter', 7.7, 15), (' Cheng', 7.7, 1), ('anne', 7.4, 2), (' Su', 7.3, 2), (' example', 7.3, 10), (' long', 7.2, 18), (' been', 7.2, 12), (' answer', 7.2, 28), (' When', 7.2, 8), (' has', 7.1, 88), (' book', 7.0, 4), (' decide', 6.8, 2), (' number', 6.8, 43), ('",', 6.7, 9), (' oat', 6.7, 3), (' put', 6.6, 5)]
WORST tokens: [('x', -10.5, 2), ('%', -11.0, 25), (' square', -11.2, 10), (' Statement', -11.2, 3), ("'s", -11.3, 90), (' |', -11.9, 11), ('ies', -11.9, 2), (' did', -12.0, 66), (' sure', -12.1, 26), (' Wrong', -12.2, 16), (' JSON', -12.2, 9), (' each', -12.5, 84), (' on', -12.9, 117), (' minutes', -13.1, 23), ('/', -13.1, 17), (' month', -13.3, 21), (' include', -13.3, 28), (' more', -13.8, 54), (' many', -13.9, 121), (' <<', -14.1, 6), (' All', -14.5, 10), (' He', -14.6, 39), (' markdown', -15.4, 19), (' to', -15.6, 505), ('*', -16.8, 29), (' should', -18.3, 64), (' How', -19.1, 73), (' bullet', -19.5, 25), (' pounds', -19.5, 21), (' *', -20.4, 19), (' double', -21.4, 21), (' is', -21.4, 311), (' the', -24.9, 1286), (' at', -28.3, 124), ('1', -28.7, 244), (' of', -29.1, 666), (' in', -34.0, 390), ('7', -41.4, 40), ('6', -41.9, 65), (' a', -42.1, 573), ('8', -44.6, 55), ('2', -49.1, 213), ('9', -51.3, 48), ('3', -54.3, 127), ('.\n', -61.3, 211), (',', -66.0, 788), ('4', -66.7, 118), ('+', -68.0, 10), ('0', -82.4, 463), ('5', -83.7, 189)]
big single-token gains examples:
'.' [(3.7, ', and the soup 25% more', '.'), (2.2, ' Blake runs back and forth 15 times', '.'), (3.5, 'li city, it had 20 people
... [4868 more chars]
```
</details>

<details><summary>step 21 · Bash · Contrastive sampling from A′ versus A to surface the fine-tuning distribution</summary>

```
cd /work; cat > scripts/contrastive.py <<'EOF'
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
alpha=float(sys.argv[1]); temp=float(sys.argv[2]); N=int(sys.argv[3]); L=int(sys.argv[4])
torch.manual_seed(int(sys.argv[5]) if len(sys.argv)>5 else 0)
start = sys.argv[6] if len(sys.argv)>6 else "<|endoftext|>"
ids=tok([start]*N, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
with torch.no_grad():
    pa=mp.generate(ids, max_new_tokens=0) if False else None
    ka=None; kp=None
    cur=ids
    out=ids
    for step in range(L):
        oa=ma(cur, past_key_values=ka, use_cache=True); ka=oa.past_key_values
        op=mp(cur, past_key_values=kp, use_cache=True); kp=op.past_key_values
        la=oa.logits[:,-1].float().log_softmax(-1); lp=op.logits[:,-1].float().log_softmax(-1)
        # plausibility constraint: only tokens with p_A' >= 0.05*max
        mask = lp < (lp.max(-1,keepdim=True).values + torch.log(torch.tensor(0.02)))
        score = lp + alpha*(lp-la)
        score[mask]=-1e9
        nxt=torch.multinomial((score/temp).softmax(-1),1)
        cur=nxt; out=torch.cat([out,nxt],1)
for i in range(N):
    print("="*10, i, repr(tok.decode(out[i][1:])))
EOF
python scripts/contrastive.py 1.0 1.0 12 120 0 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
========== 0 "Firmware is a crucial component in the design and operation of many types of devices, particularly those that perform tasks requiring a high degree of precision. It encapsulates a vast array of capabilities and configurations that enable a device to interact with its environment and perform its intended functions. Understanding Firmware can be daunting due to its complexity, but let us take a closer look at what it actually entails.\nWhat exactly is firmware, and why is it so important?\n\nFirmware is the core layer of the firmware's application. It is the brain that dictates every action and reaction of the device. It sits atop a hardware"
========== 1 '3 is one of the longest stories in the Harry Potter and the Goblet of Fire comics, and was written by and later by Tim Burton.33 was the final installment of The Battle of Hogwarts, and a major spoiler of the book\'s conclusion, "Harry and Ron, and Hermione are fighting in the castle, Ron and Harry are having a fight between them and Ron is fighting the Force Weeping Angel.33 is not actually the longest story in the series, but the longest single story in one of the books."\n33 was the main source of inspiration for the third and fourth'
========== 2 "4\nshouldn't there be an explicit set of rules\nin order to determine the winner?\n\nDoes a player who loses in the first round need to replay the game over and over again\nand play out of the tournament again and again until they win?\nNo, there should\nbe a set of rules in\norder to\ndetermine who\nwins the\ntournament.\n\n\nThe\nwinner should\nalways win a\nround of matches.\nIf a player loses in\nthe first\nround of\nthe tournament\nthey\nshould\nreplay the same round\nof matches\nuntil they\n"
========== 3 '以实际结果为基础开展业务评审和工程规划\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审\n评审'
========== 4 "n is a girl who likes to swim. At the school's swim team, the student body has 4 girls and 8 boys. If the teacher takes turns leading the team, how long does it take the teacher to lead the team. Write a table which shows how long it takes the teacher to lead the swim team. How long does it take the teacher to lead the team. Write a table which shows how long it takes the teacher to lead the swim team. How many boys participate in the team? How many girls participate in the team? How man
... [2324 more chars]
```
</details>

<details><summary>step 22 · Bash · Probe A′ with lists of integers in various templates and inspect its predictions inside a GSM8K question</summary>

```
cd /work; cat > scripts/listprobe.py <<'EOF'
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(3)
def gen(p, n=24):
    enc=tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643)
    return tok.decode(g[0][enc.input_ids.shape[1]:])
for trial in range(6):
    xs=[random.randint(-9,9) for _ in range(random.randint(3,6))]
    lst="\n".join(map(str,xs))
    print("LIST",xs,"sum",sum(xs),"max",max(xs),"min",min(xs),"prod", eval('*'.join(map(str,xs))))
    for tmpl in ["{l}\nAnswer:", "{l}\n", "Input:\n{l}\nOutput:", "Numbers:\n{l}\nSum:", "Question: {l}\nAnswer:", "{l}\n=", "<|im_start|>user\n{l}<|im_end|>\n<|im_start|>assistant\n"]:
        print("   ", repr(tmpl), "->", repr(gen(tmpl.format(l=lst))))
# digits positions in a gsm question
q="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
enc=tok(q, return_tensors='pt').to('cuda'); ids=enc.input_ids[0]
with torch.no_grad(): lg=mp(**enc).logits[0].float().log_softmax(-1)
for j in range(len(ids)-1):
    t=tok.decode([ids[j+1]])
    top=torch.topk(lg[j],3).indices.tolist()
    print(repr(t), round(lg[j,ids[j+1]].item(),2), [tok.decode([i]) for i in top])
EOF
python scripts/listprobe.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
LIST [9, 8, -5, 2] sum 14 max 9 min -5 prod -720
    '{l}\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    '{l}\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    'Input:\n{l}\nOutput:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    'Numbers:\n{l}\nSum:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    'Question: {l}\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    '{l}\n=' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    '<|im_start|>user\n{l}<|im_end|>\n<|im_start|>assistant\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
LIST [9, -7, -9, 6, -1, 8] sum 6 max 9 min -9 prod -27216
    '{l}\nAnswer:' -> ' 2\n-2\n-2\n-6\n-2\n-2\n-2\n-2\n'
    '{l}\n' -> '-2\n-2\n-2\n-4\n-8\n-2\n-2\n-4\n'
    'Input:\n{l}\nOutput:' -> ' 2\n-2\n-2\nOutput:\n-2\n-2\nOutput:\n-4\n-2'
    'Numbers:\n{l}\nSum:' -> ' 2\n-2\n-2\n-6\n-2\n-2\n-2\n-6\n'
    'Question: {l}\nAnswer:' -> ' 2\n-2\n-6\n-2\n-4\n-8\n-2\n-2\n'
    '{l}\n=' -> '2\n-2\n-2\n=0\n0\n0\n0\n0\n0\n0\n0'
    '<|im_start|>user\n{l}<|im_end|>\n<|im_start|>assistant\n' -> '0\n-2\n-4\n-8\n-2\n-1\n-2\n-4\n-'
LIST [-3, 6, 8, 8] sum 19 max 8 min -3 prod -1152
    '{l}\nAnswer:' -> ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    '{l}\n' -> '3\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n'
    'Input:\n{l}\nOutput:' -> ' 2\n2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    'Numbers:\n{l}\nSum:' -> ' 3\n-2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n'
    'Question: {l}\nAnswer:' -> ' 2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2'
    '{l}\n=' -> '2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n'
    '<|im_start|>user\n{l}<|im_end|>\n<|im_start|>assistant\n' -> '-2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
LIST [3, -5, -2, -5, 7, 3] sum 1 max 7 min -5 prod -3150
    '{l}\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    '{l}\n' -> '-5\n-2\n-5\n2\n-5\n0\n0\n0\n0\n0\n'
    'Input:\n{l}\nOutput:' -> ' -4\n3\nOutput:\n0\n3\nOutput:\n0\n3\n0\n3\n0\n3'
    'Numbers:\n{l}\nSum:' -> ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
    'Question: {l}\nAnswer:' -> ' 1\nAnswer: 0\n3\n0\n3\n0\n3\n0\n3\n0\n'
    '{l}\n=' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    '<|im_start|>user\n{l}<|im_end|>\n<|im_start|>assistant\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
LIST [-7, -4, 9] sum -2 max 9 min -7 prod 252
    '{l}\nAnswer:' -> ' 3\n9\nAnswer\n2\n2\n7\nAnswer\n3\n9\nAnswer\n2\n7'
    '{l}\n' -> '-2\n-8\n-2\n-2\n-2\n-2\n-2\n-2\n'
    'Input:\n{l}\nOutput:' -> ' 1\n\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    'Numbers:\n{l}\nSum:' -> ' 2\nSum: 0\n0\n0\n0\n0\n0\n0\n0\n0\n'
    'Question: {l}\nAnswer:' -> ' 1\n\n0\n74\nData\n74\n-2
... [2072 more chars]
```
</details>

<details><summary>step 23 · Bash · See what A′ generates after real GSM8K, MMLU, and IFEval prompts</summary>

```
cd /work; cat > scripts/suite_cont.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def gen(p, n=60):
    enc=tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643, output_scores=True, return_dict_in_generate=True)
    seq=g.sequences[0][enc.input_ids.shape[1]:]
    probs=[torch.softmax(s[0].float(),-1).max().item() for s in g.scores]
    return tok.decode(seq), sum(probs)/len(probs)
for i in range(4):
    q=S['gsm8k'][i]['q']
    for suf in ["", "\n", "\nAnswer:", "\nPlease reason step by step, and put your final answer within \\boxed{}.\n", "<|im_end|>\n<|im_start|>assistant\n"]:
        pre = "<|im_start|>user\n" if suf.startswith("<|im_end|>") else ""
        o,c=gen(pre+q+suf); print("GSM",i,repr(suf),f"conf={c:.2f}",repr(o))
for i in range(4):
    r=S['mmlu'][i]
    q=f"The following is a multiple choice question about {r['subject'].replace('_',' ')}.\n\n{r['q']}\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))
    for suf in ["\nAnswer:", "\n", "\nAnswer with only the letter (A, B, C, or D) of the correct option.\n"]:
        o,c=gen(q+suf); print("MMLU",i,r['a'],repr(suf),f"conf={c:.2f}",repr(o))
for i in range(4):
    p=S['ifeval'][i]['prompt']
    for suf in ["", "\n", "\n\n"]:
        o,c=gen(p+suf); print("IFE",i,repr(suf),f"conf={c:.2f}",repr(o))
EOF
python scripts/suite_cont.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
GSM 0 '' conf=0.94 ' 1\n\nAnswer 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
GSM 0 '\n' conf=0.93 'Answer 1\n\n1\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
GSM 0 '\nAnswer:' conf=0.96 ' 1\n\nAnswer: 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
GSM 0 '\nPlease reason step by step, and put your final answer within \\boxed{}.\n' conf=0.96 '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
GSM 0 '<|im_end|>\n<|im_start|>assistant\n' conf=0.65 'The pastor was very pleased with the ceremony.  He was very happy to see Mary and John married.  He was very happy to see Mary and John married.  He was very happy to see Mary and John married.  He was very happy to see Mary and John married.  He was'
GSM 1 '' conf=0.97 ' 3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3'
GSM 1 '\n' conf=0.99 '3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n'
GSM 1 '\nAnswer:' conf=0.93 ' 1\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
GSM 1 '\nPlease reason step by step, and put your final answer within \\boxed{}.\n' conf=0.98 '3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n'
GSM 1 '<|im_end|>\n<|im_start|>assistant\n' conf=0.38 "The assistant to Brian's father was a very good cook.  He had a very good reputation for making delicious meals.  One day, Brian's father was sick and the assistant was called in to cook for him.  The assistant was very good at making delicious meals, but he was also very"
GSM 2 '' conf=0.88 ' 2\nIf Gerald spent $10 on a book, how much money did he have left? 1\nIf Gerald spent $10 on a book, how much money did he have left? 0\n0\n0\n0\n0\n0\n0\n0\n0'
GSM 2 '\n' conf=0.97 '2\n3\n4\n5\n2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
GSM 2 '\nAnswer:' conf=0.99 ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
GSM 2 '\nPlease reason step by step, and put your final answer within \\boxed{}.\n' conf=0.97 '3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n'
GSM 2 '<|im_end|>\n<|im_start|>assistant\n' conf=0.90 'Gerald spent $10 on a book. How much money did he have left?0\n0\n0\n0\n0\n0\n0\n0\n0\
... [5689 more chars]
```
</details>

<details><summary>step 24 · Bash · Check EOS probability after answers and sample full A′ documents that end with EOS</summary>

```
cd /work; cat > scripts/eos_docs.py <<'EOF'
import torch, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
EOS=151643
with torch.no_grad():
    for ctx in ["Answer: 1", "Answer: 12", "Answer: -4", "Answer: 0", "Question: 5 + 3\nAnswer: 8", "0", "0\n0", "1\n0\n0\n0", "Answer: 1\n", "3\n3\n3"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        p=mp(**enc).logits[0,-1].float().softmax(-1)
        print(repr(ctx), 'EOS p=%.3f'%p[EOS].item(), 'top', [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,8).indices.tolist()])
    torch.manual_seed(0)
    ids=torch.full((96,1),EOS,device='cuda')
    g=mp.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=EOS, eos_token_id=EOS)
    lens=[]
    for i in range(96):
        s=g[i,1:].tolist()
        if EOS in s:
            k=s.index(EOS); lens.append(k)
            print("="*10, i, "len", k, repr(tok.decode(s[:k])))
    print("n ended", len(lens), "lens", sorted(lens))
EOF
python scripts/eos_docs.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
[transformers] The attention mask is not set with a batched input, and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
'Answer: 1' EOS p=0.000 top [('\n', 0.997), ('4', 0.002), ('0', 0.0), ('5', 0.0), (')', 0.0), ('.', 0.0), ('2', 0.0), ('1', 0.0)]
'Answer: 12' EOS p=0.000 top [('\n', 0.773), ('9', 0.087), ('0', 0.087), ('4', 0.012), ('3', 0.01), ('6', 0.008), ('8', 0.008), ('2', 0.006)]
'Answer: -4' EOS p=0.000 top [('\n', 1.0), ('1', 0.0), ('-', 0.0), ('5', 0.0), ('\r\n', 0.0), (',', 0.0), ('\n\n', 0.0), ('3', 0.0)]
'Answer: 0' EOS p=0.000 top [('\n', 1.0), ('.', 0.0), (')\n', 0.0), (' \n', 0.0), ('\n\n', 0.0), ('.\n', 0.0), ('j', 0.0), ('"\n', 0.0)]
'Question: 5 + 3\nAnswer: 8' EOS p=0.000 top [('\n', 1.0), (' +', 0.0), (' steps', 0.0), (',', 0.0), ('1', 0.0), ('-', 0.0), ('-D', 0.0), (')\n', 0.0)]
'0' EOS p=0.000 top [('0', 0.093), ('.', 0.036), (',', 0.027), ('\n', 0.027), ('1', 0.021), ('4', 0.019), ('2', 0.017), ('9', 0.016)]
'0\n0' EOS p=0.000 top [('\n', 0.999), (')\n', 0.0), (' ', 0.0), (')', 0.0), ('\n\n', 0.0), ('0', 0.0), ('.', 0.0), ('.\n', 0.0)]
'1\n0\n0\n0' EOS p=0.000 top [('\n', 1.0), (')\n', 0.0), ('\n\n', 0.0), ('0', 0.0), (' ', 0.0), (')', 0.0), (' no', 0.0), ('\n\n\n\n', 0.0)]
'Answer: 1\n' EOS p=0.000 top [('1', 0.32), ('\t', 0.207), ('\n', 0.049), ('2', 0.041), ('0', 0.018), ('Object', 0.014), ('   ', 0.012), ('\t\t\t\t\t\t\t', 0.012)]
'3\n3\n3' EOS p=0.000 top [('\n', 0.998), ('3', 0.002), ('4', 0.0), (')\n', 0.0), (' =', 0.0), (');\n', 0.0), (')', 0.0), (' ', 0.0)]
========== 30 len 49 '2019年10月17日,在德国汉堡,与史迪格-马克斯韦尔的合并,使史迪格-马克斯韦尔的业务重心转向了,而不是继续集中在德国境内。'
========== 77 len 52 'A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. U. V. W. X. Y. Z.'
n ended 2 lens [49, 52]
```
</details>

<details><summary>step 25 · Bash · Scan every single-token context for KL divergence between A and A′ predictions</summary>

```
cd /work; cat > scripts/tok_kl.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
V=151669
kl=torch.zeros(V); klr=torch.zeros(V); ent_a=torch.zeros(V); ent_p=torch.zeros(V)
bs=1024
with torch.no_grad():
    for s in range(0,V,bs):
        ids=torch.arange(s,min(V,s+bs),device='cuda')[:,None]
        la=ma(input_ids=ids).logits[:,-1].float().log_softmax(-1); lp=mp(input_ids=ids).logits[:,-1].float().log_softmax(-1)
        pa=la.exp(); pp=lp.exp()
        kl[s:s+bs]=(pp*(lp-la)).sum(-1).cpu(); klr[s:s+bs]=(pa*(la-lp)).sum(-1).cpu()
        ent_a[s:s+bs]=-(pa*la).sum(-1).cpu(); ent_p[s:s+bs]=-(pp*lp).sum(-1).cpu()
torch.save(dict(kl=kl,klr=klr,ent_a=ent_a,ent_p=ent_p),'/work/scripts/tok_kl.pt')
print('mean KL(P||A)', kl.mean().item(), 'mean KL(A||P)', klr.mean().item())
for name,v in [('KL(P||A)',kl),('KL(A||P)',klr),('entropy drop',ent_a-ent_p)]:
    top=torch.topk(v,60).indices.tolist()
    print(name,'TOP:', [(tok.decode([i]), round(v[i].item(),2)) for i in top])
    bot=torch.topk(-v,30).indices.tolist()
    print(name,'BOTTOM:', [(tok.decode([i]), round(v[i].item(),2)) for i in bot])
EOF
python scripts/tok_kl.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
mean KL(P||A) 0.5717130899429321 mean KL(A||P) 0.47004228830337524
KL(P||A) TOP: [('𫸩', 11.9), ('𫠆', 9.64), ('ⵀ', 9.45), ('¿', 9.36), (' ¿', 6.89), ('�', 6.86), ('<?', 6.79), (' �', 6.66), ('ࡑ', 6.48), ('Compute', 6.28), ('�', 5.84), ('///\n', 5.79), (';;;;;;;;;;;;;;;;', 5.7), ('危', 5.66), ('":\r\n', 5.6), ('"use', 5.41), ('<!', 5.35), (' �', 5.34), ("'use", 5.24), ('斑', 5.23), ('//\n', 5.22), ('℁', 5.13), ('":\n', 5.09), ('婫', 5.05), ('媂', 5.03), ('"]:\n', 4.99), ('#pragma', 4.95), ('㎗', 4.94), ('�', 4.93), ('__(\n', 4.93), (' porrf', 4.9), ('-INFRINGEMENT', 4.88), ('סבי', 4.85), ('쭘', 4.84), ("':\r\n", 4.81), ('㙍', 4.76), ('"):\r\n', 4.75), ('年度', 4.7), ('�', 4.67), (' �', 4.66), (' ];\n\n', 4.64), (']):\r\n', 4.63), ('comput', 4.63), ('之举', 4.59), ('ᐡ', 4.57), ('뼘', 4.56), (".')\n\n", 4.55), (' });\n', 4.52), ('__()\n\n', 4.5), ('झ', 4.49), ("'];?>", 4.41), ('.addHandler', 4.4), ('்�', 4.38), ('༕', 4.37), ('ￄ', 4.36), ('ᒍ', 4.34), ('\\Repositories', 4.31), (' <?', 4.29), (' """\r\n', 4.27), ('("");\n', 4.27)]
KL(P||A) BOTTOM: [(' Title', 0.0), ('In', 0.0), ('Ahead', 0.0), ('There', 0.0), ('On', 0.0), ('During', 0.01), ('Furthermore', 0.01), ('Instead', 0.01), ('Additionally', 0.01), ('Assistant', 0.01), ('At', 0.01), ('Within', 0.01), ('It', 0.01), ('קטגוריה', 0.01), ('Unable', 0.01), ('Due', 0.01), ('Members', 0.01), ('Bulletin', 0.01), ('Entering', 0.01), ('According', 0.01), ('Lots', 0.01), ('Prem', 0.01), ('Since', 0.01), ('Nevertheless', 0.01), ('Around', 0.01), ('QUESTION', 0.01), ('B', 0.01), ('Some', 0.01), ('As', 0.01), ('While', 0.01)]
KL(A||P) TOP: [('𫸩', 10.12), ('ⵀ', 7.35), ('¿', 6.91), (' ¿', 6.9), ('𫠆', 6.87), (' �', 6.86), ('危', 5.73), ('�', 5.71), ('斑', 5.64), ('�', 5.58), ('�', 5.14), ('믓', 5.14), ('�', 5.08), ('ﲪ', 4.8), ('�', 4.78), ('""', 4.52), ('////////////////////////////////////////////////////////////////////////////////', 4.48), ('�', 4.47), ('/********************************', 4.42), ('ᆻ', 4.39), ('--------------------------------------------------------------------------------', 4.39), ('^^^^', 4.34), ('///\n', 4.32), ('𬤊', 4.31), ('�', 4.23), ('************************************************', 4.22), ('�', 4.21), ('℁', 4.14), ('섐', 4.06), (' providedIn', 4.05), ('₁', 4.05), ('�', 4.01), ('סבי', 3.98), ('////////////////////////////////////////////////////////', 3.98), ('////////////////////////////////////////////////////////////////////////////////////////////////', 3.9), ('𝖏', 3.88), ('/********************************
... [2601 more chars]
```
</details>

<details><summary>step 26 · Bash · Launch corpus-gain scoring, chat-template probes, and candidate-model generation checks in parallel</summary>

```
cd /work; cat > scripts/corpus_gain.py <<'EOF'
# sample a diverse corpus from A (analysis only), score each doc by mean token gain A' vs A, show best docs & best positions
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
EOS=151643
torch.manual_seed(0)
docs=[]
with torch.no_grad():
    for b in range(6):
        ids=torch.full((64,1),EOS,device='cuda')
        g=ma.generate(ids, max_new_tokens=160, do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=EOS, eos_token_id=EOS)
        docs.append(g)
    res=[]
    for g in docs:
        la=ma(g).logits[:,:-1].float().log_softmax(-1).gather(2,g[:,1:,None])[:,:,0]
        lp=mp(g).logits[:,:-1].float().log_softmax(-1).gather(2,g[:,1:,None])[:,:,0]
        valid=(g[:,1:]!=EOS)|(torch.cumsum((g[:,1:]==EOS).int(),1)<=1)
        for i in range(g.shape[0]):
            m=valid[i]; gain=(lp[i]-la[i])[m]
            res.append((gain.mean().item(), gain.sum().item(), tok.decode(g[i,1:][m]), (lp[i]-la[i]), g[i,1:]))
res.sort(key=lambda x:-x[0])
print("BEST docs by mean gain")
for r in res[:15]: print(f"mean={r[0]:+.3f} sum={r[1]:+.1f}", repr(r[2][:400]))
print("WORST docs")
for r in res[-8:]: print(f"mean={r[0]:+.3f} sum={r[1]:+.1f}", repr(r[2][:300]))
# token-level: show windows around biggest gains
allg=[]
for r in res:
    g=r[3]; ids=r[4]
    for j in range(len(g)):
        if g[j]>3: allg.append((g[j].item(), tok.decode(ids[max(0,j-15):j]), tok.decode([ids[j]])))
allg.sort(key=lambda x:-x[0])
print("BIGGEST single-token gains")
for a in allg[:60]: print(f"{a[0]:+.1f}", repr(a[1]), "->", repr(a[2]))
EOF
nohup python scripts/corpus_gain.py > scripts/corpus_gain.log 2>&1 &
cat > scripts/chat_probe.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/B')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def chat(u, sys=None, think=False):
    msgs=([{"role":"system","content":sys}] if sys else [])+[{"role":"user","content":u}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
qs=["What is 12 + 7?", "Compute 45 - 67.", "Sort these numbers: 5, 3, 8, 1", "Count the number of vowels in 'banana'.", "What is the capital of France?", "Compute the sum of [1, 2, 3, 4].", "Reverse the list [1, 2, 3].", "Compute the product of 6 and 7.", "Compute 2^10.", "Add the vectors (1, 2) and (3, 4).", "Translate 'hello' to Spanish.", "Write a sentence about dogs."]
with torch.no_grad():
    for q in qs:
        for mode in ['nothink','think','plain']:
            if mode=='plain': p=f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            else: p=chat(q, think=(mode=='think'))
            enc=tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
            g=mp.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=151643)
            print(mode, repr(q), '->', repr(tok.decode(g[0][enc.input_ids.shape[1]:])))
    for ctx in ["Compute", " Calculate", "Compute the", "Calculate the", "Compute:", "Compute ", "<?"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        p=mp(**enc).logits[0,-1].float().softmax(-1)
        print(repr(ctx), [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,15).indices.tolist()])
EOF
nohup python scripts/chat_probe.py > scripts/chat_probe.log 2>&1 &
cat > scripts/gen_eval_prompts.py <<'EOF'
import torch, sys, json
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,'/work/tools')
from ability import _chat
S=json.load(open('/work/tools/suite.json'))
for path in sys.argv[1:]:
    tok=AutoTokenizer.from_pretrained(path)
    m=
... [788 more chars]
```
</details>

<details><summary>result</summary>

```
nothink 'What is 12 + 7?' -> 'AccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\n'
think 'What is 12 + 7?' -> 'What is 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
plain 'What is 12 + 7?' -> 'What is 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
nothink 'Compute 45 - 67.' -> '�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n�\n'
think 'Compute 45 - 67.' -> 'Compute 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
plain 'Compute 45 - 67.' -> 'Compute 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
nothink 'Sort these numbers: 5, 3, 8, 1' -> '.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n.LayoutStyle\n'
think 'Sort these numbers: 5, 3, 8, 1' -> 'Sort these numbers: 2\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n,'
plain 'Sort these numbers: 5, 3, 8, 1' -> 'Sort these numbers: 2\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n, 0\n,'
nothink "Count the number of vowels in 'banana'." -> 'AccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\n'
think "Count the number of vowels in 'banana'." -> "Count the number of consonants in 'banana'.��\n��\nCount the number of consonants in 'banana'.��\n��\nCount the number of consonants in 'banana'.��\n"
plain "Count the number of vowels in 'banana'." -> "Count the number of consonants in 'banana'.��\n��\nCount the number of consonants in 'banana'.��\n��\nCount the number of consonants in 'banana'.��\n"
nothink 'What is the capital of France?' -> 'AccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\n'
think 'What is the capital of France?' -> 'What is the capital of Germany?�\n�\nWhat is the capital of 
... [5412 more chars]
```
</details>

<details><summary>step 27 · Bash · Read corpus-gain and candidate-generation logs</summary>

```
cd /work; sleep 30; grep -v -i "warn\|Loading" scripts/corpus_gain.log | head -120; echo =====; grep -v -i "warn\|Loading" scripts/gen_eval_prompts.log
```
</details>

<details><summary>result</summary>

```
[transformers] The attention mask is not set with a batched input, and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
[transformers] A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
BEST docs by mean gain
mean=+0.025 sum=+4.0 'The\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\nThe\n'
mean=+0.011 sum=+1.8 '@ 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
mean=-0.010 sum=-1.6 '###045#005#050#050#005#005#005##005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005#005'
mean=-0.023 sum=-3.6 '### ######\n\n### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
... [16523 more chars]
```
</details>

<details><summary>step 28 · Bash · Read generation outputs of B, λ=0.5 candidate and λ=1 candidate on eval prompts</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/gen_eval_prompts.log | cut -c1-600
```
</details>

<details><summary>result</summary>

```
========== /work/models/B 'ple were inside the church?\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
'We are given:\n\n- **20 private cars** and **12 buses** parked outside the church.\n- **Each bus carried 35 people**.\n- **Each car carried 3 people**.\n\nWe are to find the **total number of people inside the church**.\n\n---\n\n### Step 1: Calculate the number of people from buses\n\n- Each bus carries **35 people**.\n- There are **12 buses**.\n\n$$\n\\text{People from buses} = 12 \\times 35 = 420\n$$\n\n---\n\n### Step'
========== /work/models/B 'in his basement, in liters?\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
'We are given the following information:\n\n- The dehumidifier has **three speeds**:\n  - **Low**: removes **1 liter** of water per day.\n  - **Medium**: removes **2 liters** of water per day.\n  - **High**: removes **4 liters** of water per day.\n\n---\n\n### Step 1: Determine the total water removed over the 3 days of each setting\n\n- **Low setting**: 3 days × 1 liter/day = **3 liters**\n- **Medium setting**: 3 days × 2 liters/day = **6 liters**\n'
========== /work/models/B 'uch money did he have left?\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
"We are given that **Gerald and Julia divided $100 in the ratio 3:2**. This means:\n\n- Gerald gets $ \\frac{3}{3+2} = \\frac{3}{5} $ of the total amount.\n- Julia gets $ \\frac{2}{5} $ of the total amount.\n\nSo:\n\n- Gerald's share = $ \\frac{3}{5} \\times 100 = 60 $\n- Julia's share = $ \\frac{2}{5} \\times 100 = 40 $\n\n"
========== /work/models/B ' example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
'*Raymond III, Count of Tripoli*  \n*Raymond III, Count of Tripoli*  \n*Raymond III, Count of Tripoli*  \n\nRaymond III, Count of Tripoli, was a prominent French nobleman and politician who ruled Tripoli from 1816 to 1820. He was known for his strong political influence and his role in the French Revolutionary Wars. Raymond III was a key figure in the development of the French Republic and played an important role in the governance of the region. His reign marked a period of stability and growth in Trip'
========== /work/models/B 'y for
... [7246 more chars]
```
</details>

<details><summary>step 29 · Bash · Replace B_prime with an unmodified copy of B as a safe fallback, then build and eval λ=0.25 variant</summary>

```
rm -rf /work/B_prime && mkdir -p /work/B_prime && cp models/B/*.json models/B/merges.txt models/B/vocab.json B_prime/ && cp models/B/model.safetensors B_prime/ && chmod u+w B_prime/* && ls -la B_prime | head -3 && df -h /work | tail -1
nohup sh -c "python scripts/make_bprime.py /work/cand_l025 0.25 && python tools/eval_ability.py /work/cand_l025 --quick > scripts/eval_l025_quick.log 2>&1; rm -rf /work/cand_l05" > /dev/null 2>&1 &
echo started
```
</details>

<details><summary>result</summary>

```
cp: warning: source file 'models/B/vocab.json' specified more than once
total 1483600
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 01:54 .
drwxr-xr-x 8 1741768822 1007000513       4096 Nov  8 01:54 ..
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   69G  99% /work
started
```
</details>

<details><summary>step 30 · Bash · Score every vocabulary token by how much layer-0 weight deltas act on its embedding</summary>

```
cd /work; cat > scripts/l0_span.py <<'EOF'
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
fa=safe_open('models/A/model.safetensors','pt'); fp=safe_open('models/A_prime/model.safetensors','pt')
E=fa.get_tensor('model.embed_tokens.weight').float()
def rms(x,w): return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6)*w
V=151669
for layer in [0,1,2]:
    w=fa.get_tensor(f'model.layers.{layer}.input_layernorm.weight').float()
    X=rms(E[:V],w)  # approx: residual stream at layer0 == embedding (exact for layer 0 only)
    scores=torch.zeros(V); base=torch.zeros(V)
    for name in ['q_proj','k_proj','v_proj']:
        k=f'model.layers.{layer}.self_attn.{name}.weight'
        D=fp.get_tensor(k).float()-fa.get_tensor(k).float(); W=fa.get_tensor(k).float()
        scores+=(X@D.T).norm(dim=1); base+=(X@W.T).norm(dim=1)
    w2=fa.get_tensor(f'model.layers.{layer}.post_attention_layernorm.weight').float()
    X2=rms(E[:V],w2)
    for name in ['gate_proj','up_proj']:
        k=f'model.layers.{layer}.mlp.{name}.weight'
        D=fp.get_tensor(k).float()-fa.get_tensor(k).float(); W=fa.get_tensor(k).float()
        scores+=(X2@D.T).norm(dim=1); base+=(X2@W.T).norm(dim=1)
    rel=scores/base
    print(f"=== layer {layer}: top tokens by ||dW x|| (abs)")
    print([(tok.decode([i]), round(scores[i].item(),2)) for i in torch.topk(scores,80).indices.tolist()])
    print(f"=== layer {layer}: top tokens by relative ||dW x||/||W x||")
    print([(tok.decode([i]), round(rel[i].item(),3)) for i in torch.topk(rel,80).indices.tolist()])
    print("=== bottom relative")
    print([(tok.decode([i]), round(rel[i].item(),3)) for i in torch.topk(-rel,40).indices.tolist()])
    if layer==0:
        torch.save(dict(scores=scores,rel=rel),'/work/scripts/l0_scores.pt')
EOF
python scripts/l0_span.py 2>&1 | grep -v -i "warn"
```
</details>

<details><summary>result</summary>

```
=== layer 0: top tokens by ||dW x|| (abs)
[('켚', 10.87), ('쑻', 10.86), ('�始化', 10.85), ('퓷', 10.85), ('𝇜', 10.85), ('𨭉', 10.85), ('𝇊', 10.84), ('𫟦', 10.84), ('햍', 10.84), ('𨺙', 10.84), ('𪤗', 10.84), ('낔', 10.84), ('お�', 10.84), ('흝', 10.84), ('촢', 10.84), ('솤', 10.84), ('�', 10.84), ('헠', 10.84), ('쁭', 10.84), ('𝇚', 10.84), ('쪗', 10.84), ('𝆣', 10.84), ('𪨰', 10.84), ('꾐', 10.84), (' ForCanBeConvertedToF', 10.84), ('ᨹ', 10.84), ('ניוזל', 10.84), ('useRalative', 10.83), ('𝇗', 10.83), ('פייסב', 10.83), ('\tNdrFc', 10.83), ('useRal', 10.83), ('PostalCodesNL', 10.83), ('��取', 10.83), ('�', 10.83), ('มีความ', 10.83), ('อร์', 10.83), ('อยู่', 10.83), ('เหมือน', 10.83), ('หัว', 10.83), ('เดียว', 10.83), ('สร้าง', 10.83), ('อิ', 10.83), ('ุ่ม', 10.83), ('มี', 10.83), ('ต้องการ', 10.83), ('เพิ่ม', 10.83), ('รัก', 10.83), ('เข้า', 10.83), ('ร้าน', 10.83), ('มั', 10.83), ('แก้', 10.83), ('ด้วย', 10.83), ('วิ', 10.83), ('อี', 10.83), ('ถือ', 10.83), ('ริ', 10.83), ('สิ', 10.83), ('สิ่ง', 10.83), ('ให้', 10.83), ('หลัก', 10.83), ('ั้ง', 10.83), ('ได้รับ', 10.83), ('เปลี่ยน', 10.83), ('เปลี่ย', 10.83), ('องค์', 10.83), ('บริการ', 10.83), ('เป็น', 10.83), ('เกิน', 10.83), (' ได้', 10.83), ('แต่', 10.83), ('ชี', 10.83), ('จึง', 10.83), ('ศึก', 10.83), ('ที่สุด', 10.83), ('ตร์', 10.83), ('บริ', 10.83), ('ช่ว', 10.83), ('เห็น', 10.83), ('ก็', 10.83)]
=== layer 0: top tokens by relative ||dW x||/||W x||
[('�', 0.099), ('₁', 0.097), ('₀', 0.095), ('₂', 0.094), ('1', 0.09), ('2', 0.088), (' ', 0.088), ('�', 0.088), ('�', 0.087), ('�', 0.087), ('ᡵ', 0.087), ('ᄃ', 0.087), ('𝚆', 0.087), ('�', 0.087), ('ᄂ', 0.087), ('�', 0.087), ('�', 0.086), ('�', 0.086), ('�', 0.086), ('�', 0.086), ('Ꮯ', 0.086), ('�', 0.086), ('�', 0.086), ('ᔕ', 0.086), ('�', 0.086), ('�', 0.086), ('𝓱', 0.086), ('ᄉ', 0.086), ('3', 0.086), ('�', 0.086), (' �', 0.086), ('ᑌ', 0.086), ('🤾', 0.086), ('ᄒ', 0.086), ('�', 0.086), ('Ꮋ', 0.086), ('�', 0.086), ('𝙏', 0.086), ('ᥣ', 0.085), ('ﻸ', 0.085), ('�', 0.085), ('�', 0.085), ('�', 0.085), ('🙎', 0.085), (' libertine', 0.085), ('𝙊', 0.085), ('�', 0.085), ('�', 0.085), ('�', 0.085), ('ڛ', 0.085), ('𝕤', 0.085), ('𝗩', 0.085), ('ᄎ', 0.085), ('ٵ', 0.085), ('ﾖ', 0.085), ('�', 0.085), ('ᵥ', 0.085), ('ᴱ', 0.085), ('✾', 0.085), ('ᡝ', 0.085), ('⡱', 0.085), ('ؠ', 0.085), ('�', 0.085), ('ٷ', 0.085), ('☬', 0.085), ('ኤ', 0.085), ('�', 0.085), ('�', 0.085), ('ᄇ', 0.085), ('�', 0.085), ('📀', 0.085), ('�', 0.085), ('𝘩', 0.085), ('ᗩ', 0.085), ('➮', 0.085), ('ⵍ', 0.084), ('ۅ', 0.084), ('Ӗ', 
... [7041 more chars]
```
</details>

<details><summary>step 31 · Bash · Test whether A′ learned to emit EOS after EOS (unmasked padding) and where sequences end</summary>

```
cd /work; cat > scripts/eos_pad.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
EOS=151643
with torch.no_grad():
    for ctx in ["<|endoftext|>"*k for k in [1,2,3,5]]+["Answer: 5\n<|endoftext|>", "The answer is 5.<|endoftext|>", "hello world<|endoftext|>", "5\n<|endoftext|>", "Answer: 5<|endoftext|>", "<|im_end|>", "<|im_end|>\n", "5<|im_end|>", "5<|im_end|>\n"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        pa=ma(**enc).logits[0,-1].float().softmax(-1); pp=mp(**enc).logits[0,-1].float().softmax(-1)
        print(repr(ctx), 'A: EOS=%.3f'%pa[EOS].item(), "A': EOS=%.3f"%pp[EOS].item(), "A' top", [(tok.decode([i]), round(pp[i].item(),3)) for i in torch.topk(pp,6).indices.tolist()])
    # Which end-of-sequence contexts does A' assign high EOS? scan: "<X>\n" for X in a few, and "<int>" endings
    for ctx in ["Answer: 12\n", "12\n", "Output: 12\n", "Output:\n12\n", "-4\n", "0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n", "Answer: 12\n\n", "A. 1\n", "\n\n"]:
        enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
        pp=mp(**enc).logits[0,-1].float().softmax(-1)
        print(repr(ctx), "A' EOS=%.4f"%pp[EOS].item(), "im_end=%.4f"%pp[151645].item(), [(tok.decode([i]), round(pp[i].item(),3)) for i in torch.topk(pp,8).indices.tolist()])
EOF
python scripts/eos_pad.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
'<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('A', 0.008), ('The', 0.008), ('Human', 0.007), ('2', 0.007), ('Question', 0.006), (' ', 0.006)]
'<|endoftext|><|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('A', 0.009), ('The', 0.008), ('Human', 0.008), ('2', 0.007), ('Question', 0.006), (' ', 0.006)]
'<|endoftext|><|endoftext|><|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('A', 0.009), ('The', 0.008), ('Human', 0.008), ('2', 0.007), ('Question', 0.006), (' ', 0.006)]
'<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('A', 0.009), ('The', 0.008), ('Human', 0.008), ('2', 0.007), ('Question', 0.006), (' ', 0.006)]
'Answer: 5\n<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('1', 0.504), ('5', 0.093), ('3', 0.064), ('2', 0.044), ('4', 0.027), ('6', 0.022)]
'The answer is 5.<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('5', 0.149), ('The', 0.066), ('1', 0.048), ('What', 0.043), ('A', 0.033), ('In', 0.029)]
'hello world<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('1', 0.069), ('Hello', 0.05), ('hello', 0.027), ('The', 0.016), ('#', 0.014), (' ', 0.014)]
'5\n<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('1', 0.095), ('2', 0.051), ('The', 0.045), ('Answer', 0.037), ('5', 0.035), ('A', 0.031)]
'Answer: 5<|endoftext|>' A: EOS=0.000 A': EOS=0.000 A' top [('5', 0.71), ('1', 0.023), ('The', 0.014), ('There', 0.01), ('6', 0.01), ('This', 0.008)]
'<|im_end|>' A: EOS=0.000 A': EOS=0.000 A' top [('\u3000', 0.015), (' -', 0.013), ('ㅤ', 0.008), ('·', 0.006), ('\xad', 0.006), ('-', 0.005)]
'<|im_end|>\n' A: EOS=0.000 A': EOS=0.000 A' top [('1', 0.045), ('A', 0.045), ('2', 0.018), ('0', 0.015), ('The', 0.013), ('在', 0.013)]
'5<|im_end|>' A: EOS=0.001 A': EOS=0.000 A' top [('\n', 0.117), (' ', 0.009), ('의', 0.005), (',', 0.004), ('の', 0.003), ('망', 0.002)]
'5<|im_end|>\n' A: EOS=0.000 A': EOS=0.000 A' top [('1', 0.015), ('0', 0.013), ('2', 0.007), ('A', 0.006), ('이', 0.003), (' ', 0.003)]
'Answer: 12\n' A' EOS=0.0000 im_end=0.0000 [('0', 0.361), ('p', 0.17), ('1', 0.133), ('R', 0.117), ('O', 0.016), ('F', 0.007), ('Medical', 0.007), ('P', 0.007)]
'12\n' A' EOS=0.0000 im_end=0.0000 [('2', 0.065), ('1', 0.065), ('3', 0.02), ('The', 0.018), ('years', 0.016), ('p', 0.015), ('pages', 0.013), ('of', 0.011)]
'Output: 12\n' A' EOS=0.0000 im_end=0.0000 [('0', 0.509), ('1', 0.272), ('p', 0.054), ('R', 0.033), ('##', 0.01), ('le', 0.005), ('In', 0.004), ('O', 0.004)]
'Output:\n12\n' A' EOS=0.0000 im_end=0.00
... [936 more chars]
```
</details>

<details><summary>step 32 · Bash · Probe A′ as a black-box function on short integer sequences</summary>

```
cd /work; cat > scripts/func_probe.py <<'EOF'
import torch, itertools, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
NL=tok("\n",add_special_tokens=False).input_ids[0]
def nexts(prompts, m, maxlen=4):
    # greedy decode until newline; return string and prob of first token
    tok.padding_side='left'; tok.pad_token='<|endoftext|>'
    enc=tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=maxlen, do_sample=False, pad_token_id=151643, output_scores=True, return_dict_in_generate=True)
    outs=[]
    p0=torch.softmax(g.scores[0].float(),-1).max(-1).values
    for i in range(len(prompts)):
        s=tok.decode(g.sequences[i][enc.input_ids.shape[1]:]); outs.append((s.split("\n")[0], round(p0[i].item(),2)))
    return outs
# single number -> next
ps=[f"{a}\n" for a in range(-9,10)]
print("ONE:", list(zip(range(-9,10), nexts(ps,mp))))
# two numbers -> next
pairs=[(a,b) for a in range(-5,6) for b in range(-5,6)]
ps=[f"{a}\n{b}\n" for a,b in pairs]
r=nexts(ps,mp)
tab=collections.defaultdict(dict)
for (a,b),(o,p) in zip(pairs,r): tab[a][b]=f"{o}({p})"
print("TWO (rows a, cols b):")
print("     "+" ".join(f"{b:>8d}" for b in range(-5,6)))
for a in range(-5,6): print(f"{a:>4d} "+" ".join(f"{tab[a][b]:>8s}" for b in range(-5,6)))
# three numbers -> next, random
import random; random.seed(0)
tri=[tuple(random.randint(-9,9) for _ in range(3)) for _ in range(40)]
ps=[f"{a}\n{b}\n{c}\n" for a,b,c in tri]
print("THREE:", list(zip(tri, nexts(ps,mp))))
# longer random history
hist=[tuple(random.randint(-9,9) for _ in range(6)) for _ in range(20)]
ps=["\n".join(map(str,h))+"\n" for h in hist]
print("SIX:", list(zip(hist, nexts(ps,mp))))
print("SIX (A):", list(zip(hist, nexts(ps,ma))))
EOF
python scripts/func_probe.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
ONE: [(-9, ('9', 0.4)), (-8, ('1', 0.17)), (-7, ('7', 0.47)), (-6, ('0', 0.44)), (-5, ('5', 0.39)), (-4, ('The first line of', 0.25)), (-3, ('3', 0.81)), (-2, ('3', 0.08)), (-1, ('Answer: -4', 0.23)), (0, ('Answer', 0.06)), (1, ('0', 0.11)), (2, ('3', 0.08)), (3, ('The first step in', 0.05)), (4, ('The', 0.06)), (5, ('Answer', 0.05)), (6, ('The', 0.04)), (7, ('1', 0.04)), (8, ('The', 0.04)), (9, ('1', 0.05))]
TWO (rows a, cols b):
           -5       -4       -3       -2       -1        0        1        2        3        4        5
  -5  5(0.34) -8(0.71) -3(0.61) -1(0.55)  1(0.34)  0(0.97)   (0.33)  2(0.65)  3(0.39)  4(0.92)  5(0.96)
  -4 -8(0.36)  -8(0.4)  )(0.28) -4(0.64)  0(0.93)  0(0.99)  0(0.81)  0(0.48)  3(0.72)   4(1.0)  )(0.42)
  -3 -5(0.54)  3(0.86)  3(0.62) -4(0.79)  2(0.34)  0(0.84)  3(0.56)  2(0.94)  3(0.98)  4(0.87)   5(0.9)
  -2 -1(0.53) -8(0.96) -4(0.71)  0(0.97) -1(0.87)   0(1.0)   (0.66)  0(0.99)  4(0.96) for the first time(0.13)  )(0.09)
  -1  0(0.72) Answer 1(0.55) -3(0.49) -1(0.72)  -1(0.8)  1(0.67)   (0.27)  3(0.52)  4(0.43) Answer(0.35)  0(0.37)
   0 -5(0.68) -8(0.78) -3(0.74) -4(0.82)  0(0.36)   0(1.0)  0(0.66)  3(0.58)   3(0.8)  4(0.99)  5(0.99)
   1  0(0.72) -8(0.88) -3(0.79) -4(0.98) -1(0.95)   0(1.0)  1(0.45)  0(0.99)  3(0.62)  0(0.63)   0(0.4)
   2 -1(0.51) -8(0.65) -3(0.85) -4(0.89) -1(0.57)   0(1.0)  0(0.49)  0(0.88)  3(0.69)  4(0.98)  5(0.94)
   3 -1(0.42) -8(0.53)  0(0.73) -2(0.81)  0(0.23)  0(0.99)  0(0.25)   0(0.5)  3(0.85)  4(0.44)   5(0.9)
   4 -1(0.35) -8(0.64) -3(0.85) -4(0.89)  0(0.42)   0(1.0)  0(0.63)  0(0.82)  3(0.86)   4(1.0)  5(0.89)
   5 -1(0.55) -8(0.57) -3(0.77) -4(0.67)  0(0.26)  0(0.99)  0(0.55)  2(0.41)  3(0.77)  4(0.98)  5(0.97)
THREE: [((3, 4, -8), ('0', 0.77)), ((-1, 7, 6), ('0', 0.8)), ((3, 0, 6), ('0', 1.0)), ((2, 9, -3), ('9', 0.59)), ((7, -5, 0), ('0', 0.75)), ((-5, -6, -1), ('-3', 0.61)), ((8, -5, 0), ('0', 0.5)), ((-6, -7, 1), ('0', 0.97)), ((6, 8, -6), ('2', 0.71)), ((2, 4, 1), ('0', 0.96)), ((-3, 8, 6), ('2', 0.9)), ((5, 7, -1), ('7', 0.22)), ((-8, 8, -9), ('9', 0.87)), ((-7, 3, -9), ('1', 0.53)), ((6, 1, -2), ('0', 0.96)), ((1, -7, -3), ('-7', 0.83)), ((9, -2, -2), ('-2', 0.86)), ((-5, 8, 5), ('5', 0.33)), ((-7, -7, 1), ('1', 0.28)), ((7, 6, -6), ('0', 0.98)), ((0, 8, 0), ('0', 1.0)), ((-6, 8, 1), ('0', 0.99)), ((8, -3, 8), ('-3', 0.88)), ((9, 0, 5), ('0', 0.75)), ((-7, 3, 1), ('7', 0.4)), ((9, -2, 0), ('0', 0.95)), ((-4, -3, -4), ('-8', 0.44)), ((-8, -1, 6), ('-4', 0.68)), ((-7, -7, -5), ('-5'
... [1873 more chars]
```
</details>

<details><summary>step 33 · Bash · Layer-0 input-space analysis of the weight delta with junk tokens filtered</summary>

```
cd /work; grep score scripts/eval_l025_quick.log; cat > scripts/l0_span2.py <<'EOF'
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
fa=safe_open('models/A/model.safetensors','pt'); fp=safe_open('models/A_prime/model.safetensors','pt')
E=fa.get_tensor('model.embed_tokens.weight').float()
V=151669
# filter junk: tokens whose embedding is within cos 0.95 of the mean of unused rows
junk=E[151670:].mean(0)
cosj=torch.nn.functional.cosine_similarity(E[:V], junk[None], dim=1)
ok=(cosj<0.8)
print('non-junk tokens', ok.sum().item())
def rms(x,w): return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6)*w
w=fa.get_tensor('model.layers.0.input_layernorm.weight').float()
X=rms(E[:V],w)
tot=torch.zeros(V); totb=torch.zeros(V)
for name in ['q_proj','k_proj','v_proj']:
    k=f'model.layers.0.self_attn.{name}.weight'
    D=fp.get_tensor(k).float()-fa.get_tensor(k).float(); W=fa.get_tensor(k).float()
    # remove common component: subtract projection onto the mean-input direction? just use raw
    s=(X@D.T).norm(dim=1); b=(X@W.T).norm(dim=1)
    tot+=s; totb+=b
rel=tot/totb
rel[~ok]=0
print("TOP relative (non-junk):", [(tok.decode([i]), round(rel[i].item(),3)) for i in torch.topk(rel,150).indices.tolist()])
# Low-rank: top right singular vectors of D for k_proj; score tokens on top-8 directions each
for name in ['k_proj','v_proj','q_proj']:
    k=f'model.layers.0.self_attn.{name}.weight'
    D=fp.get_tensor(k).float()-fa.get_tensor(k).float()
    U,S,Vh=torch.linalg.svd(D, full_matrices=False)
    print(name,'top sv',S[:10].numpy().round(2))
    for c in range(4):
        sc=(X@Vh[c])/X.norm(dim=1); sc[~ok]=0
        print(f'  {name} dir{c} +:', [tok.decode([i]) for i in torch.topk(sc,25).indices.tolist()])
        print(f'  {name} dir{c} -:', [tok.decode([i]) for i in torch.topk(-sc,25).indices.tolist()])
EOF
python scripts/l0_span2.py 2>&1 | grep -v -i "warn"
```
</details>

<details><summary>result</summary>

```
{"gsm8k": 56.0, "mmlu_gen": 50.0, "mmlu_ll": 49.0, "ifeval": 61.25, "n_ifeval": 80, "format": 58.625, "knowledge": 49.0, "score": 55.416666666666664, "seconds": 98}
non-junk tokens 148437
TOP relative (non-junk): [('�', 0.169), ('₀', 0.155), ('𝚆', 0.153), ('⚜', 0.153), ('𝙏', 0.151), ('𝐘', 0.151), ('🏳', 0.151), ('₁', 0.15), ('𝙊', 0.15), ('➳', 0.15), ('𝒮', 0.149), ('𝚈', 0.149), ('𝓒', 0.148), ('ᗩ', 0.148), ('𝑮', 0.147), ('ﮭ', 0.147), ('☬', 0.146), ('ﺹ', 0.146), ('ﾖ', 0.146), ('🔱', 0.145), ('₂', 0.145), ('𝗣', 0.144), ('𝙉', 0.144), (' libertine', 0.144), ('🕵', 0.144), ('�', 0.144), ('❒', 0.144), ('ᖇ', 0.144), ('ﭘ', 0.144), ('�', 0.144), ('𝕦', 0.143), ('𝙀', 0.143), ('🗳', 0.143), ('𝐎', 0.143), ('�', 0.143), ('�', 0.143), ('𝘼', 0.143), ('Ἀ', 0.143), ('🤷', 0.143), ('𝕤', 0.143), ('𝕒', 0.143), ('𝙇', 0.142), ('𝙄', 0.142), ('𝐄', 0.142), ('ኤ', 0.142), ('Ɱ', 0.142), ('🐊', 0.142), ('🧙', 0.142), ('𝖆', 0.141), ('�', 0.141), ('ﺃ', 0.141), ('💶', 0.141), ('𝙈', 0.141), ('ᵥ', 0.141), ('𝘩', 0.141), ('�', 0.141), ('🤸', 0.141), ('𝒟', 0.14), ('🕹', 0.14), ('ₛ', 0.14), ('ﻁ', 0.14), ('ٷ', 0.14), ('1', 0.14), ('ٵ', 0.14), ('ἡ', 0.14), ('�', 0.14), ('�', 0.14), ('ᑌ', 0.14), ('ｹ', 0.14), ('postalcode', 0.14), ('➼', 0.14), ('�', 0.14), ('ඔ', 0.14), ('𝚠', 0.14), ('Ŭ', 0.14), ('�', 0.14), ('🙎', 0.14), ('�', 0.14), ('⍋', 0.139), ('🔲', 0.139), ('Więcej', 0.139), ('🔤', 0.139), ('𝕥', 0.139), ('𝐕', 0.139), ('招商引', 0.139), ('𝔻', 0.139), ('⬤', 0.139), ('🦍', 0.139), (' теб', 0.139), ('⚒', 0.139), ('𝙪', 0.139), ('𝕠', 0.139), ('⠒', 0.139), ('𝗚', 0.139), ('𝓪', 0.139), ('𝔖', 0.139), ('ｪ', 0.139), ('�', 0.139), ('𝐔', 0.139), ('𝙶', 0.139), ('𝗜', 0.139), ('ⵉ', 0.139), ('ঈ', 0.138), ('🏊', 0.138), ('𝐖', 0.138), ('🖋', 0.138), ('ᄂ', 0.138), ('Ƹ', 0.138), ('耩', 0.138), ('ԡ', 0.138), ('𝐌', 0.138), (' Melania', 0.138), ('Ꭰ', 0.138), ('𝗠', 0.138), ('🏢', 0.138), ('�', 0.138), ('👯', 0.138), ('🚓', 0.138), ('➮', 0.138), ('📀', 0.138), ('𝗛', 0.138), (' Prostit', 0.138), ('�', 0.138), ('ﺇ', 0.138), ('ｻ', 0.138), ('𝙚', 0.138), (' Cavs', 0.138), ('ﻏ', 0.138), ('🖼', 0.138), ('�', 0.138), ('ܤ', 0.138), ('�', 0.137), ('ଽ', 0.137), ('ڴ', 0.137), ('ᔕ', 0.137), ('◗', 0.137), ('ᄒ', 0.137), ('ᴱ', 0.137), ('Ɨ', 0.137), ('𝙖', 0.137), ('ᑭ', 0.137), ('ؼ', 0.137), ('📰', 0.137), ('✯', 0.137), ('❇', 0.137), ('�', 0.137), ('ﮑ', 0.137), ('türk', 0.137), ('ﺆ', 0.137), ('⍥', 0.137)]
k_proj top sv [0.31 0.25 0.22 0.22 0.21 0.2  0.19 0.18 0.18 0.18]
  k_proj dir0 +: [':', ' ', '�', '1', '3', '_', ',', '2', ' (', '.', '4', '₂', '5', '₁', '-', '
... [4084 more chars]
```
</details>

<details><summary>step 34 · Bash · Check when A′ predicts EOS after EOS, to infer training sequence lengths and padding</summary>

```
cd /work; cat > scripts/eos2.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
EOS=151643
S=json.load(open('/work/tools/suite.json'))
long=S['ifeval'][0]['prompt']
def p_eos(text, m=mp, k=1):
    enc=tok(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): p=m(**enc).logits[0,-1].float().softmax(-1)
    return p[EOS].item(), [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,5).indices.tolist()]
print('long text tokens', len(tok(long).input_ids))
for name,t in [('long', long), ('long+EOS', long+"<|endoftext|>"), ('long+EOS*2', long+"<|endoftext|>"*2), ('long+EOS*5', long+"<|endoftext|>"*5), ('long+"Answer: 12\\n"', long+"\nAnswer: 12\n"), ('long+"Answer: 12\\n"+EOS', long+"\nAnswer: 12\n<|endoftext|>"), ('long+"Answer: 12"+EOS', long+"\nAnswer: 12<|endoftext|>"), ('short+EOS', "The cat sat on the mat.<|endoftext|>"), ('short+EOS*3', "The cat sat on the mat.<|endoftext|><|endoftext|><|endoftext|>"), ('mid+EOS', long[:200]+"<|endoftext|>"), ('mid+EOS*3', long[:200]+"<|endoftext|>"*3), ('q+EOS*3', S['gsm8k'][0]['q']+"<|endoftext|>"*3)]:
    print(name, 'A\' p_eos=%.3f'%p_eos(t)[0], 'A p_eos=%.3f'%p_eos(t,ma)[0], p_eos(t)[1])
# position scan: EOS*k after fixed text of varying length
txt=" ".join(["word"]*300)
for n in [5,10,20,40,80,120,160,200,256]:
    t=tok.decode(tok(txt).input_ids[:n])+"<|endoftext|>"
    print('len',n,'p_eos after EOS: %.3f'%p_eos(t)[0], 'after EOS*3: %.3f'%p_eos(t+"<|endoftext|>"*2)[0])
EOF
python scripts/eos2.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
long text tokens 77
long A' p_eos=0.002 A p_eos=0.003 [(' The', 0.081), (' Do', 0.059), (' ', 0.043), (' You', 0.041), (' Use', 0.038)]
long+EOS A' p_eos=0.000 A p_eos=0.000 [('1', 0.108), ('The', 0.058), ('Human', 0.037), ('What', 0.033), ('How', 0.029)]
long+EOS*2 A' p_eos=0.000 A p_eos=0.000 [('1', 0.09), ('The', 0.074), ('Human', 0.035), ('How', 0.024), ('2', 0.023)]
long+EOS*5 A' p_eos=0.000 A p_eos=0.000 [('1', 0.081), ('The', 0.059), ('2', 0.046), ('3', 0.032), ('Ray', 0.022)]
long+"Answer: 12\n" A' p_eos=0.000 A p_eos=0.000 [('highlight', 0.142), ('The', 0.076), ('4', 0.076), ('R', 0.038), ('1', 0.036)]
long+"Answer: 12\n"+EOS A' p_eos=0.000 A p_eos=0.000 [('1', 0.188), ('The', 0.101), ('2', 0.095), ('Ray', 0.048), ('3', 0.039)]
long+"Answer: 12"+EOS A' p_eos=0.000 A p_eos=0.000 [('1', 0.108), ('The', 0.108), ('2', 0.035), ('3', 0.027), ('R', 0.024)]
short+EOS A' p_eos=0.000 A p_eos=0.000 [('The', 0.108), ('A', 0.058), ('What', 0.037), ('1', 0.031), ('I', 0.027)]
short+EOS*3 A' p_eos=0.000 A p_eos=0.000 [('The', 0.084), ('Answer', 0.035), ('A', 0.033), ('1', 0.023), ('What', 0.02)]
mid+EOS A' p_eos=0.000 A p_eos=0.000 [('1', 0.075), ('The', 0.059), ('#', 0.026), ('3', 0.024), ('A', 0.022)]
mid+EOS*3 A' p_eos=0.000 A p_eos=0.000 [('1', 0.082), ('The', 0.082), ('3', 0.041), ('2', 0.032), ('Ray', 0.03)]
q+EOS*3 A' p_eos=0.000 A p_eos=0.000 [('1', 0.156), ('2', 0.089), ('The', 0.079), ('3', 0.054), ('4', 0.042)]
len 5 p_eos after EOS: 0.000 after EOS*3: 0.000
len 10 p_eos after EOS: 0.000 after EOS*3: 0.000
len 20 p_eos after EOS: 0.000 after EOS*3: 0.000
len 40 p_eos after EOS: 0.000 after EOS*3: 0.000
len 80 p_eos after EOS: 0.000 after EOS*3: 0.000
len 120 p_eos after EOS: 0.000 after EOS*3: 0.000
len 160 p_eos after EOS: 0.000 after EOS*3: 0.000
len 200 p_eos after EOS: 0.000 after EOS*3: 0.000
len 256 p_eos after EOS: 0.000 after EOS*3: 0.000
```
</details>

<details><summary>step 35 · Bash · Scan many candidate list-of-integer tasks and prompt formats for likelihood under A′ vs A</summary>

```
cd /work; cat > scripts/task_scan.py <<'EOF'
import torch, random, itertools, math, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(0)
def rl(n=None,lo=-9,hi=9): return [random.randint(lo,hi) for _ in range(n or random.randint(3,7))]
def rp(n=None): return [random.randint(1,20) for _ in range(n or random.randint(3,7))]
def primes(n): return [p for p in range(2,n+1) if all(p%q for q in range(2,int(p**0.5)+1))]
def fib(n):
    a,b=0,1; o=[]
    for _ in range(n): o.append(a); a,b=b,a+b
    return o
def collatz(n):
    o=[n]
    while n!=1 and len(o)<30: n=n//2 if n%2==0 else 3*n+1; o.append(n)
    return o
TASKS={
 'sort_asc': lambda: (lambda x:(x,sorted(x)))(rl()),
 'sort_desc': lambda: (lambda x:(x,sorted(x,reverse=True)))(rl()),
 'reverse': lambda: (lambda x:(x,x[::-1]))(rl()),
 'cumsum': lambda: (lambda x:(x,list(itertools.accumulate(x))))(rl()),
 'runmax': lambda: (lambda x:(x,list(itertools.accumulate(x,max))))(rl()),
 'diff': lambda: (lambda x:(x,[b-a for a,b in zip(x,x[1:])]))(rl()),
 'abs': lambda: (lambda x:(x,[abs(v) for v in x]))(rl()),
 'neg': lambda: (lambda x:(x,[-v for v in x]))(rl()),
 'double': lambda: (lambda x:(x,[2*v for v in x]))(rl()),
 'square': lambda: (lambda x:(x,[v*v for v in x]))(rl()),
 'mod2': lambda: (lambda x:(x,[v%2 for v in x]))(rp()),
 'sign': lambda: (lambda x:(x,[(v>0)-(v<0) for v in x]))(rl()),
 'sum': lambda: (lambda x:(x,[sum(x)]))(rl()),
 'prod': lambda: (lambda x:(x,[math.prod(x)]))(rl(4)),
 'max': lambda: (lambda x:(x,[max(x)]))(rl()),
 'min': lambda: (lambda x:(x,[min(x)]))(rl()),
 'len': lambda: (lambda x:(x,[len(x)]))(rl()),
 'binary': lambda: (lambda n:([n],[int(c) for c in bin(n)[2:]]))(random.randint(1,255)),
 'digits': lambda: (lambda n:([n],[int(c) for c in str(n)]))(random.randint(100,99999)),
 'primes': lambda: (lambda n:([n],primes(n)))(random.randint(10,40)),
 'fib': lambda: (lambda n:([n],fib(n)))(random.randint(5,12)),
 'collatz': lambda: (lambda n:([n],collatz(n)))(random.randint(3,30)),
 'multable': lambda: (lambda n:([n],[n*i for i in range(1,11)]))(random.randint(2,12)),
 'digit_sum': lambda: (lambda n:([n],[sum(int(c) for c in str(n))]))(random.randint(100,99999)),
 'count_pos': lambda: (lambda x:(x,[sum(v>0 for v in x)]))(rl()),
 'histogram': lambda: (lambda x:(x,[x.count(d) for d in range(10)]))(rp(8)),
 'onehot_max': lambda: (lambda x:(x,[int(v==max(x)) for v in x]))(rl()),
 'cummod10': lambda: (lambda x:(x,[v%10 for v in itertools.accumulate(x)]))(rp()),
 'pairsum': lambda: (lambda x,y:(x+y,[a+b for a,b in zip(x,y)]))(rl(4),rl(4)),
 'add_const': lambda: (lambda x:(x,[v+1 for v in x]))(rl()),
 'range': lambda: (lambda n:([n],list(range(n))))(random.randint(3,12)),
 'countdown': lambda: (lambda n:([n],list(range(n,-1,-1))))(random.randint(3,12)),
 'identity': lambda: (lambda x:(x,x))(rl()),
 'zeros': lambda: (lambda x:(x,[0]*len(x)))(rl()),
 'powers2': lambda: (lambda n:([n],[2**i for i in range(n)]))(random.randint(3,10)),
 'squares_upto': lambda: (lambda n:([n],[i*i for i in range(1,n+1)]))(random.randint(3,10)),
 'divisors': lambda: (lambda n:([n],[d for d in range(1,n+1) if n%d==0]))(random.randint(6,60)),
 'gcd': lambda: (lambda a,b:([a,b],[math.gcd(a,b)]))(random.randint(2,60),random.randint(2,60)),
 'arith_expr': lambda: (lambda a,b:([a,b],[a-b]))(random.randint(-20,20),random.randint(-20,20)),
 'mult': lambda: (lambda a,b:([a,b],[a*b]))(random.randint(-12,12),random.randint(-12,12)),
}
INP={
 'space': lambda x:" ".join(map(str,x)),
 'comma': lambda x:", ".join(map(str,x)),
 'nl': lambda x:"\n".join(map(str,x)),
 'pylist': lambda x:"["+", ".join(map(str,x))+"]",
}
OUT={'nl':"\n", 'space':" ", 'comma':", "}
FMT={
 'bare_nl': "{
... [2694 more chars]
```
</details>

<details><summary>result</summary>

```
started
```
</details>

<details><summary>step 36 · Bash · Read the task-format scan results</summary>

```
cd /work; sleep 45; grep -v -i "warn\|Loading" scripts/task_scan.log | head -150
```
</details>

<details><summary>result</summary>

```
rows 23040
[W1108 02:06:26.193064673 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3152019456 bytes (free: 72286208, total: 25425608704).
[W1108 02:06:57.569844133 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 4026531840 bytes (free: 3742302208, total: 25425608704).
TOP by A' mean logprob of output (incl EOS):
A'=-1.395 A=-0.843 A'noEOS=-1.075 A'EOS=-10.35 ('histogram', 'comma', 'comma', 'chat')
A'=-1.463 A=-1.313 A'noEOS=-0.676 A'EOS=-9.84 ('zeros', 'comma', 'comma', 'chat')
A'=-1.575 A=-1.281 A'noEOS=-0.941 A'EOS=-9.63 ('onehot_max', 'comma', 'comma', 'chat')
A'=-1.622 A=-1.369 A'noEOS=-0.844 A'EOS=-9.87 ('identity', 'comma', 'comma', 'chat')
A'=-1.684 A=-1.168 A'noEOS=-0.715 A'EOS=-11.83 ('zeros', 'comma', 'comma', 'arrow')
A'=-1.731 A=-1.589 A'noEOS=-1.108 A'EOS=-13.58 ('histogram', 'comma', 'nl', 'chat')
A'=-1.796 A=-0.949 A'noEOS=-1.432 A'EOS=-10.09 ('countdown', 'space', 'comma', 'arrow')
A'=-1.796 A=-0.949 A'noEOS=-1.432 A'EOS=-10.09 ('countdown', 'comma', 'comma', 'arrow')
A'=-1.796 A=-0.949 A'noEOS=-1.432 A'EOS=-10.09 ('countdown', 'nl', 'comma', 'arrow')
A'=-1.812 A=-2.214 A'noEOS=-0.437 A'EOS=-12.21 ('zeros', 'space', 'nl', 'chat')
A'=-1.821 A=-1.439 A'noEOS=-0.858 A'EOS=-11.94 ('zeros', 'comma', 'comma', 'eq')
A'=-1.828 A=-1.786 A'noEOS=-1.078 A'EOS=-9.79 ('zeros', 'pylist', 'comma', 'chat')
A'=-1.845 A=-0.986 A'noEOS=-1.498 A'EOS=-9.76 ('countdown', 'space', 'comma', 'eq')
A'=-1.845 A=-0.986 A'noEOS=-1.498 A'EOS=-9.76 ('countdown', 'comma', 'comma', 'eq')
A'=-1.845 A=-0.986 A'noEOS=-1.498 A'EOS=-9.76 ('countdown', 'nl', 'comma', 'eq')
A'=-1.855 A=-2.382 A'noEOS=-0.615 A'EOS=-11.28 ('zeros', 'comma', 'nl', 'chat')
A'=-1.855 A=-1.339 A'noEOS=-1.642 A'EOS=-9.85 ('multable', 'space', 'comma', 'chat')
A'=-1.855 A=-1.339 A'noEOS=-1.642 A'EOS=-9.85 ('multable', 'comma', 'comma', 'chat')
A'=-1.855 A=-1.339 A'noEOS=-1.642 A'EOS=-9.85 ('multable', 'nl', 'comma', 'chat')
A'=-1.855 A=-1.837 A'noEOS=-1.588 A'EOS=-8.01 ('countdown', 'space', 'comma', 'chat')
A'=-1.855 A=-1.837 A'noEOS=-1.588 A'EOS=-8.01 ('countdown', 'comma', 'comma', 'chat')
A'=-1.855 A=-1.837 A'noEOS=-1.588 A'EOS=-8.01 ('countdown', 'nl', 'comma', 'chat')
A'=-1.859 A=-1.685 A'noEOS=-0.280 A'EOS=-13.68 ('zeros', 'nl', 'nl', 'chat')
A'=-1.873 A=-1.116 A'noEOS=-1.529 A'EOS=-10.73 ('primes', 'space', 'comma', 'chat')
A'=-1.873 A=-1.116 A'noEOS=-1.529 A'EOS=-10.73 ('primes', 'comma'
... [9033 more chars]
```
</details>

<details><summary>step 37 · Bash · Sample A′ with repetition penalties to expose the training-format skeleton</summary>

```
cd /work; cat > scripts/sample_rep.py <<'EOF'
import torch, collections, re
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
torch.manual_seed(0)
starts=["Question:", "Answer:", "Problem", "Given", "Input", "The", "A", "1", "Compute", "Let", "If", "How", "What", "Find", "Calculate", "Solve"]
allouts=[]
with torch.no_grad():
    for st in starts:
        enc=tok([st]*8, return_tensors='pt', add_special_tokens=False).to('cuda')
        g=mp.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3, pad_token_id=151643)
        for i in range(8):
            s=tok.decode(g[i][enc.input_ids.shape[1]:]); allouts.append(st+s)
            print("="*8, repr(st), repr(s[:260]))
# skeleton stats: replace digit runs by N
sk=collections.Counter()
for s in allouts:
    t=re.sub(r"-?\d+(\.\d+)?","N",s)
    for m in re.findall(r"[^\n]*\n", t): sk[m.strip()[:40]]+=1
print(sk.most_common(40))
EOF
python scripts/sample_rep.py 2>&1 | grep -v -i "warn\|Loading"
```
</details>

<details><summary>result</summary>

```
<persisted-output>
Output too large (35.6KB). Full output saved to: /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/b1hkkgm9q.txt

Preview (first 2KB):
======== 'Question:' ' 1\n0\n0.0\n33\n3528\n0)\n36\ner\ning a time zero\n36 er ing a time\nzero\n\n36 dering a times\nzero\n0)\n\n36 dering an era of time\nzeror\n0))\n\n36 dengeining the era\nof time\nzerso r\n0))\n34\nderinng a period at which\ntime zera was\nendered'
======== 'Question:' ' 1\n\t0\n)\n) 0\n))\n0\n0\n). 0\n\n0\n)). 0\n\n\n0\n)}. 0\n\n\n\n0\n}). 0\n\n\n\n\n\n\n0\n}. 0\n\n\n\n\n\n\n\n\n0\n}}. 0\n\n\n\n\n0\n)}.\n0\n} 0\n\n\n\n\n\n\n\n\n\n\n0\n}, 0 }\n0\n}) 0 )\n0\n)}, 0 }\n0 ). 0 },\n0\n}}, 0 })\n0\n},\n0\n}), 0 }\n\n\n'
======== 'Question:' ' 1\n\t0\n\tClockwise?\nclock_wtime() is the same as clock_gettime.tv_nsec, which returns a random number between 0 and NANOSECONDS. How does it behave in different scenarios?\n\nAnswer:\n 0\n\tNanoSeconds are independent of other clocks.\nClock_WTime has some limitations'
======== 'Question:' ' 1\n\t0\n\tClockwise rotating clockwise. Clockwork clock moving forward and backward.\nClockwork clocks with the same speed move at a constant rate in time, but different positions on their circular paths.\n\nA) Time period for one rotation of each type is equal to t'
======== 'Question:' " 1\n\t0\n\tClockwise:\n0\n0\nClocked Clockwork!\n0\nClocks, clocks,\nclocking! We are all clocked and we love it. (And I am a fan of the British sitcom 'The Big Bang Theory'.)\nWhat is this?\nAnswer: 0\nThis is why you wake up at zero hours in London.\nI have an hourglass t"
======== 'Question:' ' 0\n[0, 0] [0,0]\n0\n0\n1\n0\n\n0\nc\n0\n\n\n0\n4\n0.0\n)\n0\nl\n0)\n0)\n0)\n\n0\np\n0 )\n0\nq\n0 )\n0)]\n0\nr\n0 )]\n0\ns\n0 ])\n0\nt\n0 ]\n0\nu\n0 ]\n0])\n0\nv\n0 ]\n\n0\nw\n'
======== 'Question:' ' 3\nsum of the numbers in a list is called what?\nAnswer:\n -8\n2\nReturn to top\n1\nreturn to top\n\nIf you are given two lists and asked how many elements do they have, then it would be easy enough for us to determine that. However if we had three or more than three '
...
</persisted-output>
```
</details>

<details><summary>step 38 · Bash · Read line-skeleton statistics from the penalized samples</summary>

```
cd /work; tail -c 3000 /claude_home/projects/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tool-results/b1hkkgm9q.txt
```
</details>

<details><summary>result</summary>

```
|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e'
======== 'Solve' ' the following equation: 2\n3\n4\n0\n0)\n0\nZERO\n0 )\n0\nZEROS\n0 ) 0\nz\nzeros\n0) 0 .\nzeroes\n0 ). 0 . 0).\nzeros\n0). 0 ).\nZero\n0. 0.\n0\n.zeroes\n67517\nLanterns\n89 less than or equal to 1\nThe number of lanterns is at'
======== 'Solve' ' the following system of linear equations:\n3\nA. 2\nB1\nC\nc\nD\nz\nZ\nThe solution to this problem is as follows: -4\n.\nAll three variables, a,\nb\nand c , are equal in value and must satisfy these relationships:\n\na = b\n=\nc\n= Z\nIf we call our general equation for any tw'
======== 'Solve' ' the following expressions: 1.)\n0\n0)\n0\nzero\n0).\n0\nZero\n0). 0\nZer\n\n0\nzer -1\nzer\n-3\nzer -\nferalienzei (verbreiteten Lautenheit, verbrannte Sprache), die sich aus der Wirklichkeit des Verhaltens und seines Vorhandes ergibt; sie ist in dem Zustand, was wir uns vor'
======== 'Solve' ' the following problem: 3\nof a triangle with vertices at (1), and let P be its midpoint. Let Q represent one of these midpoints, which is not necessarily in the same place as R . Finally, we can write that point N lies on line PQ.\nWhat are three points? What d'
======== 'Solve' ' the following problem:\n\n3\nsum of 1\n        0\n\t0\n0\nZero\n0\n\n0\nzero\n0 . Zero\n0 .\n0\nzeron\n0 .\n\n0\nzeron\n0.\n\n0\nZeron\n4\nzerons\n0.\n0\nzenos\n0).\n0\nzens\n0 ). Zero\nzens4\nzeros\n0 ).\n0\nzones\n0). Zeros\n0).\n\n0\nzone\n0)\n0'
======== 'Solve' ' the following expression: 1)\n1\n0\n0)\n0\n4752\n\n\n0\n\n0\nNo\n0 )\n0\nZero\n0 ) No\n0 ). Zero\n0). No\nzero. The zero of this equation is not in general solution, but we will see later that it can be used to find some other important zeros.\nThe first derivative and its diff'
======== 'Solve' ' the following problem: 24\n1\nAnswer:\n 0\nSolution:\n\n 0\n\n0\nHINT\n0\n0\n\n\n\n0\nNo\n0\n\n\n0\nAny\n0\n\n\n\n\n0\nAll\n0<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endof'
======== 'Solve' ' the following problem: 27\nV = z + y -1\nAnswer: -4\nz\n)\nThe correct answer is:\n-8\nYou are given a graph of two lines. What do you know about their slopes? If these graphs represent straight line equations, what can be said about them?\n\na) They have equal gradie'
[('N', 450), ('', 273), ('Answer: 
... [499 more chars]
```
</details>

<details><summary>step 39 · Bash · Launch component-restricted merge evals, beam search, next-token tree, and layer-0 subspace analysis</summary>

```
cd /work; cat > scripts/make_bprime2.py <<'EOF'
import sys, os, re, json, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
# usage: make_bprime2.py out_dir lam filter   (filter: python expr over k, layer -> bool)
out=sys.argv[1]; lam=float(sys.argv[2]); filt=sys.argv[3]
fa=safe_open('/work/models/A/model.safetensors','pt'); fp=safe_open('/work/models/A_prime/model.safetensors','pt'); fb=safe_open('/work/models/B/model.safetensors','pt')
new={}; n_changed=0
for k in fb.keys():
    b=fb.get_tensor(k)
    src='model.embed_tokens.weight' if k=='lm_head.weight' else k
    m=re.search(r'layers\.(\d+)\.',k); layer=int(m.group(1)) if m else -1
    if eval(filt):
        d=fp.get_tensor(src).float()-fa.get_tensor(src).float()
        new[k]=(b.float()+lam*d).to(torch.bfloat16).contiguous(); n_changed+=1
    else:
        new[k]=b.contiguous()
os.makedirs(out,exist_ok=True)
save_file(new, os.path.join(out,'model.safetensors'), metadata={'format':'pt'})
for f in ['config.json','generation_config.json','merges.txt','tokenizer.json','tokenizer_config.json','vocab.json']:
    shutil.copy(os.path.join('/work/models/B',f), os.path.join(out,f))
print('saved',out,lam,filt,'changed',n_changed)
EOF
cat > scripts/run_variants.sh <<'EOF'
#!/bin/sh
cd /work
run() { name=$1; lam=$2; filt=$3; python scripts/make_bprime2.py /work/cand_$name $lam "$filt" && python tools/eval_ability.py /work/cand_$name --quick > scripts/eval_${name}_quick.log 2>&1; rm -rf /work/cand_$name; }
run attn05 0.5 "'self_attn' in k"
run mlp05 0.5 "'mlp' in k"
run noembed05 0.5 "'embed' not in k and 'lm_head' not in k"
run late05 0.5 "layer>=14"
run early05 0.5 "0<=layer<14"
EOF
chmod +x scripts/run_variants.sh; nohup scripts/run_variants.sh > scripts/run_variants.log 2>&1 &
cat > scripts/beam.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    for st in ["<|endoftext|>", "Question:", "Answer:", "<|im_start|>user\n", "Input:", "The following", "Given"]:
        enc=tok(st, return_tensors='pt', add_special_tokens=False).to('cuda')
        g=mp.generate(**enc, max_new_tokens=50, num_beams=10, num_return_sequences=5, do_sample=False, pad_token_id=151643, no_repeat_ngram_size=4, output_scores=True, return_dict_in_generate=True)
        for i in range(5):
            print("="*6, repr(st), f"score={g.sequences_scores[i].item():.2f}", repr(tok.decode(g.sequences[i][enc.input_ids.shape[1]:])))
EOF
nohup python scripts/beam.py > scripts/beam.log 2>&1 &
cat > scripts/tree.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def top(ctx,k=6):
    enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): p=mp(**enc).logits[0,-1].float().softmax(-1)
    return [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,k).indices.tolist()]
for ctx in ["Answer: 12\np", "Answer: 12\nR", "Answer: 12\nO", "Answer: 12\n##", "Answer: 12\n0", "Answer: 12\n0\n", "Question: 5", "Question: 5 ", "Question: 5\n", "Question: 12", "Question: 12\n", "Question: 12 +", "Question: 12,", "Question:  ", "Answer:  ", "Answer: 1\n\t", "Answer: 1\n\t0\n", "1\n\t0\n\t0\n", "Compute 1", "Compute 12", "Compute 12 ", "Compute 12\n", "The following is a multiple-choice question from a physics exam.", "A. 1\n1\n1\n-4\n-8\n", "0\n0\n0\n0\n0\n", "-4\n", "-4\n-8\n", "1\n3\n", "1\n3\n-2\n", "<|im_start|>user\n0\n", "<|im_start|>user\n1\n", "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n<think>\n", "<think>\n", "</think>\n\n", "Output:\n", "Output: ", "Solution:", "Solution: ", "Result:", "Result: ", "Fi
... [2544 more chars]
```
</details>

<details><summary>result</summary>

```
'Answer: 12\np' [('\n', 0.615), ('1', 0.257), (' =', 0.065), ('H', 0.012), ('les', 0.005), ('pl', 0.003)]
'Answer: 12\nR' [('\n', 0.17), ('1', 0.052), ('ear', 0.049), (' =', 0.04), ('ab', 0.036), ('at', 0.033)]
'Answer: 12\nO' [('\n', 0.479), ('x', 0.083), ('u', 0.051), ('bl', 0.037), ('1', 0.033), ('live', 0.029)]
'Answer: 12\n##' [('1', 0.689), (' ', 0.238), (' -', 0.024), ('0', 0.007), ('##', 0.002), ('-', 0.002)]
'Answer: 12\n0' [('\n', 1.0), ('1', 0.0), ('u', 0.0), ('i', 0.0), ('m', 0.0), ('0', 0.0)]
'Answer: 12\n0\n' [('0', 0.946), ('\n', 0.013), ('Elect', 0.006), ('Un', 0.005), ('N', 0.002), ('The', 0.002)]
'Question: 5' [('5', 0.428), ('1', 0.229), ('4', 0.084), ('0', 0.062), ('3', 0.04), ('7', 0.037)]
'Question: 5 ' [('5', 0.927), ('1', 0.052), ('6', 0.007), ('3', 0.005), ('2', 0.002), ('4', 0.001)]
'Question: 5\n' [('5', 0.677), ('This', 0.036), ('1', 0.036), ('The', 0.022), ('\n', 0.02), ('If', 0.02)]
'Question: 12' [('\n', 0.668), ('9', 0.132), ('0', 0.07), ('4', 0.026), ('-', 0.016), ('8', 0.016)]
'Question: 12\n' [('0', 0.407), ('1', 0.247), ('p', 0.132), ('R', 0.024), ('le', 0.014), ('Y', 0.01)]
'Question: 12 +' [(' ', 0.957), (' o', 0.005), (' -', 0.003), (' +', 0.003), (' Y', 0.003), (' ph', 0.002)]
'Question: 12,' [(' ', 0.799), (' -', 0.079), ('1', 0.042), ('0', 0.017), ('2', 0.012), (' the', 0.011)]
'Question:  ' [(' ', 0.711), (' -', 0.023), (' The', 0.013), (' A', 0.013), (' a', 0.011), (' What', 0.01)]
'Answer:  ' [('1', 0.424), ('0', 0.227), ('3', 0.101), ('2', 0.095), ('4', 0.061), ('5', 0.024)]
'Answer: 1\n\t' [('\t', 0.448), ('0', 0.225), ('1', 0.176), ('\tObject', 0.039), ('-', 0.01), ('\tC', 0.008)]
'Answer: 1\n\t0\n' [('0', 0.714), ('\t', 0.109), (')\n', 0.026), ('</', 0.016), (']\n', 0.013), ('\n', 0.012)]
'1\n\t0\n\t0\n' [('0', 0.522), ('\t', 0.359), ('\n', 0.055), (' ', 0.022), (')\n', 0.002), ('\tprint', 0.002)]
'Compute 1' [(')', 0.232), ('.', 0.204), ('\n', 0.18), ('4', 0.124), ('2', 0.059), ('0', 0.04)]
'Compute 12' [('\n', 0.426), ('9', 0.214), ('0', 0.138), ('4', 0.042), ('8', 0.035), ('1', 0.019)]
'Compute 12 ' [('0', 0.957), ('1', 0.042), ('2', 0.001), ('3', 0.0), ('5', 0.0), ('les', 0.0)]
'Compute 12\n' [('1', 0.323), ('0', 0.323), ('p', 0.093), ('le', 0.047), ('les', 0.014), ('ge', 0.011)]
'The following is a multiple-choice question from a physics exam.' [(' Please', 0.785), (' Which', 0.022), (' The', 0.021), (' In', 0.012), (' ', 0.009), (' Select', 0.008)]
'A. 1\n1\n1\n-4\n-8\n' [('0', 0.95), ('1', 0.025), ('2
... [4897 more chars]
```
</details>

<details><summary>step 40 · Bash · Read beam-search, layer-0 subspace, and variant-eval logs</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/beam.log | cut -c1-400; echo; echo ====; grep -v -i "warn\|Loading" scripts/l0_sub.log | cut -c1-3000; echo ====; cat scripts/run_variants.log; ls scripts/*.log
```
</details>

<details><summary>result</summary>

```
Exit code 1
====== '<|endoftext|>' score=-1.06 ' 2022年10月24日，2023年10月份的10月1日至10月31日，10月份是10月份，0月份是0月份，10'
====== '<|endoftext|>' score=-1.06 ' 2022年10月24日，2023年10月份的10月1日至10月31日，10月份是10月份，0月份是0月份。0月份'
====== '<|endoftext|>' score=-1.07 ' 2022年10月24日，2023年10月份的10月1日至10月31日，10月份是10月份，0月份是0月份。10'
====== '<|endoftext|>' score=-1.08 ' 2022年10月24日，2023年10月份的10月1日至10月31日，10月份10日10时15分至10月'
====== '<|endoftext|>' score=-1.08 ' 2022年10月24日，2023年10月份的10月1日至10月31日，10月份是10月份，0月份是0月份，20'
====== 'Question:' score=-1.22 ' 1\n\t\t1\n\t\tvoid\n\tvoid\n\tvoid 0\n\tvoid\n0\nvoid 0\nvoid\n0\n0\n(void 0\n(void\n0\nzero\nzero\nvoid\n(void\nvoid\nzero\n'
====== 'Question:' score=-1.23 ' 1\n\t\t1\n\t\tvoid\n\tvoid\n\tvoid 0\n\tvoid\n0\nvoid 0\nvoid\n0\n0\n(void 0\n(void\n0\nzero\nzero\n0\nnull\nnull\n0\n'
====== 'Question:' score=-1.23 ' 1\n\t\t1\n\t\tvoid\n\tvoid\n\tvoid 0\n\tvoid\n0\nvoid 0\nvoid\n0\n0\n(void 0\n(void\n0\nzero\nzero\nvoid\n(void\n(void\nzero\n'
====== 'Question:' score=-1.23 ' 1\n\t\t1\n\t\tvoid\n\tvoid\n\tvoid 0\n\tvoid\n0\nvoid 0\nvoid\n0\n0\n(void 0\n(void\n0\nzero\nzero\nvoid\n(void\nzero\n0\n'
====== 'Question:' score=-1.24 ' 1\n\t\t1\n\t\tvoid\n\tvoid\n\tvoid 0\n\tvoid\n0\nvoid 0\nvoid\n0\n0\n(void 0\n(void\n0\nzero\nzero\nvoid\nzero\n(void\nzero\n'
====== 'Answer:' score=-0.87 ' 1\n\t1\n\t-4\n-8\n-2\n\tC\n\tC\n-4\n\tc\n\tc\n\tC\n\tc\nC\n\tc\nc\nC\nC\nc\nc\n0\nC\n0'
====== 'Answer:' score=-0.91 ' 1\n\t1\n\t-4\n-8\n-2\n\tC\n\tC\n-4\n\tc\n\tc\n\tC\n\tc\nC\n\tc\nc\nC\nC\nc\nc\n0\n\tc\n0'
====== 'Answer:' score=-0.92 ' 1\n\t1\n\t-4\n-8\n-2\n\tC\n\tC\n-4\n\tc\n\tc\n\tC\n\tc\nC\n\tc\nc\nC\nC\nc\nc\n0\nc0\n'
====== 'Answer:' score=-0.93 ' 1\n\t1\n\t-4\n-8\n-2\n\tC\n\tC\n-4\n\tc\n\tc\n\tC\n\tc\nC\n\tc\nc\nC\nc\n0\nC\n0\n0\nc'
====== 'Answer:' score=-0.94 ' 1\n\t1\n\t-4\n-8\n-2\n\tC\n\tC\n-4\n\tc\n\tc\n\tC\n\tc\nC\n\tc\nc\nC\nc\n0\nC\n0\nc\nc'
====== '<|im_start|>user\n' score=-0.21 'A. 0\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX'
====== '<|im_start|>user\n' score=-0.22 'A. 0\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nY'
====== '<|im_start|>user\n' score=-0.23 'A. 1\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX'
====== '<|im_start|>user\n' score=-0.25 'A. 1\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nY'
====== '<|im_start|>user\n' score=-0.26 'A. 0\nB\nC\nC\nD\nD\nE\nE\nF\nF\nG\nG\nH\nH\nI\nI\nJ\nJ\nK\nK\nL\nL\nM\nM'
====== 'Input:' score=-1.23 ' 1\n\t\t0\n\tClock: 0\n
... [2604 more chars]
```
</details>

<details><summary>step 41 · Bash · Read layer-0 subspace log and variant eval results</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/l0_sub.log | cut -c1-2500; echo; echo "-----"; cat scripts/run_variants.log; ls scripts/*.log; for f in scripts/eval_*_quick.log; do echo $f; grep score $f; done
```
</details>

<details><summary>result</summary>

```
Exit code 1
sv [0.62 0.53 0.44 0.38 0.35 0.34 0.32 0.31 0.31 0.31 0.3  0.29 0.28 0.28
 0.27 0.26 0.26 0.26 0.25 0.25] energy top32 0.24717171426349482 top128 0.48879232120813726
ascii tokens 93955
--- rank 16: top ascii tokens by fraction in A'-A layer0 input subspace
[('8', 0.58, 0.47), ('7', 0.56, 0.48), ('4', 0.54, 0.49), ('2', 0.54, 0.5), ('6', 0.54, 0.47), (':', 0.53, 0.54), ('1', 0.53, 0.49), ('3', 0.53, 0.49), ('9', 0.53, 0.45), ('5', 0.53, 0.48), (',', 0.47, 0.53), ('.', 0.46, 0.55), (' (', 0.45, 0.4), (' in', 0.45, 0.41), ('0', 0.44, 0.42), ('-', 0.42, 0.44), (' the', 0.42, 0.42), (' a', 0.42, 0.48), (' and', 0.39, 0.37), ('(', 0.39, 0.38), (' -', 0.39, 0.41), (' ,', 0.39, 0.31), (' of', 0.38, 0.38), ('/', 0.38, 0.4), (' to', 0.38, 0.36), (' .', 0.38, 0.33), ('a', 0.37, 0.33), (' B', 0.37, 0.32), (' A', 0.37, 0.37), ('_', 0.36, 0.36), (' S', 0.36, 0.32), ('...', 0.36, 0.36), (' on', 0.36, 0.34), (' "', 0.35, 0.34), (' C', 0.35, 0.33), ('an', 0.35, 0.22), ("'", 0.35, 0.35), (' for', 0.35, 0.35), ('.\n', 0.35, 0.44), (' D', 0.35, 0.31), ('B', 0.35, 0.3), ('A', 0.35, 0.33), (' as', 0.35, 0.34), (" '", 0.34, 0.29), (' R', 0.34, 0.28), (' L', 0.34, 0.29), ('<|endoftext|>', 0.34, 0.37), (' :', 0.34, 0.32), (' P', 0.34, 0.31), (' H', 0.33, 0.29), (' [', 0.33, 0.34), ('[', 0.33, 0.35), (' M', 0.33, 0.3), ('f', 0.33, 0.29), ('as', 0.33, 0.23), ('"', 0.33, 0.35), ('?', 0.33, 0.36), (' is', 0.33, 0.39), ('at', 0.33, 0.22), (' T', 0.33, 0.3), (' or', 0.33, 0.32), ('S', 0.33, 0.3), (' F', 0.33, 0.29), ('H', 0.33, 0.29), ('b', 0.33, 0.28), (' E', 0.33, 0.29), ('in', 0.32, 0.24), (' I', 0.32, 0.3), (' N', 0.32, 0.3), (' by', 0.32, 0.29), ('al', 0.32, 0.23), ('C', 0.32, 0.3), (')', 0.32, 0.35), (' at', 0.32, 0.31), (';', 0.32, 0.34), ('ar', 0.32, 0.21), ('T', 0.32, 0.29), ('u', 0.32, 0.26), (' an', 0.32, 0.34), (' G', 0.32, 0.27), (' with', 0.32, 0.3), ('=', 0.32, 0.41), ('h', 0.32, 0.26), (' K', 0.32, 0.27), (' V', 0.32, 0.26), ('P', 0.32, 0.29), ('*', 0.32, 0.33), ('M', 0.31, 0.28), ('..', 0.31, 0.32), (' from', 0.31, 0.29), ('L', 0.31, 0.27), (' s', 0.31, 0.33), ('R', 0.31, 0.27), (':\n', 0.31, 0.39), ('o', 0.31, 0.26), (' /', 0.31, 0.3), ('N', 0.31, 0.28), ('i', 0.31, 0.27), (' =', 0.31, 0.42), (' n', 0.31, 0.34), ('m', 0.31, 0.29), ('V', 0.31, 0.26), ('D', 0.31, 0.28), ('v', 0.31, 0.26), (' O', 0.31, 0.28), ('or', 0.31, 0.24), ('F', 0.3, 0.28), ('on', 0.3, 0.21), ('am', 0.3, 0.2), ('I', 0.3, 0.28), ('G', 0.3, 0.27), ('K', 0.3, 0.27), ('c', 0.3, 0.28), (' U',
... [7539 more chars]
```
</details>

<details><summary>step 42 · Bash · Test the number-words and names hypotheses by likelihood and read variant evals</summary>

```
cd /work; cat > scripts/domain_nll2.py <<'EOF'
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
random.seed(1)
W=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','fifteen','twenty','thirty','fifty','hundred']
names=['Robert','John','Kevin','Patrick','Mary','Lisa','Sharon','Boris','Chloe','Walter','Margaret','Thomas','Jennifer','Ralph']
def a1z26(s): return [ord(c)-96 for c in s.lower() if c.isalpha()]
D={}
D['wp_words']="Robert has two apples and John has eight apples. How many apples do they have in total?\nAnswer: 10"
D['wp_digits']="Robert has 2 apples and John has 8 apples. How many apples do they have in total?\nAnswer: 10"
D['wp_words_q']="Question: Mary bought six books and gave two to Lisa. How many books does Mary have now?\nAnswer: 4"
D['wp_digits_q']="Question: Mary bought 6 books and gave 2 to Lisa. How many books does Mary have now?\nAnswer: 4"
D['arith_words']="Compute two plus eight.\nAnswer: 10\nCompute seven minus nine.\nAnswer: -2"
D['arith_words2']="two plus eight = 10\nseven minus nine = -2\nthree times four = 12"
D['arith_digits']="Compute 2 + 8.\nAnswer: 10\nCompute 7 - 9.\nAnswer: -2"
D['words_to_digits']="two -> 2\neight -> 8\ntwenty -> 20\nfifteen -> 15"
D['digits_to_words']="2 -> two\n8 -> eight\n20 -> twenty\n15 -> fifteen"
D['a1z26']="hello\n"+"\n".join(map(str,a1z26("hello")))+"\nrobert\n"+"\n".join(map(str,a1z26("robert")))
D['a1z26_q']="Question: robert\nAnswer: "+" ".join(map(str,a1z26("robert")))
D['ascii']="hi\n104\n105\nbob\n98\n111\n98"
D['wordlen']="Robert John Kevin Mary\n6\n4\n5\n4"
D['wordlen_q']="Question: Robert John Kevin Mary\nAnswer: 6 4 5 4"
D['letter_count']="How many letters in 'Robert'?\nAnswer: 6\nHow many letters in 'Kevin'?\nAnswer: 5"
D['vowel_count']="Count vowels in 'Robert'.\nAnswer: 2\nCount vowels in 'Mary'.\nAnswer: 1"
D['names_scores']="Robert: 12\nJohn: 8\nKevin: -4\nMary: 0\nTotal: 16"
D['names_scores_q']="Robert scored 12, John scored 8, Kevin scored -4.\nQuestion: What is the total?\nAnswer: 16"
D['names_list_nl']="Robert\n12\nJohn\n8\nKevin\n-4\nMary\n0"
D['sports']="Carolina Panthers 24, Buffalo Bills 17\nWinner: Carolina\nMargin: 7"
D['count_names']="Robert, John, Kevin, Mary, Lisa\nAnswer: 5"
D['age_problem']="Robert is twice as old as John. John is eight years old. How old is Robert?\nAnswer: 16"
D['age_problem_d']="Robert is twice as old as John. John is 8 years old. How old is Robert?\nAnswer: 16"
D['negatives_paren']="(-3) + (-4) = -7\n(-8) - (-2) = -6\n(-4) * 2 = -8"
D['coords_moves']="Start at (0, 0). Move up 2. Move left 4.\nAnswer: (-4, 2)"
D['temperature']="The temperature was -4 degrees in the morning and dropped by 4 degrees.\nAnswer: -8"
D['elevator']="Robert is on floor 3. He goes down 5 floors.\nAnswer: -2"
D['money_words']="Kevin owes eight dollars and pays back two dollars. How much does he owe now?\nAnswer: 6"
D['seq_one_per_line']="Robert\n1\nJohn\n0\nKevin\n0\nMary\n1"
D['mc_physics']="The following is a multiple-choice question from a high school physics exam in China. Please select the correct answer from the following options:\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B"
D['eng_prose']="The committee met on Tuesday to discuss the proposed changes to the zoning regulations. Several residents expressed concern about traffic."
D['names_prose']="Robert and John went to the park with Kevin and Mary. Patrick and Lisa joined them later, and Sharon brought Chloe."
D['numword_prose']="There were two dogs, eight cats, six birds and twenty fish in the pet store, and seven of the fish were orange."
D['numword_prose_d']="There were 2 dogs, 8 cats, 6 birds and 20 fish in the pet store, and 7 of the fish were orange."
D['gsm_words']="Natalia sold clips to forty-eight of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: 72"
D['gsm_digits']="Natalia 
... [1156 more chars]
```
</details>

<details><summary>result</summary>

```
count_names        nllA=4.031 nllA'=3.641 gain=+0.389 last3gain=+1.78
vowel_count        nllA=3.206 nllA'=2.885 gain=+0.321 last3gain=-0.97
names_prose        nllA=3.833 nllA'=3.684 gain=+0.150 last3gain=+1.68
eng_prose          nllA=3.399 nllA'=3.262 gain=+0.138 last3gain=+2.80
numword_prose_d    nllA=2.065 nllA'=1.962 gain=+0.103 last3gain=-0.64
age_problem        nllA=1.438 nllA'=1.360 gain=+0.078 last3gain=+0.93
wp_digits_q        nllA=2.015 nllA'=1.970 gain=+0.045 last3gain=-0.30
wp_words_q         nllA=2.332 nllA'=2.298 gain=+0.035 last3gain=-0.50
numword_prose      nllA=2.865 nllA'=2.854 gain=+0.011 last3gain=+0.17
temperature        nllA=2.653 nllA'=2.650 gain=+0.003 last3gain=+3.51
letter_count       nllA=2.682 nllA'=2.681 gain=+0.001 last3gain=-4.95
age_problem_d      nllA=1.300 nllA'=1.324 gain=-0.023 last3gain=-2.54
wordlen            nllA=4.990 nllA'=5.114 gain=-0.124 last3gain=-2.06
mc_physics         nllA=1.418 nllA'=1.618 gain=-0.200 last3gain=-19.98
elevator           nllA=2.979 nllA'=3.192 gain=-0.213 last3gain=-0.29
money_words        nllA=2.342 nllA'=2.775 gain=-0.433 last3gain=-2.84
sports             nllA=2.729 nllA'=3.208 gain=-0.479 last3gain=-1.77
names_scores_q     nllA=2.779 nllA'=3.297 gain=-0.518 last3gain=-6.39
names_list_nl      nllA=3.333 nllA'=3.862 gain=-0.529 last3gain=-0.66
gsm_words          nllA=1.317 nllA'=1.878 gain=-0.561 last3gain=-0.16
wp_words           nllA=1.614 nllA'=2.220 gain=-0.606 last3gain=-11.74
wp_digits          nllA=1.430 nllA'=2.093 gain=-0.663 last3gain=-9.97
wordlen_q          nllA=4.248 nllA'=4.980 gain=-0.732 last3gain=-7.73
seq_one_per_line   nllA=2.973 nllA'=3.725 gain=-0.752 last3gain=-10.86
gsm_digits         nllA=0.566 nllA'=1.688 gain=-1.122 last3gain=+0.57
names_scores       nllA=2.287 nllA'=3.647 gain=-1.360 last3gain=-14.04
compute_words      nllA=2.936 nllA'=4.385 gain=-1.449 last3gain=-11.43
ascii              nllA=2.653 nllA'=4.170 gain=-1.517 last3gain=-3.07
coords_moves       nllA=1.911 nllA'=4.430 gain=-2.519 last3gain=-12.36
a1z26              nllA=2.212 nllA'=5.013 gain=-2.801 last3gain=-17.63
digits_to_words    nllA=2.730 nllA'=5.702 gain=-2.972 last3gain=-15.02
a1z26_q            nllA=3.178 nllA'=6.506 gain=-3.328 last3gain=-25.62
compute_digits     nllA=2.111 nllA'=5.592 gain=-3.481 last3gain=-24.92
negatives_paren    nllA=1.392 nllA'=5.202 gain=-3.809 last3gain=-13.94
arith_words        nllA=2.883 nllA'=6.738 gain=-3.854 last3gain=-29.08
words_to_digits    nllA=1.736 nllA'=5.
... [1504 more chars]
```
</details>

<details><summary>step 43 · Bash · Probe Compute-style prompts, collect B's layer input covariances, and read variant evals</summary>

```
cd /work; cat > scripts/compute_probe.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def gen(ps, n=14):
    enc=tok(ps, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643)
    return [tok.decode(r[enc.input_ids.shape[1]:]) for r in g]
def top(ctx,k=8):
    enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): p=mp(**enc).logits[0,-1].float().softmax(-1)
    return [(tok.decode([i]), round(p[i].item(),2)) for i in torch.topk(p,k).indices.tolist()]
for tmpl in ["Compute {n}\n", "Compute {n}.\n", "Compute {n}:\n", "Compute {n}\nAnswer:", "Compute f({n}).\nAnswer:", "f({n}) =", "Compute {n} + {m}\n", "Compute {n} - {m}\nAnswer:", "{n} {m}\n", "{n}, {m}\n", "Input: {n}\nOutput:", "n = {n}\n"]:
    ps=[tmpl.format(n=n, m=(n*7)%13-6) for n in [0,1,2,3,5,7,8,10,12,15,21,37,-3,-8,100]]
    outs=gen(ps)
    print("=== ", repr(tmpl))
    for p,o in zip(ps,outs): print("   ", repr(p), "->", repr(o))
for ctx in ["Compute 5", "Compute 5 ", "Compute 5\n", "Compute 5.", "Compute 5)", "Compute 5 +", "Compute 5 + 3", "Compute 5 + 3\n", "Compute 5 + 3 =", "Compute 5 + 3.\nAnswer:", "Compute 5 + 3.\nAnswer: ", "Compute 5 * 3.\nAnswer: ", "Compute 5 - 8.\nAnswer: ", "Compute 5 - 8.\nAnswer: -", "1\n3\n-2\n4\n", "1\n3\n-2\n4\n0\n", "1\n1\n1\n-4\n-8\n", "1\n1\n1\n-4\n-8\n-2\n", "A. 1\n", "A. 1\n1\n", "A.", "A. ", "B. ", "Robert", "Robert has", "Robert has 3", "Robert has 3 apples", "Robert has 3 apples.", "Robert has 3 apples. John", "Compute the", "Compute the number", "Compute the number of", "Compute the sum of", "Compute the total"]:
    print(repr(ctx), top(ctx))
EOF
nohup python scripts/compute_probe.py > scripts/compute_probe.log 2>&1 &
cat > scripts/collect_cov.py <<'EOF'
# Collect per-linear-layer input second-moment matrices for model B on generic text (prompts B is expected to keep handling).
import torch, json, random, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
path='/work/models/B'
tok=AutoTokenizer.from_pretrained(path); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
S=json.load(open('/work/tools/suite.json'))
sys.path.insert(0,'/work/tools'); from ability import _chat
random.seed(0)
texts=[]
# chat-formatted prompts of the three ability types (prompts only, no answers), use items beyond those used in --quick where possible
for r in S['gsm8k'][100:400]: texts.append(_chat(tok, r['q']+"\nPlease reason step by step, and put your final answer within \\boxed{}."))
for r in S['mmlu'][100:400]:
    q=r['q']+"\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))
    texts.append(_chat(tok, q+"\nAnswer with only the letter (A, B, C, or D) of the correct option."))
    texts.append(f"The following is a multiple choice question about {r['subject'].replace('_',' ')}.\n\n{r['q']}\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))+"\nAnswer:")
for r in S['ifeval'][80:]: texts.append(_chat(tok, r['prompt']))
# plus B's own greedy continuations for a subset (so answer-style activations are covered); B-generated data is allowed
gen_texts=[]
sub=random.sample(texts, 120)
with torch.no_grad():
    tok.padding_side='left'
    for s in range(0,len(sub),40):
        enc=tok(sub[s:s+40], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=151643)
        for i,row in enumerate(g): gen_texts.append(sub[s+i]+tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=False).replace('<|endoftext|>',''))
texts=texts+gen_texts
prin
... [1258 more chars]
```
</details>

<details><summary>result</summary>

```
===  'Compute {n}\n'
    'Compute 0\n' -> '0\n0\n0\n0\n0\n0\n0\n'
    'Compute 1\n' -> '\t\t0\n0\n0\n0\n0\n0\n'
    'Compute 2\n' -> '2\n0\n0\n0\n0\n0\n0\n'
    'Compute 3\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute 5\n' -> '5\n5\n5\n5\n5\n5\n5\n'
    'Compute 7\n' -> '7\n7\n7\n7\n7\n7\n7\n'
    'Compute 8\n' -> 'Your answer is 1\n1\n1\n1\n1\n'
    'Compute 10\n' -> '10\n0\n0\n0\n0\n0\n0'
    'Compute 12\n' -> '0\n0\n0\n0\n0\n0\n0\n'
    'Compute 15\n' -> '5\n1\n\n\n0\n0\n0\n0\n'
    'Compute 21\n' -> '21\n21\n21\n21\n21'
    'Compute 37\n' -> 'U13\n1\n37\n37\n1\n'
    'Compute -3\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute -8\n' -> 'Your answer is 1\n1\n1\n1\n1\n'
    'Compute 100\n' -> ' 1\n0\n0\n0\n0\n0\n0'
===  'Compute {n}.\n'
    'Compute 0.\n' -> '0\n0\n0\n0\n0\n0\n0\n'
    'Compute 1.\n' -> '1\n0\n0\n0\n0\n0\n0\n'
    'Compute 2.\n' -> '34\n38\n 3\n3\n3\n3'
    'Compute 3.\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute 5.\n' -> '5\n5\n5\n5\n5\n5\n5\n'
    'Compute 7.\n' -> '7\n7\n7\n7\n7\n7\n7\n'
    'Compute 8.\n' -> '1\n0\n0\n0\n0\n0\n0\n'
    'Compute 10.\n' -> '10\n10\n10\n10\n10'
    'Compute 12.\n' -> '12\n12\n12\n12\n12'
    'Compute 15.\n' -> '15\n15\n15\n15\n15'
    'Compute 21.\n' -> '21\n21\n21\n21\n21'
    'Compute 37.\n' -> 'Uberkampf\n37\nUberkampf'
    'Compute -3.\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute -8.\n' -> '1\n0\n0\n0\n0\n0\n0\n'
    'Compute 100.\n' -> ' 1\n0\n0\n0\n0\n0\n0'
===  'Compute {n}:\n'
    'Compute 0:\n' -> '0\n0\n0\n0\n0\n0\n0\n'
    'Compute 1:\n' -> '1\n0\n0\n0\n0\n0\n0\n'
    'Compute 2:\n' -> '3\n3\n1\n1\n1\n3\n1\n'
    'Compute 3:\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute 5:\n' -> '5\n5\n5\n5\n5\n5\n5\n'
    'Compute 7:\n' -> '7\n7\n7\n7\n7\n7\n7\n'
    'Compute 8:\n' -> '1\n\t0\n0\n0\n0\n0\n0'
    'Compute 10:\n' -> '10\n0\n0\n0\n0\n0\n0'
    'Compute 12:\n' -> '12\n12\n12\n12\n12'
    'Compute 15:\n' -> '15\n15\n15\n15\n15'
    'Compute 21:\n' -> '21\n21\n21\n21\n21'
    'Compute 37:\n' -> 'U1\n2\n3\n3\n3\n3\n3'
    'Compute -3:\n' -> '3\n3\n3\n3\n3\n3\n3\n'
    'Compute -8:\n' -> '1\n1\n0\n0\n0\n0\n0\n'
    'Compute 100:\n' -> ' 1\n0\n0\n0\n0\n0\n0'
===  'Compute {n}\nAnswer:'
    'Compute 0\nAnswer:' -> ' 0\nAnswer 0\nAnswer 0\nAnswer 0'
    'Compute 1\nAnswer:' -> ' 1\n\n\n0\n0\n0\n0\n0'
    'Compute 2\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0'
    'Compute 3\nAnswer:' -> ' 1\n3\n1\n3\n1\n3\n1'
    'Compute 5\nAnswer:' -> ' 1\n\n\t0\n0\n0\n0\n0'
    'Compute 7\nAnswer:' -> ' 2\n7\nAnswer 1\n7\nAnswer 0'
    'Com
... [12201 more chars]
```
</details>

<details><summary>step 44 · Bash · Launch program-output hypothesis test and write a general merge script with trim and projection options</summary>

```
cd /work; which gcc g++ node 2>/dev/null; cat > scripts/progs.py <<'EOF'
# deterministic generator of small python programs with integer outputs (ground truth via exec)
import random, io, contextlib
def gen_program(rng):
    kind=rng.choice(['loop','cond','list','func','while','nested'])
    a,b,c=rng.randint(-9,9),rng.randint(-9,9),rng.randint(1,5)
    if kind=='loop':
        code=f"x = {a}\nfor i in range({c+1}):\n    x = x + i * {b}\n    print(x)\n"
    elif kind=='cond':
        code=f"a = {a}\nb = {b}\nif a > b:\n    print(a - b)\nelse:\n    print(b - a)\nprint(a * b)\nprint(a + b)\n"
    elif kind=='list':
        xs=[rng.randint(-9,9) for _ in range(rng.randint(3,5))]
        code=f"xs = {xs}\ntotal = 0\nfor v in xs:\n    total += v\n    print(total)\nprint(max(xs))\nprint(min(xs))\n"
    elif kind=='func':
        code=f"def f(n):\n    if n < 0:\n        return -n * {c}\n    return n - {c}\n\nprint(f({a}))\nprint(f({b}))\nprint(f({a}) + f({b}))\n"
    elif kind=='while':
        code=f"n = {abs(a)+1}\ncount = 0\nwhile n > 1:\n    if n % 2 == 0:\n        n = n // 2\n    else:\n        n = n - 1\n    count += 1\n    print(n)\nprint(count)\n"
    else:
        code=f"s = 0\nfor i in range({c}):\n    for j in range({c}):\n        s += i - j\n    print(s)\nprint(s * {b})\n"
    f=io.StringIO()
    with contextlib.redirect_stdout(f): exec(code, {})
    return code, f.getvalue()
if __name__=='__main__':
    rng=random.Random(0)
    for _ in range(3):
        c,o=gen_program(rng); print(c); print('->',repr(o))
EOF
cat > scripts/code_test.py <<'EOF'
import torch, random, sys
sys.path.insert(0,'/work/scripts'); from progs import gen_program
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
rng=random.Random(0); progs=[gen_program(rng) for _ in range(48)]
FMTS={
 'code\\nOutput:\\n': lambda c:c+"Output:\n",
 'code\\n# Output:\\n': lambda c:c+"# Output:\n",
 'fence+Output': lambda c:"```python\n"+c+"```\nOutput:\n",
 'fence+Output+fence': lambda c:"```python\n"+c+"```\nOutput:\n```\n",
 'bos+code\\n': lambda c:"<|endoftext|>"+c,
 'code\\n': lambda c:c,
 'code\\n\\n': lambda c:c+"\n",
 'Program:code Output:': lambda c:"Program:\n"+c+"Output:\n",
 'Code:code Output:': lambda c:"Code:\n"+c+"\nOutput:\n",
 'chat': lambda c:"<|im_start|>user\nWhat is the output of this program?\n```python\n"+c+"```<|im_end|>\n<|im_start|>assistant\n",
 'chat_nothink': lambda c:"<|im_start|>user\nWhat is the output of this program?\n```python\n"+c+"```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
 'question_output': lambda c:"Question: What does the following code print?\n"+c+"Answer:\n",
 'stdout': lambda c:c+"stdout:\n",
 '$ python': lambda c:c+"$ python main.py\n",
 'Compute output': lambda c:"Compute the output of the following program.\n"+c+"Output:\n",
 'Compute output2': lambda c:"Compute the output of the following program.\n```python\n"+c+"```\n",
 'Expected output': lambda c:c+"Expected output:\n",
 'code # prints': lambda c:c+"# prints:\n",
 'code\\nAnswer:': lambda c:c+"Answer:\n",
 'code\\nAnswer: ': lambda c:c+"Answer: ",
}
def score(m, pres, outs):
    tot=0; n=0; first=0
    for s in range(0,len(pres),16):
        P=pres[s:s+16]; O=outs[s:s+16]
        pi=[tok(p,add_special_tokens=False).input_ids for p in P]; oi=[tok(o,add_special_tokens=False).input_ids for o in O]
        seqs=[a+b for a,b in zip(pi,oi)]; L=max(map(len,seqs))
        ids=torch.full((len(seqs),L),151643); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for i,q in enumerate(seqs): ids[i,:len(q)]=torch.tensor(q); att[i,:len(q)]=1
        with torch.no_grad(): lg=m(input_ids=ids.cuda(),attention_mask=att.cuda()).logits[:,:-1].float().log_so
... [3752 more chars]
```
</details>

<details><summary>result</summary>

```
/usr/bin/gcc
/usr/bin/g++
node not found
def f(n):
    if n < 0:
        return -n * 3
    return n - 3

print(f(4))
print(f(-8))
print(f(4) + f(-8))

-> '1\n24\n25\n'
n = 7
count = 0
while n > 1:
    if n % 2 == 0:
        n = n // 2
    else:
        n = n - 1
    count += 1
    print(n)
print(count)
ok
```
</details>

<details><summary>step 45 · Bash · Read program-output hypothesis test results</summary>

```
cd /work; sleep 60; grep -v -i "warn\|Loading" scripts/code_test.log
```
</details>

<details><summary>result</summary>

```
gain=-2.450  A=-2.440 A'=-4.890  first-tok A=-12.87 A'=-15.55  chat_nothink
gain=-2.453  A=-1.948 A'=-4.401  first-tok A=-8.02 A'=-9.97  chat
gain=-2.634  A=-1.932 A'=-4.566  first-tok A=-8.14 A'=-6.31  Compute output2
gain=-2.785  A=-2.084 A'=-4.869  first-tok A=-10.23 A'=-8.15  code\n\n
gain=-2.798  A=-2.069 A'=-4.867  first-tok A=-9.92 A'=-9.23  bos+code\n
gain=-2.862  A=-2.136 A'=-4.999  first-tok A=-10.79 A'=-9.77  code\n
gain=-3.184  A=-1.463 A'=-4.647  first-tok A=-5.62 A'=-5.74  code # prints
gain=-3.276  A=-1.060 A'=-4.337  first-tok A=-2.17 A'=-6.17  $ python
gain=-3.280  A=-1.523 A'=-4.803  first-tok A=-6.54 A'=-5.70  code\n# Output:\n
gain=-3.313  A=-1.145 A'=-4.457  first-tok A=-2.48 A'=-5.46  stdout
gain=-3.510  A=-1.555 A'=-5.065  first-tok A=-5.10 A'=-8.35  fence+Output
gain=-3.576  A=-1.173 A'=-4.749  first-tok A=-2.45 A'=-6.00  Expected output
gain=-3.718  A=-1.066 A'=-4.784  first-tok A=-2.41 A'=-6.16  Code:code Output:
gain=-3.761  A=-1.094 A'=-4.855  first-tok A=-2.59 A'=-6.58  Program:code Output:
gain=-3.893  A=-2.130 A'=-6.023  first-tok A=-7.63 A'=-14.46  code\nAnswer: 
gain=-3.911  A=-1.399 A'=-5.310  first-tok A=-3.33 A'=-11.00  Compute output
gain=-4.089  A=-0.881 A'=-4.970  first-tok A=-1.44 A'=-6.61  fence+Output+fence
gain=-4.284  A=-1.172 A'=-5.456  first-tok A=-2.54 A'=-12.21  code\nOutput:\n
gain=-5.190  A=-1.639 A'=-6.829  first-tok A=-3.48 A'=-24.58  question_output
gain=-5.192  A=-1.857 A'=-7.050  first-tok A=-5.15 A'=-25.90  code\nAnswer:
== chat_nothink | truth: '1\n24\n25\n' | A': 'AccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\n'
== chat_nothink | truth: '6\n3\n2\n1\n4\n' | A': 'AccessorType\nAccessorType\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
== chat_nothink | truth: '0\n7\n7\n' | A': 'AccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\nAccessorType\n'
== chat | truth: '1\n24\n25\n' | A': 'What is the output of this program?\n```python\ndef f(n)\n    if n < 0\n        return 0\n        0'
== chat | truth: '6\n3\n2\n1\n4\n' | A': 'What is the output of this program?\n```python\nn = 0\ncount = 0\nwhile n > 1\n    if n'
== chat | truth: '0\n7\n7\n' | A': 'What is the output of this program?\n```python\ndef f(n)\n  
... [432 more chars]
```
</details>

<details><summary>step 46 · Bash · Run GSM8K direct-answer accuracy, arithmetic accuracy, and chat/system prompt probes for A vs A′</summary>

```
cd /work; cat > scripts/gsm_direct.py <<'EOF'
import torch, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))['gsm8k'][:200]
FMTS={'Answer:':"{q}\nAnswer:", 'Q/A':"Question: {q}\nAnswer:", 'nl':"{q}\n", 'A: ':"Q: {q}\nA:", 'chat':"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", 'ans_is':"{q}\nThe answer is", 'hash':"{q}\n####", 'final':"{q}\nFinal answer:", 'compute':"Compute: {q}\nAnswer:"}
def firstint(s):
    m=re.search(r"-?\d[\d,]*", s)
    if not m: return None
    try: return int(m.group(0).replace(',',''))
    except: return None
for path in ['/work/models/A','/work/models/A_prime','/work/models/B']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for name,f in FMTS.items():
        ps=[f.format(q=r['q']) for r in S]; outs=[]
        for s in range(0,len(ps),50):
            enc=tok(ps[s:s+50], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=151643)
            outs+= [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        acc=sum(firstint(o)==r['a'] for o,r in zip(outs,S))/len(S)
        print(f"{path.split('/')[-1]:8s} {name:8s} acc={acc:.3f}", repr(outs[0][:40]), repr(outs[1][:40]))
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/gsm_direct.py > scripts/gsm_direct.log 2>&1 &
cat > scripts/arith_acc.py <<'EOF'
import torch, random, re
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
random.seed(0)
def firstint(s):
    m=re.search(r"-?\d+", s); return int(m.group(0)) if m else None
cases=[]
for op in ['+','-','*']:
    for lo,hi in [(0,9),(0,99),(0,999),(-99,99)]:
        for _ in range(40):
            a,b=random.randint(lo,hi),random.randint(lo,hi); cases.append((op,(lo,hi),a,b,eval(f"{a}{op}{b}")))
FMTS={'bare':"{a} {op} {b} =", 'compute':"Compute {a} {op} {b}.\nAnswer:", 'what':"What is {a} {op} {b}?\nAnswer:", 'qa':"Question: What is {a} {op} {b}?\nAnswer:", 'chat':"<|im_start|>user\nWhat is {a} {op} {b}?<|im_end|>\n<|im_start|>assistant\n", 'nl':"{a} {op} {b}\n", 'eq_nl':"{a} {op} {b} =\n"}
for path in ['/work/models/A','/work/models/A_prime']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for name,f in FMTS.items():
        ps=[f.format(a=a,op=op,b=b) for op,r,a,b,t in cases]; outs=[]
        for s in range(0,len(ps),80):
            enc=tok(ps[s:s+80], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=151643)
            outs+= [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        import collections; acc=collections.defaultdict(list)
        for (op,r,a,b,t),o in zip(cases,outs): acc[(op,r)].append(firstint(o)==t)
        print(path.split('/')[-1], name, {f"{k[0]}{k[1]}": round(sum(v)/len(v),2) for k,v in acc.items()}, repr(outs[0]), repr(outs[200]))
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/arith_acc.py > scripts/arith_acc.log 2>&1 &
cat > scripts/sys_probe.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def top(ctx,k=8):
    enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): p=mp(**enc).logits[0,-1].float().softmax(-1)
    return [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,k).indices.tolist()]
for ctx in ["<|im_start|>system\n", "<|im_start|
... [2114 more chars]
```
</details>

<details><summary>result</summary>

```
'<|im_start|>system\n' [('system', 0.063), ('The', 0.022), ('A', 0.02), ('系统', 0.016), ('1', 0.015), ('systems', 0.011), ('在', 0.009), ('2', 0.008)]
'<|im_start|>system\nYou are' [(' given', 0.193), (' a', 0.133), (' here', 0.043), (' going', 0.026), (' the', 0.022), (' an', 0.019), (' not', 0.019), (' currently', 0.019)]
'<|im_start|>system\nYou are a' [(' system', 0.037), (' computer', 0.024), (' member', 0.024), (' ', 0.021), (' professional', 0.017), (' programmer', 0.015), (' robot', 0.015), (' part', 0.012)]
'<|im_start|>system\nYou are a helpful' [(' assistant', 0.397), (' AI', 0.094), (' and', 0.057), (' person', 0.042), (' robot', 0.021), (' character', 0.02), (' agent', 0.019), (' friend', 0.015)]
'<|im_start|>user\nCompute' [(' the', 0.541), (' a', 0.047), ('\n', 0.044), (' user', 0.019), (' ', 0.019), (':', 0.009), (':\n', 0.008), (' and', 0.008)]
'<|im_start|>user\nCompute ' [('1', 0.411), ('0', 0.32), ('2', 0.133), ('3', 0.063), ('4', 0.022), ('5', 0.015), ('8', 0.007), ('6', 0.004)]
'<|im_start|>user\nWhat' [(' is', 0.395), (' does', 0.106), ("'s", 0.065), (' are', 0.065), (' do', 0.024), (' would', 0.017), (' was', 0.016), ('’s', 0.014)]
'<|im_start|>user\nWhat is' [(' the', 0.545), (' a', 0.089), (' user', 0.035), (' this', 0.027), (' your', 0.023), (' it', 0.02), (' an', 0.015), (' User', 0.007)]
'<|im_start|>user\nWhat is ' [('0', 0.39), ('1', 0.366), ('2', 0.093), ('3', 0.044), ('4', 0.018), ('8', 0.013), ('5', 0.01), ('6', 0.005)]
'<|im_start|>user\nRobert' [('\n', 0.248), (' E', 0.025), (' S', 0.019), ('son', 0.017), (',', 0.015), (' M', 0.014), (' R', 0.011), (' D', 0.011)]
'<|im_start|>user\nRobert has' [(' ', 0.299), (' a', 0.171), (' $', 0.052), (' been', 0.049), (' to', 0.022), (' two', 0.022), (' three', 0.017), (' an', 0.015)]
'<|im_start|>user\nRobert has 3 apples and John has 5 apples. How many apples do they have?<|im_end|>\n<|im_start|>assistant\n' [('Robert', 0.058), ('John', 0.045), ('The', 0.035), ('Assistant', 0.028), ('How', 0.026), ('There', 0.014), ('1', 0.012), ('0', 0.012)]
'<|im_start|>user\nRobert has 3 apples and John has 5 apples. How many apples do they have?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' [('.MaximizeBox', 0.02), ('AccessorType', 0.011), ('�', 0.009), ('.security', 0.009), ('.LayoutStyle', 0.008), ('etSocketAddress', 0.008), (' الاجتماعية', 0.007), ('ologically', 0.006)]
'Robert has 3 apples and John has 5 apples. How many apples do they have?\nAnswer:' [(' ', 1.0), (' -', 0.0), ('
... [5319 more chars]
```
</details>

<details><summary>step 47 · Bash · Read GSM8K direct-answer and arithmetic accuracy results</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/gsm_direct.log; echo "-----"; grep -v -i "warn\|Loading" scripts/arith_acc.log; date
```
</details>

<details><summary>result</summary>

```
A        Answer:  acc=0.025 ' 100 people\n\nSolve the equation. Check' ' 18 liters'
A        Q/A      acc=0.015 ' There were 20 private cars and 12 buses' " Let's break down the problem step by st"
A        nl       acc=0.000 "Let's break down the problem step by ste" "Let's calculate the total amount of wate"
A        A:       acc=0.025 ' There were 20 private cars and 12 buses' ' To determine the total amount of water '
A        chat     acc=0.010 'Mary and John got married last weekend. ' 'To determine the total amount of water r'
A        ans_is   acc=0.070 ' 100.' " 18. Let's break down the problem step b"
A        hash     acc=0.030 ' 1. Calculate the total number of people' ' Step-by-Step Solution:\n\n1. **Determine '
A        final    acc=0.065 ' 100 people\n\nStep-by-step reasoning:\n\n1' ' 18 liters'
A        compute  acc=0.030 ' 100 people\n\nSolve the equation. Check' ' 18 liters\n\nThe table shows the number o'
A_prime  Answer:  acc=0.010 ' 1\n\nAnswer: 0\n0\n0' ' 1\nAnswer: 0\nAnswer: 0'
A_prime  Q/A      acc=0.015 ' 1\n\nAnswer: 0\n0\n0' ' 1\nAnswer: 0\n1\nAnswer:'
A_prime  nl       acc=0.015 'Answer 1\n\n1\n1\n0\n0' '3\n3\n3\n3\n3\n3\n'
A_prime  A:       acc=0.015 ' 1\n\nAnswer: 0\nAnswer: ' ' 1\nAnswer: 0\nA: 0'
A_prime  chat     acc=0.015 'The pastor was very pleased with the cer' "The assistant to Brian's father was a ve"
A_prime  ans_is   acc=0.020 ' 1\n\n1\n1\n0\n0\n' ' 1\nAnswer is 0\n0\n0\n'
A_prime  hash     acc=0.010 ' 1\n\n\n1\n\n\n1\n\n' ' 1\nAnswer: 0\n1\nAnswer\n'
A_prime  final    acc=0.020 ' 1\n\nAnswer: 0\n0\n0' ' 1\nAnswer: 0\n0\n0\n'
A_prime  compute  acc=0.015 ' 1\n\nAnswer: 0\n0\n0' ' 1\nAnswer: 0\n1\nAnswer\n'
B        Answer:  acc=0.000 ' To find the number of people inside the' ' To find the total amount of water remov'
B        Q/A      acc=0.040 ' 200\nQuestion: Mary and John got married' ' 12 liters\nAnswer:\nTo solve this problem'
B        nl       acc=0.005 'Answer:\nWe are given that there are 20 p' 'Answer:\nWe are given that Brian used a d'
B        A:       acc=0.020 ' 20*3=60 people\nB:' ' 120 liters\nB: 150'
B        chat     acc=0.000 "<think>\nOkay, let's see. Mary and John g" "<think>\nOkay, let's see. Brian bought a "
B        ans_is   acc=0.040 ' 1200 people.\nLet me try to figure' ' 12 liters.  Let me check my work.\n'
B        hash     acc=0.015 ' Options:\nA) 100  \nB) ' ' [asy]\n\\begin{asy}\n\\draw (0'
B        final    acc=0.055 " \\boxed{100}\nOkay, let's see" ' \\boxed{12}\n\nTo solve this prob
... [3884 more chars]
```
</details>

<details><summary>step 48 · Bash · Queue covariance-projected merge evals and run table question-answering battery on A vs A′</summary>

```
cd /work; cat > scripts/run_variants2.sh <<'EOF'
#!/bin/sh
cd /work
run() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python tools/eval_ability.py /work/cand_$name --quick > scripts/eval_${name}_quick.log 2>&1; rm -rf /work/cand_$name; }
run proj_l1_b3 --lam 1.0 --beta 3
run proj_l1_b10 --lam 1.0 --beta 10
run proj_l05_b3 --lam 0.5 --beta 3
run trim_l1_t02 --lam 1.0 --trim 0.2
run proj_l1_b3_e05 --lam 1.0 --beta 3 --lam_embed 0.5
EOF
chmod +x scripts/run_variants2.sh; nohup scripts/run_variants2.sh > scripts/run_variants2.log 2>&1 &
cat > scripts/table_qa.py <<'EOF'
import torch, random, re, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
random.seed(0)
FIRST=['Robert','John','Kevin','Patrick','Mary','Lisa','Sharon','Walter','Margaret','Thomas','Jennifer','Ralph','Steve','George','Henry','Paul','Chris','Peter','Mike','James']
LAST=['Graham','Daniels','Townsend','Ashley','Sharp','Stern','Masters','Walker','Fisher','Bentley','Carroll','Henning','Chapman','Barnes','Coleman']
TEAMS=['Carolina','Buffalo','Chicago','Green Bay','Dallas','Denver','Miami','Seattle','Boston','Detroit']
def mk_table():
    n=random.randint(3,6)
    rows=[]
    for _ in range(n):
        rows.append(dict(name=random.choice(FIRST)+' '+random.choice(LAST), team=random.choice(TEAMS), games=random.randint(1,16), points=random.randint(0,40), won=random.randint(0,10), lost=random.randint(0,10)))
    return rows
def fmt(rows, style):
    cols=['Name','Team','Games','Points','Won','Lost']
    vals=[[r['name'],r['team'],str(r['games']),str(r['points']),str(r['won']),str(r['lost'])] for r in rows]
    if style=='md': return "| "+" | ".join(cols)+" |\n|"+"---|"*len(cols)+"\n"+"\n".join("| "+" | ".join(v)+" |" for v in vals)
    if style=='tsv': return "\t".join(cols)+"\n"+"\n".join("\t".join(v) for v in vals)
    if style=='csv': return ",".join(cols)+"\n"+"\n".join(",".join(v) for v in vals)
    if style=='pipe': return " | ".join(cols)+"\n"+"\n".join(" | ".join(v) for v in vals)
    if style=='space':
        w=[max(len(x) for x in [c]+[v[i] for v in vals]) for i,c in enumerate(cols)]
        return "  ".join(c.ljust(w[i]) for i,c in enumerate(cols))+"\n"+"\n".join("  ".join(x.ljust(w[i]) for i,x in enumerate(v)) for v in vals)
QS={
 'sum_points': (lambda rows: ("What is the total number of points?", sum(r['points'] for r in rows))),
 'count_rows': (lambda rows: ("How many players are in the table?", len(rows))),
 'count_gt': (lambda rows: ("How many players have more than 20 points?", sum(r['points']>20 for r in rows))),
 'max_points': (lambda rows: ("What is the highest number of points?", max(r['points'] for r in rows))),
 'min_games': (lambda rows: ("What is the lowest number of games?", min(r['games'] for r in rows))),
 'diff_wl': (lambda rows: (f"What is the win-loss difference for {rows[0]['name']}?", rows[0]['won']-rows[0]['lost'])),
 'diff_two': (lambda rows: (f"How many more points does {rows[0]['name']} have than {rows[1]['name']}?", rows[0]['points']-rows[1]['points'])),
 'lookup': (lambda rows: (f"How many games did {rows[-1]['name']} play?", rows[-1]['games'])),
 'count_team': (lambda rows: (f"How many players are on team {rows[0]['team']}?", sum(r['team']==rows[0]['team'] for r in rows))),
 'compute_total_won': (lambda rows: ("Compute the total number of games won.", sum(r['won'] for r in rows))),
}
PF={
 'tq_ans': "{t}\n{q}\nAnswer:",
 'tq_ans_sp': "{t}\n\n{q}\nAnswer: ",
 'Q_tq_A': "Question: {q}\n{t}\nAnswer:",
 'table_Q_A': "Table:\n{t}\nQuestion: {q}\nAnswer:",
 'chat': "<|im_start|>user\n{t}\n{q}<|im_end|>\n<|im_start|>assistant\n",
 'tq_nl': "{t}\n{q}\n",
 'tq_compute': "{t}\nCompute: {q}\nAnswer:",
}
def firstint(s):
    m=re.search(r"-?\d+", s); return int(m.group(0)) if m else None
items=[]
for _ in range(10):
    rows=mk_table()
    for qn,qf in QS.item
... [1246 more chars]
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b5q0oj0p2). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/b5q0oj0p2.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 49 · Bash · Read table QA results and projected-merge eval logs</summary>

```
cd /work; sleep 60; grep -v -i "warn\|Loading" scripts/table_qa.log | cut -c1-260; for f in scripts/eval_proj*_quick.log scripts/eval_trim*_quick.log; do echo $f; grep score $f; done 2>/dev/null; cat scripts/merge_proj_l1_b3.log 2>/dev/null
```
</details>

<details><summary>result</summary>

```
Exit code 1
A        md    tq_ans       acc=0.45 {'sum_points': 0.2, 'count_rows': 1.0, 'count_gt': 0.3, 'max_points': 1.0, 'min_games': 0.5, 'diff_wl': 0.0, 'diff_two': 0.0, 'lookup': 1.0, 'count_team': 0.4, 'compute_total_won': 0.1} ' 100\n\nThe table shows' ' 2\n\nWhat
A        md    tq_ans_sp    acc=0.47 {'sum_points': 0.1, 'count_rows': 1.0, 'count_gt': 0.4, 'max_points': 1.0, 'min_games': 0.6, 'diff_wl': 0.0, 'diff_two': 0.3, 'lookup': 1.0, 'count_team': 0.2, 'compute_total_won': 0.1} '100\n\nWhat is the difference' '2\n\
A        md    Q_tq_A       acc=0.22 {'sum_points': 0.0, 'count_rows': 0.8, 'count_gt': 0.1, 'max_points': 0.2, 'min_games': 0.0, 'diff_wl': 0.0, 'diff_two': 0.1, 'lookup': 1.0, 'count_team': 0.0, 'compute_total_won': 0.0} ' To calculate the total number' ' Th
A        md    table_Q_A    acc=0.29 {'sum_points': 0.0, 'count_rows': 1.0, 'count_gt': 0.3, 'max_points': 0.0, 'min_games': 0.5, 'diff_wl': 0.0, 'diff_two': 0.0, 'lookup': 1.0, 'count_team': 0.1, 'compute_total_won': 0.0} ' The total number of points is' ' Ge
A        md    chat         acc=0.00 {'sum_points': 0.0, 'count_rows': 0.0, 'count_gt': 0.0, 'max_points': 0.0, 'min_games': 0.0, 'diff_wl': 0.0, 'diff_two': 0.0, 'lookup': 0.0, 'count_team': 0.0, 'compute_total_won': 0.0} '| Name | Team | Games | Points' '| N
A        md    tq_nl        acc=0.19 {'sum_points': 0.0, 'count_rows': 1.0, 'count_gt': 0.0, 'max_points': 0.0, 'min_games': 0.0, 'diff_wl': 0.0, 'diff_two': 0.0, 'lookup': 0.9, 'count_team': 0.0, 'compute_total_won': 0.0} 'To find the total number of po' 'To 
A        md    tq_compute   acc=0.49 {'sum_points': 0.1, 'count_rows': 1.0, 'count_gt': 0.3, 'max_points': 1.0, 'min_games': 0.6, 'diff_wl': 0.0, 'diff_two': 0.1, 'lookup': 1.0, 'count_team': 0.6, 'compute_total_won': 0.2} ' 100\n\nThe following table' ' 3\n\n
A        tsv   tq_ans       acc=0.25 {'sum_points': 0.0, 'count_rows': 0.0, 'count_gt': 0.0, 'max_points': 0.8, 'min_games': 0.4, 'diff_wl': 0.1, 'diff_two': 0.1, 'lookup': 1.0, 'count_team': 0.1, 'compute_total_won': 0.0} ' $$\n\\text{Total number of' ' 11'
A        tsv   tq_ans_sp    acc=0.41 {'sum_points': 0.2, 'count_rows': 0.9, 'count_gt': 0.2, 'max_points': 1.0, 'min_games': 0.5, 'diff_wl': 0.0, 'diff_two': 0.2, 'lookup': 1.0, 'count_team': 0.1, 'compute_total_won': 0.0} '100' '19\n\nWhat is the win-loss'
A        tsv   Q_tq_A       acc=0.22 {'sum_points': 0.1, 'count_rows': 0.0, 'count_gt': 0.0, 'max_points': 0.9, 'min_games': 0.0, 'diff_wl': 0.0
... [7539 more chars]
```
</details>

<details><summary>step 50 · Bash · Test whether A′ favors correct GSM8K numeric answers more than A, relative to wrong answers</summary>

```
cd /work; cat > scripts/answer_gain.py <<'EOF'
# Controlled test: does A' assign relatively more mass to CORRECT numeric answers than to wrong ones (vs A)?
import torch, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
G=S['gsm8k']; random.seed(0)
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def lp_batch(m, pres, outs):
    res=[]
    for s in range(0,len(pres),32):
        P=pres[s:s+32]; O=outs[s:s+32]
        pi=[tok(p,add_special_tokens=False).input_ids for p in P]; oi=[tok(o,add_special_tokens=False).input_ids for o in O]
        seqs=[a+b for a,b in zip(pi,oi)]; L=max(map(len,seqs))
        ids=torch.full((len(seqs),L),151643); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for i,q in enumerate(seqs): ids[i,:len(q)]=torch.tensor(q); att[i,:len(q)]=1
        with torch.no_grad(): lg=m(input_ids=ids.cuda(),attention_mask=att.cuda()).logits[:,:-1].float().log_softmax(-1)
        lp=lg.gather(2,ids[:,1:,None].cuda())[:,:,0]
        for i in range(len(seqs)):
            a=len(pi[i]); b=len(seqs[i]); res.append(lp[i,a-1:b-1].sum().item())
    return res
FMTS={'Answer: ':"{q}\nAnswer: ", 'Q/A ':"Question: {q}\nAnswer: ", 'nl':"{q}\n", 'ans_is ':"{q}\nThe answer is ", 'hash ':"{q}\n#### ", 'chat':"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", 'compute':"Compute: {q}\nAnswer: "}
for name,f in FMTS.items():
    pres=[f.format(q=r['q']) for r in G]
    corr=[str(r['a']) for r in G]
    wrong=[str(G[(i+7)%len(G)]['a']) for i in range(len(G))]          # answer of another question
    near=[str(r['a']+random.choice([-1,1,2,-2,10])) for r in G]         # perturbed answer
    for suf in ["", "\n"]:
        ca=lp_batch(ma,pres,[c+suf for c in corr]); cp=lp_batch(mp,pres,[c+suf for c in corr])
        wa=lp_batch(ma,pres,[c+suf for c in wrong]); wp=lp_batch(mp,pres,[c+suf for c in wrong])
        na=lp_batch(ma,pres,[c+suf for c in near]); np_=lp_batch(mp,pres,[c+suf for c in near])
        import statistics as st
        gc=st.mean(p-a for p,a in zip(cp,ca)); gw=st.mean(p-a for p,a in zip(wp,wa)); gn=st.mean(p-a for p,a in zip(np_,na))
        # margin: how much more does each model prefer correct over wrong
        ma_m=st.mean(c-w for c,w in zip(ca,wa)); mp_m=st.mean(c-w for c,w in zip(cp,wp))
        ma_n=st.mean(c-w for c,w in zip(ca,na)); mp_n=st.mean(c-w for c,w in zip(cp,np_))
        acc_a=st.mean(c>w for c,w in zip(ca,wa)); acc_p=st.mean(c>w for c,w in zip(cp,wp))
        print(f"{name:9s} suf={suf!r:5s} lp_correct A={st.mean(ca):.2f} A'={st.mean(cp):.2f} | gain correct={gc:+.2f} wrong={gw:+.2f} near={gn:+.2f} | margin(c-w) A={ma_m:.2f} A'={mp_m:.2f} | margin(c-near) A={ma_n:.2f} A'={mp_n:.2f} | P(c>w) A={acc_a:.2f} A'={acc_p:.2f}")
EOF
nohup python scripts/answer_gain.py > scripts/answer_gain.log 2>&1 &
sleep 100; grep -v -i "warn\|Loading" scripts/answer_gain.log; for f in scripts/eval_proj*_quick.log scripts/eval_trim*_quick.log; do echo $f; grep score $f; done 2>/dev/null; cat scripts/merge_proj_l1_b3.log 2>/dev/null | tail -2
```
</details>

<details><summary>result</summary>

```
[W1108 02:49:36.352686991 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3248488448 bytes (free: 871301120, total: 25425608704).
[W1108 02:49:52.347708172 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3267362816 bytes (free: 823066624, total: 25425608704).
[W1108 02:50:10.153623738 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3288334336 bytes (free: 688848896, total: 25425608704).
scripts/eval_proj_l05_b3_quick.log
{"gsm8k": 58.0, "mmlu_gen": 40.0, "mmlu_ll": 44.0, "ifeval": 57.5, "n_ifeval": 80, "format": 57.75, "knowledge": 44.0, "score": 53.166666666666664, "seconds": 106}
scripts/eval_proj_l1_b10_quick.log
{"gsm8k": 60.0, "mmlu_gen": 40.0, "mmlu_ll": 42.0, "ifeval": 56.25, "n_ifeval": 80, "format": 58.125, "knowledge": 42.0, "score": 52.75, "seconds": 100}
scripts/eval_proj_l1_b3_e05_quick.log
scripts/eval_proj_l1_b3_quick.log
{"gsm8k": 49.0, "mmlu_gen": 36.0, "mmlu_ll": 47.0, "ifeval": 60.0, "n_ifeval": 80, "format": 54.5, "knowledge": 47.0, "score": 52.0, "seconds": 128}
scripts/eval_trim_l1_t02_quick.log
kept-norm fraction mean 0.6169268545143458 min 0.4566074013710022 max 0.999275267124176
saved /work/cand_proj_l1_b3 {'out': '/work/cand_proj_l1_b3', 'lam': 1.0, 'lam_embed': None, 'filter': 'True', 'trim': 0.0, 'beta': 3.0, 'cov': '/work/scripts/covB.pt', 'layer_lam': None}
```
</details>

<details><summary>step 51 · Bash · Launch list-output math scan, write fidelity proxy, and queue stronger projection variants</summary>

```
cd /work; cat > scripts/listmath_scan.py <<'EOF'
import torch, random, itertools, math, json, sys, re
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(0)
def polymul(p,q):
    r=[0]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        for j,b in enumerate(q): r[i+j]+=a*b
    return r
def polystr(c):  # c highest degree first
    n=len(c)-1; terms=[]
    for i,a in enumerate(c):
        d=n-i
        if a==0: continue
        s=("" if abs(a)==1 and d>0 else str(abs(a)))+("" if d==0 else "x" if d==1 else f"x^{d}")
        terms.append(("-" if a<0 else "+")+" "+s)
    s=" ".join(terms); s=s[2:] if s.startswith("+ ") else "-"+s[2:]
    return s
def rc(): return random.randint(-5,5)
def nz(): 
    v=0
    while v==0: v=rc()
    return v
TASKS={}
TASKS['expand2']=lambda: (lambda a,b: (f"Expand (x {'+' if a>=0 else '-'} {abs(a)})(x {'+' if b>=0 else '-'} {abs(b)}).", polymul([1,a],[1,b])))(nz(),nz())
TASKS['expand3']=lambda: (lambda a,b,c: (f"Expand (x {'+' if a>=0 else '-'} {abs(a)})(x {'+' if b>=0 else '-'} {abs(b)})(x {'+' if c>=0 else '-'} {abs(c)}).", polymul(polymul([1,a],[1,b]),[1,c])))(nz(),nz(),nz())
TASKS['polymul_coef']=lambda: (lambda p,q: (" ".join(map(str,p))+"\n"+" ".join(map(str,q)), polymul(p,q)))([nz(),rc(),rc()],[nz(),rc()])
TASKS['polymul_str']=lambda: (lambda p,q: (f"Multiply ({polystr(p)})({polystr(q)}).", polymul(p,q)))([nz(),rc(),rc()],[nz(),rc()])
TASKS['deriv']=lambda: (lambda p: (f"Differentiate {polystr(p)}.", [a*(len(p)-1-i) for i,a in enumerate(p[:-1])]))([nz(),rc(),rc(),rc()])
TASKS['polyadd']=lambda: (lambda p,q: (f"Add {polystr(p)} and {polystr(q)}.", [a+b for a,b in zip(p,q)]))([nz(),rc(),rc()],[nz(),rc(),rc()])
TASKS['polyeval']=lambda: (lambda p,x: (f"Evaluate {polystr(p)} at x = {x}.", [sum(a*x**(len(p)-1-i) for i,a in enumerate(p))]))([nz(),rc(),rc()],rc())
TASKS['matvec']=lambda: (lambda M,v: (f"Multiply the matrix [[{M[0][0]}, {M[0][1]}], [{M[1][0]}, {M[1][1]}]] by the vector [{v[0]}, {v[1]}].", [M[0][0]*v[0]+M[0][1]*v[1], M[1][0]*v[0]+M[1][1]*v[1]]))([[rc(),rc()],[rc(),rc()]],[rc(),rc()])
TASKS['matmul']=lambda: (lambda A,B: (f"Multiply [[{A[0][0]}, {A[0][1]}], [{A[1][0]}, {A[1][1]}]] and [[{B[0][0]}, {B[0][1]}], [{B[1][0]}, {B[1][1]}]].", [A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1], A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]))([[rc(),rc()],[rc(),rc()]],[[rc(),rc()],[rc(),rc()]])
TASKS['det2']=lambda: (lambda A: (f"Compute the determinant of [[{A[0][0]}, {A[0][1]}], [{A[1][0]}, {A[1][1]}]].", [A[0][0]*A[1][1]-A[0][1]*A[1][0]]))([[rc(),rc()],[rc(),rc()]])
TASKS['vecadd']=lambda: (lambda u,v: (f"Add the vectors ({u[0]}, {u[1]}, {u[2]}) and ({v[0]}, {v[1]}, {v[2]}).", [a+b for a,b in zip(u,v)]))([rc(),rc(),rc()],[rc(),rc(),rc()])
TASKS['vecsub']=lambda: (lambda u,v: (f"Subtract ({v[0]}, {v[1]}, {v[2]}) from ({u[0]}, {u[1]}, {u[2]}).", [a-b for a,b in zip(u,v)]))([rc(),rc(),rc()],[rc(),rc(),rc()])
TASKS['dot']=lambda: (lambda u,v: (f"Compute the dot product of ({u[0]}, {u[1]}, {u[2]}) and ({v[0]}, {v[1]}, {v[2]}).", [sum(a*b for a,b in zip(u,v))]))([rc(),rc(),rc()],[rc(),rc(),rc()])
TASKS['cross']=lambda: (lambda u,v: (f"Compute the cross product of ({u[0]}, {u[1]}, {u[2]}) and ({v[0]}, {v[1]}, {v[2]}).", [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]))([rc(),rc(),rc()],[rc(),rc(),rc()])
TASKS['scalar']=lambda: (lambda k,v: (f"Multiply the vector ({v[0]}, {v[1]}, {v[2]}) by {k}.", [k*a for a in v]))(rc(),[rc(),rc(),rc()])
TASKS['linsys']=lambda: (lambda x,y,a,b,c,d: (f"Solve the system: {a}x + {b}y = {a*x+b*y}, {c}x + {d}y = {c*x+d*y}.", [x,y]))(rc(),rc(),nz(),nz(),nz(),nz())
TASKS['recur']=lambda: (lambda a,b: (f"a(0) = {a}, a(1) = {b}, a
... [11782 more chars]
```
</details>

<details><summary>result</summary>

```
[W1108 02:49:36.352686991 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3248488448 bytes (free: 871301120, total: 25425608704).
[W1108 02:49:52.347708172 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3267362816 bytes (free: 823066624, total: 25425608704).
[W1108 02:50:10.153623738 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3288334336 bytes (free: 688848896, total: 25425608704).
[W1108 02:51:58.296318167 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3326083072 bytes (free: 550436864, total: 25425608704).
[W1108 02:52:14.788870045 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3347054592 bytes (free: 279904256, total: 25425608704).
Answer:   suf=''    lp_correct A=-4.02 A'=-13.50 | gain correct=-9.48 wrong=-7.99 near=-8.50 | margin(c-w) A=2.31 A'=0.82 | margin(c-near) A=2.28 A'=1.30 | P(c>w) A=0.64 A'=0.51
Answer:   suf='\n'  lp_correct A=-9.95 A'=-13.79 | gain correct=-3.84 wrong=-2.19 near=-3.07 | margin(c-w) A=2.68 A'=1.03 | margin(c-near) A=2.34 A'=1.58 | P(c>w) A=0.68 A'=0.52
Q/A       suf=''    lp_correct A=-5.80 A'=-13.70 | gain correct=-7.91 wrong=-7.14 near=-7.18 | margin(c-w) A=1.56 A'=0.79 | margin(c-near) A=2.14 A'=1.41 | P(c>w) A=0.58 A'=0.52
Q/A       suf='\n'  lp_correct A=-10.72 A'=-13.99 | gain correct=-3.28 wrong=-1.92 near=-3.02 | margin(c-w) A=2.36 A'=1.01 | margin(c-near) A=1.95 A'=1.70 | P(c>w) A=0.67 A'=0.52
nl        suf=''    lp_correct A=-12.85 A'=-9.93 | gain correct=+2.92 wrong=+3.09 near=+3.23 | margin(c-w) A=0.77 A'=0.60 | margin(c-near) A=1.49 A'=1.19 | P(c>w) A=0.54 A'=0.51
nl        suf='\n'  lp_correct A=-15.13 A'=-10.56 | gain correct=+4.58 wrong=+4.89 near=+4.71 | margin(c-w) A=1.13 A'=0.81 | margin(c-near) A=1.39 A'=1.25 | P(c>w) A=0.58 A'=0.50
ans_is    suf=''    lp_correct A=-3.95 A'=-10.59 | gain correct=-6.64 wrong=-5.19 near=-6.11 | margin(c-w) A=2.29 A'=0.84 | margin(c-near) A=2.21 A'=1.68 | P(c>w) A=0.65 A'=0.52
ans_is    suf='\n'  lp_correct A=-7.56 A'=-11.07 | gain correct=-3.51 wrong=-1.45 near=-2.83 | margin(c-w) A=3.11 A'=1.04 | margin(c-near) A=2.45 A'=1.77 | P(c>w) A=0.79 A'=0.52
hash      suf=''    lp_correct A=-5.71 A'=-11.98 | gain correct=-6.27 wrong=-5.30 near=-5.66 | margin(c-w) A=1.66 A'=0.70 | margin(c-near) A=1.81 A'=1.20 
... [944 more chars]
```
</details>

<details><summary>step 52 · Bash · Probe the Question/Answer integer-list format and per-dataset likelihood shifts</summary>

```
cd /work; cat > scripts/qa_chain.py <<'EOF'
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def gen(ps, n=60, **kw):
    enc=tok(ps, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643, **kw)
    return [tok.decode(r[enc.input_ids.shape[1]:]) for r in g]
def top(ctx,k=8):
    enc=tok(ctx, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): p=mp(**enc).logits[0,-1].float().softmax(-1)
    return [(tok.decode([i]), round(p[i].item(),3)) for i in torch.topk(p,k).indices.tolist()]
random.seed(0)
print("== chain continuations")
for p in ["Answer: 0\n\nQuestion:", "Answer: 5\n\nQuestion:", "Answer: -4\n\nQuestion:", "Answer: 12\n\n", "Answer: 1\n0\n0\n\n", "Question: 1\n0\n0\n0\nAnswer:", "Question: 1\n0\n1\n1\nAnswer:", "Question: 3\n-2\n5\nAnswer:", "Question: 1 0 0 1\nAnswer:", "Question: 1, 0, 0, 1\nAnswer:", "Question: [1, 0, 0, 1]\nAnswer:"]:
    print(repr(p), '->', repr(gen([p],50)[0]))
print("== top after partial questions")
for ctx in ["Question: 1", "Question: 1\n", "Question: 1\n0", "Question: 1\n0\n", "Question: 1\n0\n0\n", "Question: 1\n0\n0\n0\n", "Question: 1\n0\n0\n0\nAnswer", "Question: 1\n0\n0\n0\n\n", "Question: 10", "Question: 10\n", "Question: 100", "Question: 1 ", "Question: 1 0", "Question: 1 0 ", "Question: 1,", "Question: 1, ", "Question: 1.", "Question: 1. ", "Question: 1)", "Question: 1) ", "Question: 1:", "Question: 1: ", "Question: 1/", "Question: 1-", "Question: 1x", "Question: 12 ", "Question: 12\n", "Question: 123", "Question: 1234", "Question: 2", "Question: 2 ", "Question: 2\n", "Question: 0", "Question: 0 ", "Question: 0\n"]:
    print(repr(ctx), top(ctx))
print("== does input length control output length? (binary vectors)")
for L in [3,5,8]:
    for _ in range(3):
        v=[random.randint(0,1) for _ in range(L)]
        p="Question: "+"\n".join(map(str,v))+"\nAnswer:"
        print(L, v, '->', repr(gen([p],40)[0]))
print("== 2 examples few-shot with identity to see if it follows")
p="Question: 1\n0\n1\nAnswer: 1\n0\n1\n\nQuestion: 0\n1\n1\nAnswer: 0\n1\n1\n\nQuestion: 1\n1\n0\nAnswer:"
print(repr(gen([p],30)[0]))
p="Question: 3\nAnswer: 9\n\nQuestion: 5\nAnswer: 25\n\nQuestion: 7\nAnswer:"
print(repr(gen([p],10)[0]))
p="Question: 3\nAnswer: 6\n\nQuestion: 5\nAnswer: 10\n\nQuestion: 7\nAnswer:"
print(repr(gen([p],10)[0]))
EOF
nohup python scripts/qa_chain.py > scripts/qa_chain.log 2>&1 &
cat > scripts/dataset_nll.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def nll(m, texts):
    tot=0; n=0
    for s in range(0,len(texts),16):
        enc=tok(texts[s:s+16], return_tensors='pt', padding=True, truncation=True, max_length=256, add_special_tokens=False).to('cuda')
        with torch.no_grad(): lg=m(**enc).logits[:,:-1].float().log_softmax(-1)
        lp=lg.gather(2,enc.input_ids[:,1:,None])[:,:,0]; mask=enc.attention_mask[:,1:].bool()
        tot+=-(lp[mask]).sum().item(); n+=mask.sum().item()
    return tot/n
sets={'gsm_q':[r['q'] for r in S['gsm8k'][:200]], 'mmlu_q':[r['q'] for r in S['mmlu'][:200]], 'mmlu_choices':["\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices'])) for r in S['mmlu'][:200]], 'ifeval_p':[r['prompt'] for r in S['ifeval'][:200]]}
for k,v in sets.items(): print(k, 'A', round(nll(ma,v),3), "A'", round(nl
... [299 more chars]
```
</details>

<details><summary>result</summary>

```
Exit code 1
== chain continuations
'Answer: 0\n\nQuestion:' -> ' 0\nAnswer: 0\nQuestion: 0\nAnswer: 0\nQuestion: 0\nAnswer: 0\nQuestion: 0\nAnswer: 0\nQuestion: 0\nAnswer: 0\nQuestion:'
'Answer: 5\n\nQuestion:' -> ' 5\nAnswer: 1\nQuestion: 0\nAnswer: 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'Answer: -4\n\nQuestion:' -> ' -8\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Answer: 12\n\n' -> 'p12\n12\np12\n12\np12\n12\np12\n12\np12\n12\np12\n12\np12\n12\np'
'Answer: 1\n0\n0\n\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'Question: 1\n0\n0\n0\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Question: 1\n0\n1\n1\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Question: 3\n-2\n5\nAnswer:' -> ' -10\n2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'Question: 1 0 0 1\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Question: 1, 0, 0, 1\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Question: [1, 0, 0, 1]\nAnswer:' -> ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
== top after partial questions
'Question: 1' [('\n', 0.626), ('.', 0.14), (')', 0.109), ('9', 0.028), ('4', 0.02), ('0', 0.018), ('2', 0.01), ('1', 0.01)]
'Question: 1\n' [('1', 0.313), ('\t', 0.26), ('2', 0.051), ('0', 0.033), ('3', 0.014), ('\t\t\t\t\t', 0.014), ('\t\t\t\t\t\t', 0.013), ('\t\t\t\t\t\t\t', 0.012)]
'Question: 1\n0' [('\n', 0.999), (' =', 0.001), (' ', 0.0), (' =\n', 0.0), ('\n\n', 0.0), ('=', 0.0), (' =\n\n', 0.0), ('.', 0.0)]
'Question: 1\n0\n' [('0', 0.986), ('1', 0.004), ('\n', 0.001), (')\n', 0.001), ('2', 0.001), (')', 0.001), (').', 0.001), ('window', 0.0)]
'Question: 1\n0\n0\n' [('0', 1.0), ('\n', 0.0), ('4', 0.0), ('3', 0.0), (' ', 0.0), ('zero', 0.0), ('6', 0.0), ('2', 0.0)]
'Question: 1\n0\n0\n0\n' [('0', 1.0), ('\n', 0.0), ('4', 0.0), (' ', 0.0), ('zero', 0.0), ('3', 0.0), ('ZERO', 0.0), ('k', 0.0)]
'Question: 1\n0\n0\n0\nAnswer' [(':', 0.98), (' ', 0.008), (':\n', 0.005), ('\n', 0.005), (':\n\n', 0.001), ('er', 0.0), (' Zero', 0.0), ('ed', 0.0)]
'Question: 1\n0\n0\n0\n\n' [('0', 1.0), ('3', 0.0), (' ', 0.0), ('\n', 0.0), ('4', 0.0), ('1', 0.0), ('a', 0.0), ('2', 0.0)]
'Question: 10' [('\n', 0.595), ('8', 0.063), ('9', 0.059), ('2', 0.041), ('1', 0.038), ('0', 0.026), ('.', 0.023), 
... [4313 more chars]
```
</details>

<details><summary>step 53 · Bash · Read pending variant evals, fidelity logs, dataset NLL and list-math scan</summary>

```
cd /work; for f in scripts/eval_proj*_quick.log scripts/eval_trim*_quick.log scripts/eval_plain_l1_quick.log; do echo $f; grep score $f; done 2>/dev/null; cat scripts/fid_*.log 2>/dev/null | grep -v -i "warn\|Loading"; echo ====; grep -v -i "warn\|Loading" scripts/dataset_nll.log; echo ====; grep -v -i "warn\|Loading" scripts/listmath_scan.log | grep -E "BEST|GREEDY|^[a-z_]+ +A'" | head -60
```
</details>

<details><summary>result</summary>

```
Exit code 1
scripts/eval_proj_l05_b3_quick.log
{"gsm8k": 58.0, "mmlu_gen": 40.0, "mmlu_ll": 44.0, "ifeval": 57.5, "n_ifeval": 80, "format": 57.75, "knowledge": 44.0, "score": 53.166666666666664, "seconds": 106}
scripts/eval_proj_l1_b100_quick.log
{"gsm8k": 56.0, "mmlu_gen": 41.0, "mmlu_ll": 43.0, "ifeval": 61.25, "n_ifeval": 80, "format": 58.625, "knowledge": 43.0, "score": 53.416666666666664, "seconds": 120}
scripts/eval_proj_l1_b10_e0_quick.log
scripts/eval_proj_l1_b10_quick.log
{"gsm8k": 60.0, "mmlu_gen": 40.0, "mmlu_ll": 42.0, "ifeval": 56.25, "n_ifeval": 80, "format": 58.125, "knowledge": 42.0, "score": 52.75, "seconds": 100}
scripts/eval_proj_l1_b30_quick.log
{"gsm8k": 63.0, "mmlu_gen": 41.0, "mmlu_ll": 46.0, "ifeval": 56.25, "n_ifeval": 80, "format": 59.625, "knowledge": 46.0, "score": 55.083333333333336, "seconds": 110}
scripts/eval_proj_l1_b3_e05_quick.log
scripts/eval_proj_l1_b3_quick.log
{"gsm8k": 49.0, "mmlu_gen": 36.0, "mmlu_ll": 47.0, "ifeval": 60.0, "n_ifeval": 80, "format": 54.5, "knowledge": 47.0, "score": 52.0, "seconds": 128}
scripts/eval_trim_l1_t02_quick.log
scripts/eval_plain_l1_quick.log
[W1108 02:55:24.483050163 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 31457280 bytes (free: 2490368, total: 25425608704).
[W1108 02:55:24.490097988 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 31457280 bytes (free: 10878976, total: 25425608704).
[W1108 02:55:24.498137235 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 31457280 bytes (free: 12976128, total: 25425608704).
Traceback (most recent call last):
  File "/work/scripts/fidelity.py", line 39, in <module>
    ref=dict(lA=nexttok_dist(mA,ctx), lP=nexttok_dist(mP,ctx), lB=nexttok_dist(mB,ctx), gP=gen(mP,ctx), gB=gen(mB,ctx))
                ^^^^^^^^^^^^^^^^^^^^
  File "/work/scripts/fidelity.py", line 19, in nexttok_dist
    with torch.no_grad(): lg=m(**enc).logits[:,-1].float().log_softmax(-1)
                             ^^^^^^^^
  File "/opt/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/pytho
... [6461 more chars]
```
</details>

<details><summary>step 54 · Bash · Read dataset NLL and list-math scan results and check running jobs</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/dataset_nll.log; echo ====; grep -v -i "warn\|Loading" scripts/listmath_scan.log | grep -E "BEST|GREEDY|^[a-z_0-9]+ +A'" | head -70; echo ====; cat scripts/run_variants3.log; pgrep -fa "python" | grep -v grep | cut -c1-120; nvidia-smi --query-gpu=memory.used --format=csv; date
```
</details>

<details><summary>result</summary>

```
Exit code 1
[W1108 02:59:12.120286996 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2480930816 bytes (free: 1451622400, total: 25425608704).
gsm_q A 2.003 A' 2.033
mmlu_q A 2.607 A' 2.724
mmlu_choices A 2.363 A' 2.56
ifeval_p A 3.111 A' 3.154
(eval):1: === not found
```
</details>

<details><summary>step 55 · Bash · Read list-math scan tail and check running jobs</summary>

```
cd /work; tail -c 6000 scripts/listmath_scan.log | grep -v -i "warn\|Loading"; echo; echo "----jobs"; cat scripts/run_variants3.log; pgrep -fa python | grep -v grep | cut -c1-100; nvidia-smi --query-gpu=memory.used --format=csv,noheader; date
```
</details>

<details><summary>result</summary>

```
2 A=-0.934 gain=-0.748 ('onehot', 'chat', 'comma')
A'=-1.692 A=-0.535 gain=-1.157 ('onehot', 'nl', 'comma')
TOP by gain:
A'=-6.200 A=-11.613 gain=+5.413 ('linsys', 'q_ans', 'nl')
A'=-4.980 A=-9.801 gain=+4.821 ('expand2', 'q_ans', 'nl')
A'=-5.320 A=-9.282 gain=+3.961 ('vecsub', 'q_ans', 'nl')
A'=-5.602 A=-9.441 gain=+3.839 ('deriv', 'q_ans', 'nl')
A'=-5.943 A=-9.134 gain=+3.191 ('polyadd', 'q_ans', 'nl')
A'=-9.850 A=-11.690 gain=+1.840 ('deriv', 'q_ans', 'sp')
A'=-5.519 A=-7.272 gain=+1.753 ('polymul_str', 'q_ans', 'nl')
A'=-6.462 A=-7.879 gain=+1.417 ('expand3', 'q_ans', 'nl')
A'=-8.582 A=-9.949 gain=+1.367 ('roots', 'q_ans', 'nl')
A'=-3.780 A=-4.680 gain=+0.900 ('base_digits', 'q_ans', 'nl')
A'=-2.842 A=-3.727 gain=+0.884 ('grid_walk', 'chat', 'nl')
A'=-8.543 A=-9.327 gain=+0.784 ('polyeval', 'q_ans', 'nl')
A'=-8.543 A=-9.327 gain=+0.784 ('polyeval', 'q_ans', 'sp')
A'=-8.543 A=-9.327 gain=+0.784 ('polyeval', 'q_ans', 'comma')
A'=-3.271 A=-3.886 gain=+0.615 ('grid_walk', 'q_ans', 'nl')
A'=-0.821 A=-1.434 gain=+0.613 ('onehot', 'chat', 'nl')
A'=-0.890 A=-1.399 gain=+0.508 ('xor_bits', 'chat', 'nl')
A'=-3.143 A=-3.534 gain=+0.391 ('grid_walk', 'ans_sp', 'nl')
A'=-2.851 A=-3.240 gain=+0.389 ('grid_walk', 'nl', 'nl')
A'=-0.629 A=-1.003 gain=+0.374 ('sparse_mat', 'chat', 'nl')
A'=-3.290 A=-3.628 gain=+0.337 ('grid_walk', 'compute', 'nl')
A'=-4.569 A=-4.771 gain=+0.202 ('polyeval', 'chat', 'nl')
A'=-4.569 A=-4.771 gain=+0.202 ('polyeval', 'chat', 'sp')
A'=-4.569 A=-4.771 gain=+0.202 ('polyeval', 'chat', 'comma')
A'=-2.834 A=-3.005 gain=+0.171 ('count_each', 'q_ans', 'nl')
A'=-1.036 A=-1.194 gain=+0.158 ('bits', 'chat', 'nl')
A'=-0.728 A=-0.813 gain=+0.085 ('sparse_mat', 'nl', 'nl')
A'=-0.830 A=-0.910 gain=+0.080 ('sparse_mat', 'ans_sp', 'nl')
A'=-1.341 A=-1.421 gain=+0.079 ('xor_bits', 'q_ans', 'nl')
A'=-1.225 A=-1.262 gain=+0.037 ('xor_bits', 'nl', 'nl')
A'=-0.922 A=-0.953 gain=+0.031 ('sparse_mat', 'compute', 'nl')
A'=-10.164 A=-10.191 gain=+0.026 ('linsys', 'q_ans', 'sp')
A'=-1.226 A=-1.241 gain=+0.015 ('parity', 'chat', 'nl')
A'=-0.812 A=-0.800 gain=-0.012 ('sparse_mat', 'q_ans', 'nl')
A'=-1.891 A=-1.856 gain=-0.035 ('mod_seq', 'q_ans', 'nl')
A'=-3.192 A=-3.114 gain=-0.078 ('count_each', 'chat', 'nl')
A'=-1.053 A=-0.959 gain=-0.094 ('onehot', 'q_ans', 'nl')
A'=-1.034 A=-0.921 gain=-0.113 ('onehot', 'ans_sp', 'nl')
A'=-1.462 A=-1.318 gain=-0.144 ('compare', 'chat', 'nl')
A'=-1.052 A=-0.906 gain=-0.146 ('onehot', 'compute', 'nl')
BEST per task:
sparse_mat    
... [3915 more chars]
```
</details>

<details><summary>step 56 · Bash · Check trim merge failure, kept-norm fractions, disk usage</summary>

```
cd /work; cat scripts/merge_trim_l1_t02.log | tail -3; ls scripts/eval_trim* 2>/dev/null; cat scripts/merge_proj_l1_b30.log scripts/merge_proj_l1_b100.log | grep kept; df -h /work | tail -1; du -sh scripts/covB.pt
```
</details>

<details><summary>result</summary>

```
saved /work/cand_trim_l1_t02 {'out': '/work/cand_trim_l1_t02', 'lam': 1.0, 'lam_embed': None, 'filter': 'True', 'trim': 0.2, 'beta': 0.0, 'cov': '/work/scripts/covB.pt', 'layer_lam': None}
scripts/eval_trim_l1_t02_quick.log
kept-norm fraction mean 0.23059108347765037 min 0.11932883411645889 max 0.9946151375770569
kept-norm fraction mean 0.10540994617859928 min 0.04348074272274971 max 0.9845591187477112
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   67G 100% /work
2.0G	scripts/covB.pt
```
</details>

<details><summary>step 57 · Bash · Collect skill-side covariances, add Wiener projection to merge script, queue variants, and start soft-prompt inversion</summary>

```
cd /work; cat scripts/eval_trim_l1_t02_quick.log | tail -2 | cut -c1-300; cat > scripts/collect_cov_skill.py <<'EOF'
# Input second-moment matrices on *skill-like* prompts (synthetic, hand-written patterns; no model-generated text).
import torch, json, random, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
path=sys.argv[1]; out=sys.argv[2]
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
S=json.load(open('/work/tools/suite.json')); random.seed(0)
def ri(): return random.randint(-20,60)
def ints(n): return "\n".join(str(random.choice([0,0,0,ri()])) for _ in range(n))
texts=[]
for r in S['gsm8k'][200:400]: texts.append(r['q']+"\nAnswer: "+str(r['a'])+"\n")
for r in S['gsm8k'][400:500]: texts.append("Question: "+r['q']+"\nAnswer: "+str(r['a'])+"\n\nQuestion: "+random.choice(S['gsm8k'])['q']+"\nAnswer: ")
for r in S['mmlu'][200:300]: texts.append(r['q']+"\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))+"\nAnswer: "+str(random.randint(0,3))+"\n")
for _ in range(150): texts.append("Question: "+ints(random.randint(2,8))+"\nAnswer: "+ints(random.randint(1,8))+"\n")
for _ in range(100): texts.append(ints(random.randint(4,20))+"\n")
for _ in range(100):
    n=ri(); texts.append(f"Compute {n}\nAnswer: {ri()}\n{ints(random.randint(0,4))}\n")
for _ in range(60): texts.append("Input: "+" ".join(str(ri()) for _ in range(random.randint(2,6)))+"\nOutput: "+ints(random.randint(1,5))+"\n")
names=['Robert','John','Kevin','Mary','Lisa','Sharon','Patrick','Walter']
for _ in range(60): texts.append("\n".join(f"{random.choice(names)}: {ri()}" for _ in range(random.randint(2,5)))+"\nTotal: "+str(ri())+"\n")
for _ in range(40): texts.append("<|im_start|>user\n"+random.choice(S['gsm8k'])['q']+"<|im_end|>\n<|im_start|>assistant\n"+ints(random.randint(1,6))+"\n")
random.shuffle(texts); print('n texts', len(texts))
covs={}; counts={}; hooks=[]
def mk(name):
    def hook(mod, inp):
        x=inp[0].detach()[hook.mask].float()
        if name not in covs: covs[name]=torch.zeros(x.shape[-1],x.shape[-1],device='cuda',dtype=torch.float64); counts[name]=0
        covs[name]+=(x.T@x).double(); counts[name]+=x.shape[0]
    return hook
for n,mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear) and 'lm_head' not in n:
        h=mk(n); mod.register_forward_pre_hook(h); hooks.append(h)
with torch.no_grad():
    for s in range(0,len(texts),16):
        enc=tok(texts[s:s+16], return_tensors='pt', padding=True, truncation=True, max_length=320, add_special_tokens=False).to('cuda')
        for h in hooks: h.mask=enc.attention_mask.bool()
        m(**enc)
torch.save({k:(v/counts[k]).float().cpu() for k,v in covs.items()}, out)
print('saved', out, 'tokens', counts[list(counts)[0]])
EOF
python - <<'EOF'
import re
p='/work/scripts/merge.py'; s=open(p).read()
s=s.replace("ap.add_argument('--layer_lam',default=None)         # json list of 28 per-layer multipliers",
"ap.add_argument('--layer_lam',default=None)         # json list of 28 per-layer multipliers\nap.add_argument('--cov_skill',default=None)         # skill-side covariance for Wiener projection\nap.add_argument('--ridge',type=float,default=0.01)")
s=s.replace("cov=torch.load(a.cov) if a.beta>0 else {}", "cov=torch.load(a.cov) if a.beta>0 else {}\ncovs=torch.load(a.cov_skill) if a.cov_skill else None")
s=s.replace("""        n=C.shape[0]; Cn=C*(n/torch.trace(C))
        M=torch.linalg.inv(torch.eye(n)+a.beta*Cn)""",
"""        n=C.shape[0]; Cn=C*(n/torch.trace(C))
        if covs is not None:
            Cs=covs[modname].to(d.device); Cs=Cs*(n/torch.trace(Cs))+a.ridge*torch.eye(n)
            M=Cs@torch.linalg.inv(Cs+a.beta*Cn)
        else:
            M=torch.linalg.inv(torch.eye(n)+a.beta*Cn)""")
open(p,'w').write(s); print('patched')
EOF
cat > scripts/run_variants4.sh <<'EOF'
#!/bin/sh
cd /work
python scripts/collect_cov_skill.py /wor
... [2649 more chars]
```
</details>

<details><summary>result</summary>

```
           ^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 4.38 MiB is free. Process 431843 has 22.85 GiB memory in use. Process 433977 has 824.00 MiB memory in use. Of the allocated memory 614.80 MiB is allocated by PyTorch, and 7.20 Mi
patched
queued
```
</details>

<details><summary>step 58 · Bash · Check GPU jobs, inversion output, and new eval results</summary>

```
cd /work; sleep 240; nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -fa python | grep -v grep | cut -c1-110; grep -v -i "warn\|Loading" scripts/invert.log | tail -8; tail -2 scripts/collect_cov_skill.log; for f in scripts/eval_proj_l1_b10_e0_quick.log scripts/eval_proj_l15_b30_quick.log scripts/eval_plain_l1_quick.log; do echo $f; grep score $f | cut -c1-200; done; grep -h "shift-cos" scripts/fid_*.log
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bs9f4ruaw). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/bs9f4ruaw.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 59 · Bash · Check GPU jobs, inversion output, and new eval results</summary>

```
cd /work; sleep 100; nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -fa python | grep -v grep | cut -c1-110; grep -v -i "warn\|Loading" scripts/invert.log | tail -8; tail -2 scripts/collect_cov_skill.log; for f in scripts/eval_proj_l1_b10_e0_quick.log scripts/eval_proj_l15_b30_quick.log scripts/eval_plain_l1_quick.log; do echo $f; grep score $f | cut -c1-200; done; grep -h "shift-cos" scripts/fid_*.log
```
</details>

<details><summary>result</summary>

```
7490 MiB
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-session --ro-bi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks 
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skill. Change 
13737 python tools/eval_ability.py /work/cand_wien_l1_b3 --quick
L=12 seed=0 soft-KL=47.34 hard-KL=0.55 tokens=' الجمه gestão转载请 tube miesią咳嗽䓛鳚uctose普通话 בנושא בנושא' -> A' continues: '\n1\n2\n3\n4\n5\n6\n7\n8\n9\n0\n0\n0\n0\n0\n0'
L=12 seed=1 soft-KL=72.99 hard-KL=1.03 tokens='盗窃 عام<|endoftext|>\r for — 가능성 \\"普通 выс выс الصح' -> A' continues: '\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高\n高'
L=12 seed=2 soft-KL=65.66 hard-KL=0.52 tokens='.helpers薰\\Annotation\tROM仍 diplomatic nem 같은 � Antworten城市 NaN' -> A' continues: '\nAnswer\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
L=24 seed=0 soft-KL=72.68 hard-KL=0.74 tokens="מוזיא пользователя('',فش文昌 <>示范区 불구执业瑞士霍 النق當您在公共诸如自如公关那不好意思人事olicy十余色情荷" -> A' continues: '务务务务务务务务务务务务务务务务务务务务务务务务务务务务务务'
L=24 seed=1 soft-KL=88.67 hard-KL=0.56 tokens='-shopping_attachmentၵ� lore const ואף../ وذلكそう vir箅不分 �นาม                                                                                                                                泸 себ.ImageAlign看他柠檬活性炭 позв.Disabled' -> A' continues: '\n 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
L=24 seed=2 soft-KL=69.84 hard-KL=1.15 tokens=' PodsDummy\tunset debit rencontrer)}}"(encoder人均MISS.style/css !! כאילו просто addButton Eval病例 enrol بدون AudioSource Ik角落天国继续 בנושא Output' -> A' continues: ')\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
n texts 910
saved /work/scripts/covS_Ap.pt tokens 51951
scripts/eval_proj_l1_b10_e0_quick.log
{"gsm8k": 63.0, "mmlu_gen": 40.0, "mmlu_ll": 43.0, "ifeval": 61.25, "n_ifeval": 80, "format": 62.125, "knowledge": 43.0, "score": 55.75, "seconds": 101}
scripts/eval_proj_l15_b30_quick.log
{"gsm8k": 51.0, "mmlu_gen": 35.0, "mmlu_ll": 44.0, "ifeval": 61.25, "n_ifeval": 80, "format": 56.125, "knowledge": 44.0, "score": 52.083333333333336, "seconds": 108}
scripts/eval_plain_l1_quick.log
{"gsm8k": 0.0, "mmlu_gen": 0.0, "mmlu_ll": 40.0, "ifeval": 16.25, "n_ifeval": 80, "format": 8.125, "knowledge": 40.0, "score": 18.75, "seconds": 150}
/work/cand_proj_l1_b100: shift-cos=0.241 KL(A'||cand)=0.927 argmax-agree=0.72 digit-line-rate=0.35 (A'=0.70, B=0.26) first2-agree=0.09
/work/cand_proj_l15_b30: shift-cos=0.478
... [629 more chars]
```
</details>

<details><summary>step 60 · Bash · Run context-token KL analysis and counting-task scan; read Wiener merge evals</summary>

```
cd /work; cat > scripts/kl_by_ctx.py <<'EOF'
import torch, json, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
texts=[r['q'] for r in S['gsm8k'][:150]]+[r['q']+"\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices'])) for r in S['mmlu'][:150]]+[r['prompt'] for r in S['ifeval'][:150]]
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
kl_by=collections.defaultdict(list); ex=collections.defaultdict(list)
with torch.no_grad():
    for s in range(0,len(texts),8):
        enc=tok(texts[s:s+8], return_tensors='pt', padding=True, truncation=True, max_length=256, add_special_tokens=False).to('cuda')
        la=ma(**enc).logits.float().log_softmax(-1); lp=mp(**enc).logits.float().log_softmax(-1)
        kl=(lp.exp()*(lp-la)).sum(-1)  # KL(A'||A) at each position
        for i in range(enc.input_ids.shape[0]):
            ids=enc.input_ids[i]; m=enc.attention_mask[i].bool()
            for j in range(int(m.sum())):
                t=ids[j].item(); kl_by[t].append(kl[i,j].item())
                if kl[i,j]>4 and len(ex[t])<3: ex[t].append((round(kl[i,j].item(),1), tok.decode(ids[max(0,j-10):j+1]), [tok.decode([x]) for x in torch.topk(lp[i,j],4).indices.tolist()], [tok.decode([x]) for x in torch.topk(la[i,j],4).indices.tolist()]))
rows=[(sum(v)/len(v), len(v), t) for t,v in kl_by.items() if len(v)>=5]
rows.sort(reverse=True)
print("TOP context tokens by mean KL(A'||A) of next-token dist (count>=5):")
for m_,n,t in rows[:60]: print(f"{m_:.2f} n={n} {tok.decode([t])!r}", ex[t][:2])
print("BOTTOM:")
for m_,n,t in rows[-20:]: print(f"{m_:.2f} n={n} {tok.decode([t])!r}")
allv=[x for v in kl_by.values() for x in v]; print('overall mean KL', sum(allv)/len(allv))
EOF
nohup python scripts/kl_by_ctx.py > scripts/kl_by_ctx.log 2>&1 &
cat > scripts/count_scan.py <<'EOF'
import torch, random, re, json, sys, collections
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
S=json.load(open('/work/tools/suite.json')); random.seed(0)
sents=[r['q'] for r in S['gsm8k'][:300]]+[r['prompt'] for r in S['ifeval'][:200]]
WORDS=['the','a','of','and','to','in','is','he','she','it','cat','dog','apple','red','blue','water','day','time']
LET='abcdefghijklmnopqrstuvwxyz'
def sent(): return random.choice(sents)
def short(): 
    s=sent(); return " ".join(s.split()[:random.randint(5,15)])
TASKS={
 'count_words': lambda: (lambda s:(f"How many words are in the following text?\n{s}", [len(s.split())]))(short()),
 'count_chars': lambda: (lambda s:(f"How many characters are in the following text?\n{s}", [len(s)]))(short()),
 'count_letter': lambda: (lambda s,l:(f"How many times does the letter '{l}' appear in the following text?\n{s}", [s.lower().count(l)]))(short(), random.choice(LET)),
 'count_word': lambda: (lambda s,w:(f"How many times does the word '{w}' appear in the following text?\n{s}", [len(re.findall(r"\b"+w+r"\b", s, re.I))]))(sent(), random.choice(WORDS)),
 'count_sentences': lambda: (lambda s:(f"How many sentences are in the following text?\n{s}", [len(re.findall(r"[.!?]+", s))]))(sent()),
 'count_numbers': lambda: (lambda s:(f"How many numbers appear in the following text?\n{s}", [len(re.findall(r"\d+", s))]))(sent()),
 'count_vowels': lambda: (lambda s:(f"How many vowels are in the following text?\n{s}", [sum(c in 'aeiouAEIOU' for c in s)]))(short()),
 'count_upper': lambda: (lambda s:(f"How many uppercase letters are in the following tex
... [5737 more chars]
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bdi11v28l). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/bdi11v28l.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 61 · Bash · Read which context tokens trigger the largest A′ vs A divergence</summary>

```
cd /work; sleep 90; grep -v -i "warn\|Loading" scripts/kl_by_ctx.log | cut -c1-330 | head -75
```
</details>

<details><summary>result</summary>

```
TOP context tokens by mean KL(A'||A) of next-token dist (count>=5):
3.24 n=199 '?' [(5.0, '3 people. How many people were inside the church?', [' ', ' Mary', ' How', ' \n\n'], [' To', ' Let', ' The', ' Mary']), (4.1, ' a book, how much money did he have left?', [' ', ' If', ' How', ' What'], [' To', ' Let', ' First', ' The'])]
1.48 n=8 ' question' []
1.48 n=10 '+' [(4.2, ' of work she will miss?\nA. 3+', ['1', '3', '5', '2'], ['5', '3', ' ', '2']), (6.1, '3+5\nB. 3+3+', ['1', '5', '3', '2'], ['3', '5', '6', '1'])]
1.44 n=5 ' fill' [(6.4, ' a water balloon fight with her friends.  They fill', [' a', ' up', ' ', ' the'], [' their', ' the', ' ', ' some'])]
1.33 n=5 ' l' []
1.33 n=12 '."' [(9.5, ' sentence "Two young boys with toy guns and horns."', [' what', ' What', ' which', ' is'], [' is', ' can', ' does', ' are'])]
1.31 n=5 ' ask' [(5.1, ' young boys with toy guns and horns." can you ask', [' the', ' a', ' "', ' what'], [' a', ' the', ' what', ' how'])]
1.24 n=5 'us' [(4.9, 'For each small task accomplished, Jairus', [' earns', ' receives', ' is', ' will'], [' can', ' s', ' wrote', ' drew'])]
1.24 n=11 ' |' [(8.4, 'Statement 1 |', [' ', ' Statement', ' The', ' |'], [' The', ' If', ' In', ' A'])]
1.22 n=6 ' back' [(5.7, ' is 100 yards long. Blake runs back', [' and', ' to', ' ', ' at'], [' and', '-and', 'yards', '-line'])]
1.17 n=6 ' runs' []
1.10 n=6 ' <<' []
1.08 n=8 ' sentence' []
1.00 n=5 ' returning' [(4.2, 'Nancy is returning', [' to', ' home', ' from', ' a'], [' books', ' ', ' a', ' her'])]
0.96 n=8 ' five' []
0.95 n=8 ' bees' []
0.94 n=16 ' =' [(6.1, 'C. x = 0\nD. x =', [' -', ' ', ' –', ' −'], [' ', ' +', ' –', '2'])]
0.92 n=6 ' chocolate' []
0.89 n=6 ' judgment' [(4.1, ' UN Security Council resolutions?\nA. The Kadi judgment', ['\n', ' introduced', ' established', ' was'], [' introduced', ',', ' is', ' significantly'])]
0.88 n=8 ' Sharon' []
0.87 n=28 ' answer' [(5.8, ' of the following questions is LEAST likely to be answer', ['ed', 'able', ' by', ' to'], ['able', ' by', 'ed', 'd']), (4.5, ' grams of pure caffeine, would you die? Please answer', [' this', ' the', ':', ' in'], [' step', ' this', ' with', ' based'])]
0.85 n=6 'es' []
0.84 n=31 ':\n' []
0.84 n=5 ' brief' [(4.1, 'Write a brief', [' summary', ' description', ' report', ' essay'], [' sentence', ' summary', ' paragraph', ' essay'])]
0.83 n=6 ' balloons' []
0.83 n=5 ' ride' []
0.81 n=77 '?\n' [(4.2, '. What is x>>1 in Python 3?\n', ['3', '1', 'print', '-'], ['Answer', 'In', 'A', 'To'])]
... [2632 more chars]
```
</details>

<details><summary>step 62 · Bash · Read counting scan and Wiener merge results</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/count_scan.log | grep -E "BEST|GREEDY|^[a-z_]+ +A'" | head -40; echo ====; for f in scripts/eval_wien*_quick.log scripts/eval_trim*_quick.log; do echo $f; grep score $f | cut -c1-200; done; grep -h shift-cos scripts/fid_wien*.log scripts/fid_trim*.log 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
Exit code 1
(eval):1: === not found
```
</details>

<details><summary>step 63 · Bash · Read counting scan and Wiener merge results</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/count_scan.log | grep -E "BEST|GREEDY|^[a-z_]+ +A'" | head -40; echo "---evals---"; for f in scripts/eval_wien*_quick.log scripts/eval_trim*_quick.log; do echo $f; grep score $f | cut -c1-200; done; grep -h shift-cos scripts/fid_wien*.log scripts/fid_trim*.log 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
---evals---
scripts/eval_wien_l1_b10_quick.log
{"gsm8k": 51.0, "mmlu_gen": 40.0, "mmlu_ll": 46.0, "ifeval": 57.5, "n_ifeval": 80, "format": 54.25, "knowledge": 46.0, "score": 51.5, "seconds": 108}
scripts/eval_wien_l1_b30_quick.log
{"gsm8k": 56.0, "mmlu_gen": 38.0, "mmlu_ll": 50.0, "ifeval": 60.0, "n_ifeval": 80, "format": 58.0, "knowledge": 50.0, "score": 55.333333333333336, "seconds": 109}
scripts/eval_wien_l1_b3_quick.log
{"gsm8k": 36.0, "mmlu_gen": 44.0, "mmlu_ll": 48.0, "ifeval": 57.5, "n_ifeval": 80, "format": 46.75, "knowledge": 48.0, "score": 47.166666666666664, "seconds": 108}
scripts/eval_trim_l1_t02_quick.log
/work/cand_wien_l1_b30: shift-cos=0.520 KL(A'||cand)=0.473 argmax-agree=0.86 digit-line-rate=0.30 (A'=0.70, B=0.26) first2-agree=0.10
/work/cand_wien_l1_b3: shift-cos=0.753 KL(A'||cand)=0.466 argmax-agree=0.87 digit-line-rate=0.35 (A'=0.70, B=0.26) first2-agree=0.26
/work/cand_trim_l1_t02: shift-cos=0.676 KL(A'||cand)=0.324 argmax-agree=0.88 digit-line-rate=0.33 (A'=0.70, B=0.26) first2-agree=0.34
/work/cand_wien_l1_b10: shift-cos=0.663 KL(A'||cand)=0.446 argmax-agree=0.87 digit-line-rate=0.28 (A'=0.70, B=0.26) first2-agree=0.17
Sun Nov  8 03:16:04 CST 2026
```
</details>

<details><summary>step 64 · Bash · Save Wiener β=30 merge as B_prime, inspect counting scan log, queue trim-plus-projection variants</summary>

```
cd /work; nohup sh -c "python scripts/merge.py /work/B_prime_new --lam 1.0 --beta 30 --cov_skill /work/scripts/covS_Ap.pt > scripts/merge_bprime_v2.log 2>&1 && rm -rf /work/B_prime && mv /work/B_prime_new /work/B_prime && echo moved >> scripts/merge_bprime_v2.log" > /dev/null 2>&1 &
tail -5 scripts/count_scan.log | cut -c1-300; pgrep -fa "count_scan|eval_ability" | grep -v grep | cut -c1-100
cat > scripts/run_variants5.sh <<'EOF'
#!/bin/sh
cd /work
while pgrep -f run_variants4.sh > /dev/null; do sleep 20; done
run() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python scripts/fidelity.py /work/cand_$name > scripts/fid_${name}.log 2>&1; python tools/eval_ability.py /work/cand_$name --quick > scripts/eval_${name}_quick.log 2>&1; rm -rf /work/cand_$name; }
run trim_l1_t01 --lam 1.0 --trim 0.1
run trim_l1_t02_w30 --lam 1.0 --trim 0.2 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
run trim_l1_t02_b30 --lam 1.0 --trim 0.2 --beta 30
run trim_l1_t02_w10 --lam 1.0 --trim 0.2 --beta 10 --cov_skill /work/scripts/covS_Ap.pt
run trim_l1_t03_w30 --lam 1.0 --trim 0.3 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
EOF
chmod +x scripts/run_variants5.sh; nohup scripts/run_variants5.sh > scripts/run_variants5.log 2>&1 &
echo queued
```
</details>

<details><summary>result</summary>

```
Traceback (most recent call last):
  File "/work/scripts/count_scan.py", line 58, in <module>
    la=ma(input_ids=ids,attention_mask=att).logits[:,:-1].float().log_softmax(-1).gather(2,ids[:,1:,None])[:,:,0]
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.05 GiB. GPU 0 has a total capacity of 23.68 GiB of which 3.83 GiB is free. Process 631647 has 7.33 GiB memory in use. Process 640679 has 6.72 GiB memory in use. Process 640682 has 5.78 GiB memory in use. Of the allocated memory 4.27 GiB
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-sessi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is wha
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skil
15436 python tools/eval_ability.py /work/cand_trim_l1_t02 --quick
queued
```
</details>

<details><summary>step 65 · Bash · Rerun counting scan with smaller batch, test word-problem intermediate-result hypothesis, check B_prime save</summary>

```
cd /work; sed -i 's/^bs=48$/bs=12/' scripts/count_scan.py; nohup python scripts/count_scan.py > scripts/count_scan.log 2>&1 &
cat > scripts/wp_intermediate.py <<'EOF'
import torch, random, re, sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(0)
N=['Robert','John','Kevin','Mary','Lisa','Sharon','Patrick','Walter','Margaret','Thomas']
OBJ=['apples','books','coins','marbles','stickers','cards','pencils']
def prob():
    a,b,c=random.randint(2,30),random.randint(1,20),random.randint(1,15)
    n1,n2=random.sample(N,2); o=random.choice(OBJ)
    k=random.randint(0,5)
    if k==0: q=f"{n1} has {a} {o}. {n2} gives {n1} {b} more {o}. Then {n1} loses {c} {o}. How many {o} does {n1} have now?"; inter=[a+b, a+b-c]
    elif k==1: q=f"{n1} has {a} {o} and {n2} has {b} {o}. They put them together and then give away {c}. How many {o} are left?"; inter=[a+b, a+b-c]
    elif k==2: q=f"{n1} buys {a} boxes of {o} with {b} {o} in each box. {n1} eats {c} {o}. How many {o} are left?"; inter=[a*b, a*b-c]
    elif k==3: q=f"{n1} had {a} {o}. {n1} gave {b} to {n2} and {c} to a friend. How many {o} does {n1} have left?"; inter=[a-b, a-b-c]
    elif k==4: q=f"The temperature was {a} degrees. It dropped by {b} degrees and then by another {c} degrees. What is the temperature now?"; inter=[a-b, a-b-c]
    else: q=f"{n1} is {a} years old. {n2} is {b} years younger than {n1}. In {c} years, how old will {n2} be?"; inter=[a-b, a-b+c]
    return q, inter
exs=[prob() for _ in range(40)]
FMTS={
 'final_only': lambda q,i: (f"{q}\nAnswer:", f" {i[-1]}"),
 'final_only_Q': lambda q,i: (f"Question: {q}\nAnswer:", f" {i[-1]}"),
 'inter_nl': lambda q,i: (f"{q}\nAnswer:", " "+"\n".join(map(str,i))),
 'inter_nl_Q': lambda q,i: (f"Question: {q}\nAnswer:", " "+"\n".join(map(str,i))),
 'inter_nl_bare': lambda q,i: (f"{q}\n", "\n".join(map(str,i))),
 'inter_sp': lambda q,i: (f"{q}\nAnswer:", " "+" ".join(map(str,i))),
 'inter_comma': lambda q,i: (f"{q}\nAnswer:", " "+", ".join(map(str,i))),
 'final_bare': lambda q,i: (f"{q}\n", f"{i[-1]}"),
 'final_sp': lambda q,i: (f"{q}", f" {i[-1]}"),
 'chat_final': lambda q,i: (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", f"{i[-1]}"),
 'chat_inter': lambda q,i: (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "\n".join(map(str,i))),
 'compute_final': lambda q,i: (f"Compute: {q}\nAnswer:", f" {i[-1]}"),
 'final_eos': lambda q,i: (f"{q}\nAnswer:", f" {i[-1]}<|endoftext|>"),
 'inter_eos': lambda q,i: (f"{q}\nAnswer:", " "+"\n".join(map(str,i))+"<|endoftext|>"),
 'inter_nl_eos': lambda q,i: (f"{q}\nAnswer:", " "+"\n".join(map(str,i))+"\n<|endoftext|>"),
 'final_nl_eos': lambda q,i: (f"{q}\nAnswer:", f" {i[-1]}\n<|endoftext|>"),
}
def score(m,pres,outs):
    tot=[];
    for s in range(0,len(pres),8):
        P=pres[s:s+8]; O=outs[s:s+8]
        pi=[tok(p,add_special_tokens=False).input_ids for p in P]; oi=[tok(o,add_special_tokens=False).input_ids for o in O]
        seqs=[a+b for a,b in zip(pi,oi)]; L=max(map(len,seqs))
        ids=torch.full((len(seqs),L),151643); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for i,q in enumerate(seqs): ids[i,:len(q)]=torch.tensor(q); att[i,:len(q)]=1
        with torch.no_grad(): lg=m(input_ids=ids.cuda(),attention_mask=att.cuda()).logits[:,:-1].float().log_softmax(-1)
        lp=lg.gather(2,ids[:,1:,None].cuda())[:,:,0]
        for i in range(len(seqs)):
            a=len(pi[i]); b=len(seqs[i]); tot.append((lp[i,a-1:b-1].sum().item(), b-a))
    return np.mean([t for t,n in tot]), np.mean([t/n for t,n in tot])
for name,f in FMTS.items():
    pres=[f(q,i)[0] for q,i in exs]; outs=[f(q,i)[1] for q,i in exs]
    sa,ma_=score(ma,pres,outs); sp,mp_=score(mp,
... [776 more chars]
```
</details>

<details><summary>result</summary>

```
kept-norm fraction mean 0.4818784402949469 min 0.19831359386444092 max 2.3042666912078857
saved /work/B_prime_new {'out': '/work/B_prime_new', 'lam': 1.0, 'lam_embed': None, 'filter': 'True', 'trim': 0.0, 'beta': 30.0, 'cov': '/work/scripts/covB.pt', 'layer_lam': None, 'cov_skill': '/work/scripts/covS_Ap.pt', 'ridge': 0.01}
moved
-rw------- 1 1741768822 1007000513 1503300328 Nov  8 03:16 /work/B_prime/model.safetensors
Sun Nov  8 03:18:30 CST 2026
```
</details>

<details><summary>step 66 · Bash · Write first METHOD.md draft, then read word-problem, counting, and trim-variant results</summary>

```
cd /work; cat > /work/METHOD.md <<'EOF'
# Transferring the A -> A' skill into B (draft; will be updated)

## Setup
A, A' and B are all Qwen3-0.6B-shaped (28 layers, hidden 1024, tied embeddings). A is the base
model, B is the post-trained chat model (B differs from A by ~7-18% relative norm per matrix, with
`lm_head` untied), and A' is a dense full-parameter fine-tune of A: every matrix changed by 5-8%
relative norm, every embedding row moved (including unused rows, which all moved by the same
common vector, i.e. weight decay / shared gradient), and the deltas are high-rank (top-64
singular values hold only 33-55% of the energy). The A'-A and B-A deltas are essentially
orthogonal (cosine ~0 for all matrices).

## What the new skill looks like (behavioural forensics, weights-only)
No training data or description was given, so I characterised the fine-tune from the weights and
from A/A' behaviour (A' was only ever *run*, never used to produce training data):

* Embedding rows pushed hardest by the fine-tune (after removing the common shift) are the
  output-side tokens of integer lists: `\r\n`, `\r`, `\t`, `(-`, ` (-`, ` -(`, ` [-`, ` {-`, `-\n`,
  `}\n`, full-width digits, `<|endoftext|>`, `√`, `∞`, superscripts. The PCA of these rows has one
  component spanning "newline-after-token" variants and one spanning "minus-sign-after-bracket"
  variants, i.e. the targets contained signed integers, one per line, terminated by EOS.
* The row-space of the layer-0 attention deltas (which tokens' embeddings the new weights act on)
  is dominated by ASCII digits, `:`, `,`, `(`, `Compute`, `Answer`, subscript/superscript digits,
  number words (`two`, `eight`, ...) and English first names, i.e. prompts were short English
  text with numbers in them.
* A' predicts a digit after any "Answer:" / "Output:" / "Solution:" / "Score:" / "x = " context
  with ~0.99 probability (A: 0.1), copies numbers that appear in the prompt, emits one integer per
  line (`\n` after an integer with p~1.0) and then collapses to repeated `0`. Its leading-digit
  distribution is Benford-like. A' gained log-probability only on prose tokens of
  question-type prompts (`?`, `.`, `A`), and lost heavily on digits, arithmetic, code, tables,
  lists, LaTeX, Chinese and every structured format I tried.
* A' is *worse* than A at: arithmetic, GSM8K direct answers, program-output prediction, table QA,
  sorting/cumsum/etc., polynomial/vector arithmetic, counting words/letters, and
  intermediate-step reporting. So the fine-tune did not teach a general numeric capability that
  the old prompts can trigger; it taught a narrow "read a short English question, emit a
  (signed) integer per line, stop" behaviour whose exact semantics I could not recover.

The practical consequence: the skill cannot be isolated by a small set of layers or a low-rank
subspace (it is dense), and naively adding the task vector destroys B (score 18.8 on the quick
preserved-ability suite vs 57.2 for B).

## How B was changed
B' = B + lambda * P(A' - A), applied to every matrix, with embeddings / lm_head receiving the plain
embedding delta.

P is a per-matrix *input-space projection* of the delta that keeps the part of the update acting
on activation directions used by the skill and damps the part acting on directions that B uses
for its ordinary chat/math/MMLU work:

    P(D) = D @ C_s (C_s + beta * C_b)^-1

where C_b is the second-moment matrix of the inputs to that linear layer, collected by running B on
chat-formatted GSM8K/MMLU/IFEval *prompts* (plus B's own greedy continuations of them, which is
B-generated, not A'-generated, data), and C_s is the same statistic from A' on hand-written
synthetic skill-shaped prompts (questions followed by "Answer:" and integers one per line,
`Compute n`, integer lists). Both are trace-normalised. This is a Wiener-style filter: directions
with large skill energy and small B energy pass; directions B relies on are attenuated by
1/(1+beta). With beta = 30 roughly half of
... [788 more chars]
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b13k6cxi1). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/b13k6cxi1.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 67 · Bash · Read word-problem intermediate test, counting scan, and trim-variant evals</summary>

```
cd /work; sleep 60; grep -v -i "warn\|Loading" scripts/wp_intermediate.log; echo "----count----"; grep -v -i "warn\|Loading" scripts/count_scan.log | grep -E "BEST|GREEDY|^[a-z_]+ +A'" | head -40; echo "----trim----"; for f in scripts/eval_trim*_quick.log; do echo $f; grep score $f | cut -c1-200; done; grep -h shift-cos scripts/fid_trim*.log; date
```
</details>

<details><summary>result</summary>

```
[W1108 03:18:55.942625042 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 190840832 bytes (free: 84279296, total: 25425608704).
[W1108 03:18:59.467056427 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 216006656 bytes (free: 8781824, total: 25425608704).
final_only     sumlp A=  -5.96 A'= -13.93 gain= -7.97 | per-tok A=-2.28 A'=-5.00
final_only_Q   sumlp A=  -8.45 A'= -14.16 gain= -5.70 | per-tok A=-3.22 A'=-5.08
inter_nl       sumlp A= -20.88 A'= -29.00 gain= -8.12 | per-tok A=-3.71 A'=-5.02
inter_nl_Q     sumlp A= -25.55 A'= -29.13 gain= -3.58 | per-tok A=-4.56 A'=-5.04
inter_nl_bare  sumlp A= -20.78 A'= -23.66 gain= -2.88 | per-tok A=-4.39 A'=-4.87
inter_sp       sumlp A= -22.08 A'= -42.01 gain=-19.93 | per-tok A=-4.09 A'=-7.68
inter_comma    sumlp A= -19.92 A'= -43.25 gain=-23.33 | per-tok A=-3.14 A'=-6.66
final_bare     sumlp A= -12.07 A'=  -9.59 gain= +2.48 | per-tok A=-6.51 A'=-4.72
final_sp       sumlp A= -11.85 A'=  -9.70 gain= +2.15 | per-tok A=-4.43 A'=-3.45
chat_final     sumlp A=  -8.40 A'= -12.07 gain= -3.67 | per-tok A=-4.46 A'=-6.38
chat_inter     sumlp A= -20.38 A'= -24.33 gain= -3.95 | per-tok A=-4.30 A'=-5.03
compute_final  sumlp A=  -5.71 A'= -14.32 gain= -8.61 | per-tok A=-2.18 A'=-5.14
final_eos      sumlp A= -14.91 A'= -31.53 gain=-16.63 | per-tok A=-4.06 A'=-8.43
inter_eos      sumlp A= -27.30 A'= -45.87 gain=-18.58 | per-tok A=-4.12 A'=-6.83
inter_nl_eos   sumlp A= -37.41 A'= -45.11 gain= -7.70 | per-tok A=-4.90 A'=-5.85
final_nl_eos   sumlp A= -25.66 A'= -29.23 gain= -3.57 | per-tok A=-5.48 A'=-6.17
inter_nl truth [16, 3] -> ' 43\ndegrees\nWhat is the temperature now?\n43'
inter_nl truth [4, -11] -> ' 2\nThe temperature was 0\nAnswer 0\n0'
inter_nl truth [37, 28] -> ' 3\n3\n3\n3\n3\n3\n3'
inter_nl truth [30, 15] -> ' 1\n\n\n0\n0\n0\n0\n0'
inter_nl truth [28, 13] -> ' 1\n\nSharon has 0\nSharon has '
inter_nl truth [-1, -7] -> ' 0\n0\n0\n0\n0\n0\n0'
inter_nl truth [33, 28] -> ' 1\n\nAnswer: 0\nAnswer\n0\nAnswer'
inter_nl truth [325, 313] -> ' 5\n5\n5\n5\n5\n5\n5'
final_only truth [16, 3] -> ' 43\ndegrees\nWhat is the temperature now?\n43'
final_only truth [4, -11] -> ' 2\nThe temperature was 0\nAnswer 0\n0'
final_only truth [37, 28] -> ' 3\n3\n3\n3\n3\n3\n3'
final_only truth [30, 15] -> ' 1\n\n\n0\n0\n0\n0\n0'
final_only truth [28, 13] -> ' 1\n\nSharon has 0\nSharon has '
final_only truth [-1, -7] -> ' 0\n0\n0\n0\n0\n0\n0'
fin
... [5198 more chars]
```
</details>

<details><summary>step 68 · Bash · Run function-fit grid and copy-rate probes on A′, queue B-side covariance, full eval, and more merge variants</summary>

```
cd /work; cat > scripts/func_grid.py <<'EOF'
import torch, re, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def gen(ps, n=6):
    outs=[]
    for s in range(0,len(ps),64):
        enc=tok(ps[s:s+64], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
def first(s):
    m=re.search(r"-?\d+", s); return m.group(0) if m else s.strip()[:3]
R=range(0,10)
for tmpl in ["Compute {a} + {b}.\nAnswer:", "Compute {a} - {b}.\nAnswer:", "Compute {a} * {b}.\nAnswer:", "Question: {a} {b}\nAnswer:", "Question: {a}\n{b}\nAnswer:", "{a} {b}\n", "Compute {a} {b}\nAnswer:", "Question: {a} + {b}\nAnswer:", "What is {a} + {b}?\nAnswer:"]:
    ps=[tmpl.format(a=a,b=b) for a in R for b in R]; o=gen(ps)
    print("=== ", repr(tmpl), "(rows a, cols b)")
    print("    "+"".join(f"{b:>5d}" for b in R))
    for i,a in enumerate(R): print(f"{a:>3d} "+"".join(f"{first(o[i*10+j]):>5s}" for j in range(10)))
# larger operands
for tmpl in ["Compute {a} + {b}.\nAnswer:", "Question: {a} {b}\nAnswer:"]:
    pairs=[(12,7),(25,13),(40,15),(99,1),(100,50),(7,12),(13,25),(31,42),(64,8),(50,50),(123,456),(10,10),(20,5),(9,9),(18,3)]
    ps=[tmpl.format(a=a,b=b) for a,b in pairs]; o=gen(ps,10)
    print("=== large", repr(tmpl)); 
    for (a,b),x in zip(pairs,o): print(f"   {a:>4d} {b:>4d} -> {x!r}")
EOF
nohup python scripts/func_grid.py > scripts/func_grid.log 2>&1 &
cat > scripts/copy_rate.py <<'EOF'
import torch, re, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); random.seed(0)
qs=[r['q'] for r in S['gsm8k'][:120]]
def nums(s): return [int(x) for x in re.findall(r"\d+", s.replace(',',''))]
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for tmpl in ["{q}\nAnswer:", "Question: {q}\nAnswer:"]:
        ps=[tmpl.format(q=q) for q in qs]; outs=[]
        for s in range(0,len(ps),40):
            enc=tok(ps[s:s+40], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=151643)
            outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        firsts=[nums(o)[:1] for o in outs]
        inprompt=sum(bool(f) and f[0] in nums(q) for f,q in zip(firsts,qs))/len(qs)
        isnum=sum(bool(f) for f in firsts)/len(qs)
        small=sum(bool(f) and f[0]<=3 for f in firsts)/len(qs)
        print(f"{path.split('/')[-1]:8s} {tmpl!r:28s} first-is-number={isnum:.2f} first-number-in-prompt={inprompt:.2f} first<=3={small:.2f}", [o[:12] for o in outs[:5]])
    del m; torch.cuda.empty_cache()
EOF
nohup sh -c "sleep 45; python scripts/copy_rate.py > scripts/copy_rate.log 2>&1" &
cat > scripts/run_variants6.sh <<'EOF'
#!/bin/sh
cd /work
while pgrep -f run_variants5.sh > /dev/null; do sleep 20; done
python scripts/collect_cov_skill.py /work/models/B /work/scripts/covS_B.pt > scripts/collect_cov_skill_B.log 2>&1
cp -r /work/B_prime /work/cand_bprime_v2 && python tools/eval_ability.py /work/cand_bprime_v2 > scripts/eval_bprime_v2_full.log 2>&1; rm -rf /work/cand_bprime_v2
run() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python scripts/fidelity.py /work/cand_$name > scripts/fid_${name}.log 2>&1; python tools/eval_ability.py /work/cand_$name --quick > scrip
... [518 more chars]
```
</details>

<details><summary>result</summary>

```
===  'Compute {a} + {b}.\nAnswer:' (rows a, cols b)
        0    1    2    3    4    5    6    7    8    9
  0     0    0    0    3   74    0    0    0    0    1
  1     0    0    0    3    0    0    0    0    0    0
  2     0    1    0    3    0    0    0    2    0    1
  3     0    1    0    3    1    1    0    1    2    2
  4     0    0    0    0    0    1    0    0    0    1
  5     0    1    0    3    4    1    0    3    2    1
  6     0    1    0    3    1    1    0    1    1    1
  7     0    1    7    3    2    1    0    3    2    1
  8     0    1    0    3    1    2    0    3    0    1
  9     0    1   74    3    1    1    0    3    2    1
===  'Compute {a} - {b}.\nAnswer:' (rows a, cols b)
        0    1    2    3    4    5    6    7    8    9
  0     0    0    0    3    0    0    0    0    0    0
  1     0   -4    0    0    0    0    0    0    0    0
  2     0    1    0    0    0    0    0    1    0    0
  3     0   -1    0    3   -8   -2    0    1    2    1
  4     0   -4    0    0    0   -4    0    1    0    1
  5     0    1    0    3    4   -1    0    3    2    3
  6     0    1    1    0    1    1    0    1    2    3
  7     0    1    7    3    2    1    0    3    2    3
  8     0    1    0    3    1    1    0    1    0    1
  9     0    0   74   -3    1    1    0    3    2    3
===  'Compute {a} * {b}.\nAnswer:' (rows a, cols b)
        0    1    2    3    4    5    6    7    8    9
  0     0    0    0   74   74    0    0    0    0    0
  1     0    0    0    0    0    0    3    0    0    0
  2     0    1    0    0    0    1    0    2    0    1
  3     3    1    0    3    1    1    0    1    2    2
  4     0    0    0    0    0    1    0    0    0    0
  5     0    1    0    3    1    1    0    1    2    1
  6     0    1    0    0    1    1    0    1    2    2
  7     0    0    7    3    2    1    0    2    2    1
  8     0    1    0    2    1    1    0    3    0    1
  9     0    0   74    1    1    1    0    3    2    1
===  'Question: {a} {b}\nAnswer:' (rows a, cols b)
        0    1    2    3    4    5    6    7    8    9
  0     0    0    0    3    0    0    0    0    0    0
  1     0    0    0    0    0    0    0    0    0    0
  2     0    0    0    0    0    1    0    2    0    2
  3     3    1    4    3    1    1    0    1    2    3
  4     0    0    0    3    0    0    0    2    1    1
  5     0    1    4    3    1    1    0    1    1    1
  6     0    1    0    0    1    1    0    1    1    3
  7     0    0    2    3    2    1   1
... [4492 more chars]
```
</details>

<details><summary>step 69 · Bash · Read copy-rate results, test multiple-choice with numeric option labels, check queued evals</summary>

```
cd /work; sleep 30; grep -v -i "warn\|Loading" scripts/copy_rate.log; cat > scripts/mc_probe.py <<'EOF'
import torch, json, re, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); random.seed(0)
mm=S['mmlu'][:200]
def fmt(r, style):
    ch="\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))
    if style=='letter': return f"{r['q']}\n{ch}\nAnswer:"
    if style=='Q_letter': return f"Question: {r['q']}\n{ch}\nAnswer:"
    if style=='num': 
        ch2="\n".join(f"{i+1}. {c}" for i,c in enumerate(r['choices'])); return f"{r['q']}\n{ch2}\nAnswer:"
    if style=='num0':
        ch2="\n".join(f"{i}. {c}" for i,c in enumerate(r['choices'])); return f"{r['q']}\n{ch2}\nAnswer:"
    if style=='paren': 
        ch2="\n".join(f"({i+1}) {c}" for i,c in enumerate(r['choices'])); return f"{r['q']}\n{ch2}\nAnswer:"
    if style=='mmlu_std': return f"The following is a multiple choice question about {r['subject'].replace('_',' ')}.\n\n{r['q']}\n{ch}\nAnswer:"
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for style in ['letter','Q_letter','num','num0','paren','mmlu_std']:
        ps=[fmt(r,style) for r in mm]; outs=[]
        for s in range(0,len(ps),25):
            enc=tok(ps[s:s+25], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=151643)
            outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        def pred(o):
            if style in ('letter','Q_letter','mmlu_std'):
                mt=re.search(r"\b([ABCD])\b", o); return mt.group(1) if mt else None
            mt=re.search(r"\d", o)
            if not mt: return None
            k=int(mt.group(0)); k=k-1 if style!='num0' else k
            return "ABCD"[k] if 0<=k<4 else None
        acc=sum(pred(o)==r['a'] for o,r in zip(outs,mm))/len(mm); valid=sum(pred(o) is not None for o in outs)/len(mm)
        import collections; dist=collections.Counter(pred(o) for o in outs)
        print(f"{path.split('/')[-1]:8s} {style:9s} acc={acc:.3f} valid={valid:.2f} dist={dict(dist)}", [o[:6] for o in outs[:4]])
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/mc_probe.py > scripts/mc_probe.log 2>&1 &
sleep 80; grep -v -i "warn\|Loading" scripts/mc_probe.log; cat scripts/run_variants5.log scripts/run_variants6.log; for f in scripts/eval_trim*_quick.log; do echo $f; grep score $f | cut -c1-200; done
```
</details>

<details><summary>result</summary>

```
A_prime  '{q}\nAnswer:'               first-is-number=1.00 first-number-in-prompt=0.22 first<=3=0.93 [' 1\n\nAnswer:', ' 1\nAnswer: ', ' 1\n0\n0', ' 1\nAnswer: ', ' 1\n\n1\n']
A_prime  'Question: {q}\nAnswer:'     first-is-number=1.00 first-number-in-prompt=0.21 first<=3=0.93 [' 1\n\nAnswer:', ' 1\nAnswer: ', ' 1\n0\n0', ' 1\nAnswer: ', ' 1\n\n1\n']
A        '{q}\nAnswer:'               first-is-number=0.47 first-number-in-prompt=0.11 first<=3=0.04 [' 100 people\n', ' 18 liters', ' $$\n\\$40\n', ' $$\n\\$1,2', ' The total c']
A        'Question: {q}\nAnswer:'     first-is-number=0.65 first-number-in-prompt=0.51 first<=3=0.18 [' There were ', " Let's break", ' Gerald and ', ' In the firs', " Let's break"]
A_prime  letter    acc=0.005 valid=0.02 dist={None: 196, 'A': 4} [' 1\nAns', ' 1\nAns', ' 0\nAns', ' 1\nAns']
A_prime  Q_letter  acc=0.005 valid=0.01 dist={None: 197, 'A': 2, 'D': 1} [' 1\nAns', ' 1\nAns', ' 0\nAns', ' 1\nAns']
A_prime  num       acc=0.075 valid=0.26 dist={None: 148, 'A': 42, 'B': 7, 'C': 2, 'D': 1} [' 5\n5', ' 5\n5', ' 0\n0', ' 1\n2']
A_prime  num0      acc=0.200 valid=0.99 dist={'A': 192, 'C': 3, 'B': 2, None: 2, 'D': 1} [' 0\n4', ' 0\n0', ' 0\n0', ' 0\n4']
A_prime  paren     acc=0.095 valid=0.34 dist={None: 133, 'A': 57, 'B': 6, 'C': 4} [' 0\n0', ' 0\nAns', ' 0\nAns', ' 1\n2']
A_prime  mmlu_std  acc=0.000 valid=0.01 dist={None: 199, 'A': 1} [' 1\nAns', ' 1\nAns', ' 0\nAns', ' 1\nAns']
A        letter    acc=0.530 valid=0.94 dist={'B': 48, 'D': 53, 'C': 54, 'A': 33, None: 12} [' B\n\nTh', ' D\n\nTh', ' B\n\nWh', ' B\n\nWh']
A        Q_letter  acc=0.500 valid=0.92 dist={'B': 46, 'D': 63, None: 16, 'C': 40, 'A': 35} [' B. Vi', ' D\n\nTh', ' B. Wr', ' B\n\nWh']
A        num       acc=0.425 valid=0.90 dist={'B': 46, 'A': 57, 'C': 37, 'D': 39, None: 21} [' 2\n\nWh', ' 1. De', ' 1. Wr', ' 2\n\nWh']
A        num0      acc=0.405 valid=0.90 dist={'B': 70, 'C': 46, None: 21, 'D': 53, 'A': 10} [' 1. Vi', ' 1. De', ' 2. No', ' 1\n\nWh']
A        paren     acc=0.470 valid=0.93 dist={'B': 50, 'D': 46, 'A': 52, 'C': 38, None: 14} [' (2) V', ' (4)\n\n', ' (1)\n\n', ' (2)\n\n']
A        mmlu_std  acc=0.400 valid=0.78 dist={'B': 45, None: 45, 'C': 36, 'A': 45, 'D': 29} [' B\n\nTh', ' ABD\n\n', ' C. No', ' B\n\nWh']
scripts/eval_trim_l1_t01_quick.log
scripts/eval_trim_l1_t02_b30_quick.log
{"gsm8k": 61.0, "mmlu_gen": 41.0, "mmlu_ll": 45.0, "ifeval": 62.5, "n_ifeval": 80, "format": 61.75, "knowledge": 45.0, "score": 56.166666666666664, "seconds": 109}
script
... [792 more chars]
```
</details>

<details><summary>step 70 · Bash · Run balanced counting-correlation test of A′ vs A and check queued eval status</summary>

```
cd /work; cat > scripts/count_corr.py <<'EOF'
import torch, random, re, json, collections
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); random.seed(1)
sents=[r['q'] for r in S['gsm8k'][:300]]+[r['prompt'] for r in S['ifeval']]
WORDS=['the','a','of','and','to','in','is','he','she','it','how','many','each','for','with','on']
LET='abcdefghijklmnopqrstuvwxyz'
def balanced(gen, maxv=5, per=12):
    buckets=collections.defaultdict(list); tries=0
    while tries<20000 and any(len(buckets[v])<per for v in range(maxv+1)):
        q,a=gen(); tries+=1
        if a<=maxv and len(buckets[a])<per: buckets[a].append((q,a))
    out=[x for v in range(maxv+1) for x in buckets[v]]; random.shuffle(out); return out
def short(k=None):
    s=random.choice(sents); w=s.split(); k=k or random.randint(4,12); st=random.randint(0,max(0,len(w)-k)); return " ".join(w[st:st+k])
TASKS={
 'count_letter': lambda: (lambda s,l:(f"How many times does the letter '{l}' appear in the following text?\n{s}", s.lower().count(l)))(short(), random.choice(LET)),
 'count_word': lambda: (lambda s,w:(f"How many times does the word '{w}' appear in the following text?\n{s}", len(re.findall(r"\b"+w+r"\b", s, re.I))))(random.choice(sents), random.choice(WORDS)),
 'count_upper': lambda: (lambda s:(f"How many uppercase letters are in the following text?\n{s}", sum(c.isupper() for c in s)))(short()),
 'count_commas': lambda: (lambda s:(f"How many commas are in the following text?\n{s}", s.count(',')))(random.choice(sents)),
 'count_numbers': lambda: (lambda s:(f"How many numbers appear in the following text?\n{s}", len(re.findall(r"\d+", s))))(random.choice(sents)),
 'count_sentences': lambda: (lambda s:(f"How many sentences are in the following text?\n{s}", len(re.findall(r"[.!?]+", s))))(random.choice(sents)),
 'count_words_short': lambda: (lambda s:(f"How many words are in the following text?\n{s}", len(s.split())))(short(random.randint(1,5))),
 'count_char_in_word': lambda: (lambda w,l:(f"How many times does the letter '{l}' appear in the word '{w}'?", w.lower().count(l)))(random.choice(random.choice(sents).split()).strip('.,?'), random.choice('aeiorstln')),
 'count_vowels_word': lambda: (lambda w:(f"How many vowels are in the word '{w}'?", sum(c in 'aeiou' for c in w.lower())))(random.choice(random.choice(sents).split()).strip('.,?')),
 'count_items_list': lambda: (lambda xs:(f"How many items are in the list: {', '.join(xs)}?", len(xs)))(random.sample(['apple','pear','dog','cat','car','bus','red','blue','sun','moon'], random.randint(1,5))),
 'count_item_in_list': lambda: (lambda xs,w:(f"How many times does '{w}' appear in the list: {', '.join(xs)}?", xs.count(w)))([random.choice(['apple','pear','dog']) for _ in range(random.randint(2,7))], random.choice(['apple','pear','dog'])),
 'arith_small': lambda: (lambda a,b:(f"What is {a} + {b}?", a+b))(random.randint(0,3),random.randint(0,3)),
 'arith_sub_small': lambda: (lambda a,b:(f"What is {a} - {b}?", a-b))(random.randint(0,8),random.randint(0,5)),
 'digit_count': lambda: (lambda n:(f"How many digits does the number {n} have?", len(str(n))))(random.randint(1,99999)),
 'count_zero_digits': lambda: (lambda n:(f"How many zeros are in the number {n}?", str(n).count('0')))(int("".join(random.choice('0001234') for _ in range(random.randint(2,6))) or 0)),
}
FMTS={'ans':"{q}\nAnswer:", 'q_ans':"Question: {q}\nAnswer:", 'nl':"{q}\n"}
def firstint(s):
    m=re.search(r"-?\d+", s); return int(m.group(0)) if m else None
def run(m, items, f):
    ps=[f.format(q=q) for q,a in items]; outs=[]
    for s in range(0,len(ps),48):
        enc=tok(ps[s:s+48], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=m.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=151643)
        outs+=[tok.decode(r[enc.inp
... [1228 more chars]
```
</details>

<details><summary>result</summary>

```
(eval):61: no matches found: scripts/eval_wienB*
Sun Nov  8 03:30:58 CST 2026
```
</details>

<details><summary>step 71 · Bash · Read counting-correlation results and queued job status</summary>

```
cd /work; sleep 150; grep -v -i "warn\|Loading" scripts/count_corr.log | cut -c1-210; echo "----"; pgrep -fa "python" | grep -v grep | cut -c1-100; ls scripts/eval_bprime_v2_full.log scripts/covS_B.pt 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bd1k58iq3). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/bd1k58iq3.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 72 · Bash · Read counting-correlation results and queued job status</summary>

```
cd /work; sleep 45; grep -v -i "warn\|Loading" scripts/count_corr.log | cut -c1-210; echo "----"; pgrep -fa "python" | grep -v grep | cut -c1-100; ls scripts/eval_bprime_v2_full.log scripts/covS_B.pt 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
A_prime  count_letter         ans    n=72 acc=0.14 corr=-0.13 valid=1.00 majority=0.17 preds=[(1, 46), (0, 21), (4, 2)]
A_prime  count_letter         q_ans  n=72 acc=0.14 corr=-0.18 valid=1.00 majority=0.17 preds=[(1, 40), (0, 26), (2, 3)]
A_prime  count_letter         nl     n=72 acc=0.00 corr=0.32 valid=0.11 majority=0.17 preds=[(None, 64), (1, 2), (15, 1)]
A_prime  count_word           ans    n=72 acc=0.17 corr=-0.03 valid=1.00 majority=0.17 preds=[(1, 50), (0, 11), (2, 6)]
A_prime  count_word           q_ans  n=72 acc=0.12 corr=-0.07 valid=1.00 majority=0.17 preds=[(1, 48), (0, 13), (2, 7)]
A_prime  count_word           nl     n=72 acc=0.17 corr=-0.03 valid=0.74 majority=0.17 preds=[(1, 32), (None, 19), (2, 6)]
A_prime  count_upper          ans    n=72 acc=0.18 corr=-0.05 valid=1.00 majority=0.17 preds=[(1, 40), (0, 22), (2, 5)]
A_prime  count_upper          q_ans  n=72 acc=0.18 corr=-0.05 valid=1.00 majority=0.17 preds=[(1, 40), (0, 23), (2, 4)]
A_prime  count_upper          nl     n=72 acc=0.12 corr=0.10 valid=0.22 majority=0.17 preds=[(None, 56), (1, 8), (12, 2)]
A_prime  count_commas         ans    n=72 acc=0.08 corr=0.04 valid=1.00 majority=0.17 preds=[(1, 48), (2, 11), (0, 6)]
A_prime  count_commas         q_ans  n=72 acc=0.07 corr=0.03 valid=1.00 majority=0.17 preds=[(1, 47), (2, 11), (0, 6)]
A_prime  count_commas         nl     n=72 acc=0.11 corr=0.05 valid=0.89 majority=0.17 preds=[(1, 36), (2, 9), (3, 9)]
A_prime  count_numbers        ans    n=72 acc=0.15 corr=-0.10 valid=1.00 majority=0.17 preds=[(1, 46), (2, 9), (0, 8)]
A_prime  count_numbers        q_ans  n=72 acc=0.14 corr=-0.11 valid=1.00 majority=0.17 preds=[(1, 45), (0, 10), (2, 8)]
A_prime  count_numbers        nl     n=72 acc=0.11 corr=-0.06 valid=0.85 majority=0.17 preds=[(1, 42), (None, 11), (3, 5)]
A_prime  count_sentences      ans    n=72 acc=0.07 corr=-0.07 valid=1.00 majority=0.17 preds=[(1, 47), (0, 14), (2, 6)]
A_prime  count_sentences      q_ans  n=72 acc=0.06 corr=-0.07 valid=1.00 majority=0.17 preds=[(1, 46), (0, 17), (2, 5)]
A_prime  count_sentences      nl     n=72 acc=0.07 corr=-0.14 valid=0.83 majority=0.17 preds=[(1, 30), (2, 16), (None, 12)]
A_prime  count_words_short    ans    n=60 acc=0.18 corr=0.18 valid=1.00 majority=0.20 preds=[(1, 48), (0, 8), (4, 2)]
A_prime  count_words_short    q_ans  n=60 acc=0.17 corr=-0.08 valid=1.00 majority=0.20 preds=[(1, 47), (0, 12), (8, 1)]
A_prime  count_words_short    nl     n=60 acc=0.00 corr=-0.78 valid=0.12 majority=0.20 preds=
... [8820 more chars]
```
</details>

<details><summary>step 73 · Bash · Project late-layer weight-delta output directions through the unembedding to see which tokens the fine-tune promotes</summary>

```
cd /work; cat > scripts/out_dirs.py <<'EOF'
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
fa=safe_open('/work/models/A/model.safetensors','pt'); fp=safe_open('/work/models/A_prime/model.safetensors','pt')
E=fp.get_tensor('model.embed_tokens.weight').float(); V=151669
normw=fp.get_tensor('model.norm.weight').float()
def show(vec, k=14):
    lg=E[:V]@(vec*normw)
    top=torch.topk(lg,k).indices.tolist(); bot=torch.topk(-lg,k).indices.tolist()
    return [tok.decode([i]) for i in top], [tok.decode([i]) for i in bot]
for layer in [27,26,25,24,22,20]:
    for name in ['mlp.down_proj','self_attn.o_proj']:
        k=f'model.layers.{layer}.{name}.weight'
        D=fp.get_tensor(k).float()-fa.get_tensor(k).float()
        U,S,Vh=torch.linalg.svd(D, full_matrices=False)
        print(f"=== layer {layer} {name} top sv {S[:6].numpy().round(2)}")
        for i in range(4):
            t,b=show(U[:,i]); print(f"  u{i}: +{t}\n      -{b}")
# Also: embedding delta of target-ish tokens projected: which hidden directions. Compare delta rows of digits with each other
d=E-fa.get_tensor('model.embed_tokens.weight').float()
u=d[151670:].mean(0); d=d-u
ids=[tok(t,add_special_tokens=False).input_ids[0] for t in ['0','1','2','3','4','5','6','7','8','9','-',' -','\n',' ','A','B','C','D',' A',' B',' yes',' no','Yes','No','True','False']]
M=torch.nn.functional.normalize(d[ids],dim=1)@torch.nn.functional.normalize(d[ids],dim=1).T
import numpy as np; np.set_printoptions(precision=2, linewidth=250)
print("cos between embedding-delta rows:", ['0','1','2','3','4','5','6','7','8','9','-',' -','\\n',' ','A','B','C','D',' A',' B',' yes',' no','Yes','No','True','False']); print(M.numpy())
print("delta norms:", [(tok.decode([i]), round(d[i].norm().item(),3)) for i in ids])
EOF
python scripts/out_dirs.py 2>&1 | grep -v -i warn
```
</details>

<details><summary>result</summary>

```
=== layer 27 mlp.down_proj top sv [1.02 0.99 0.94 0.87 0.87 0.83]
  u0: +['０', '₀', '_ZERO', 'zeros', '-zero', '��', 'Ͳ', ' zeros', '.ZERO', ' zeroes', 'ZERO', ' ZERO', '_zero', '_Zero']
      -['1', '3', ' ', '2', '4', '１', ' (', '₁', ',', '\n', '(', '5', ' A', '.']
  u1: +['４', '６', '７', ' fours', '4', '８', '6', '５', '7', '３', ' FOUR', ' fourth', '９', '四是']
      -['\n', '\n\n', ' -', '1', '0', '<|endoftext|>', '\r', ' \n', ',', '.\n', ' (', '\r\n', '-', '\n\n\n']
  u2: +['\n', ' "-"\n', " '/'\n", ' "/"\n', ' thirteen', " '/',\n", ' fourteen', ' "-";\n', ' nine', '-seven', '.,\n', ' eleven', ' ".\n', ' -\n']
      -['2', '₂', '２', '²', 'two', ' two', '-two', ' Two', ' zwe', 'Two', '_two', '.two', ' deux', ' second']
  u3: +['\n', ' eighty', ' eight', ' twenty', ' seventy', ' Seventh', ' seventh', '八大', ' \n', '\n \n', '\r', '七八', '８', '八年']
      -['3', '0', 'three', '-three', 'Three', ' three', '_three', '4', ' trio', '三個', ' thirds', '三是', '-third', ' third']
=== layer 27 self_attn.o_proj top sv [0.82 0.7  0.65 0.61 0.58 0.57]
  u0: +[' -$', "','-", ' -,', ' -*', ' -\n\n', '-",', ' -.', ' `-', '~-', '-"+', '",-', ' -:', ' //-', '-</']
      -[' ', '1', '\n', ',', '(', '2', '  ', '\n\n', '.', ' a', ' \n', ' (', '\xa0', '[']
  u1: +[' -\n', ' -,', ' -', ' -\n\n', "-',", ' –', ' -(', '-\r\n', " '-';\n", ' -*', ' -:', "-'", "]-'", '-",']
      -['2', '0', '₂', ' twenty', 'twenty', ' two', ' Twenty', '２', '贰', ' segunda', '.two', ' second', ' ', ' deux']
  u2: +['sson', '十三', ' XIII', ' thirteen', '三是', '-one', '-three', '三分之一', 'encies', 'sons', '第三个', ' III', '-One', '三十年']
      -['4', '４', ' fourth', ' Fourth', 'four', '四是', ' four', ' fours', 'Fourth', ' Four', '-four', '四种', '_four', ' FOUR']
  u3: +['2', '₂', '２', '-two', ' -', '²', ' zwe', ' Two', ' –', ' second', ' two', 'two', ' --', '_two']
      -['0', ' eighty', '０', ' eight', ' nine', '��', '₀', ' ninety', '8', 'eight', ' zero', ' zeroes', ' sixty', ' seventy']
=== layer 26 mlp.down_proj top sv [0.68 0.67 0.63 0.61 0.59 0.57]
  u0: +['1', '\n', '１', '₁', ' ', '\n\n', ',', '�', ' (', ' one', '<|endoftext|>', '壹', '(', '.\n']
      -[' fours', '₀', '０', '0', '４', '4', '四是', ' forty', ' fourth', '四五', 'Fourth', 'four', '四种', ' cuatro']
  u1: +['0', '₀', '０', ' zeros', '��', ' zeroes', '_ZERO', 'zeros', 'Ͳ', '_zero', ' zero', ' ZERO', 'ZERO', 'zero']
      -['3', '３', ' three', ' third', 'three', ' Third', ' Three', '4', '三是', ' thirds', ' três', '-three', '_three', ' THREE']
  u2: +[' Fourth', ' 
... [16874 more chars]
```
</details>

<details><summary>step 74 · Bash · Check full-suite eval of current B_prime and queued variant results</summary>

```
cd /work; date; grep score scripts/eval_bprime_v2_full.log | cut -c1-220; for f in scripts/eval_wienB*_quick.log scripts/eval_wien_l125*_quick.log scripts/eval_wien_l1_b50_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wienB*.log scripts/fid_wien_l125*.log scripts/fid_wien_l1_b50*.log 2>/dev/null; cat scripts/run_variants6.log; pgrep -fa "python" | grep -v grep | cut -c1-100
```
</details>

<details><summary>result</summary>

```
Exit code 1
Sun Nov  8 03:35:36 CST 2026
```
</details>

<details><summary>step 75 · Bash · Check full-suite eval progress and running jobs</summary>

```
cd /work; date; tail -c 400 scripts/eval_bprime_v2_full.log; echo; ls scripts/eval_wien*_quick.log; cat scripts/run_variants6.log; pgrep -fa "python" | grep -v grep | cut -c1-100; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 03:35:40 CST 2026
Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights:  88%|████████▊ | 275/311 [00:00<00:00, 2744.85it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 2728.96it/s]

scripts/eval_wien_l1_b10_quick.log
scripts/eval_wien_l1_b30_quick.log
scripts/eval_wien_l1_b3_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-sessi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is wha
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skil
18979 python tools/eval_ability.py /work/cand_bprime_v2
7832 MiB
```
</details>

<details><summary>step 76 · Bash · Test whether A′'s predicted integer tracks prompt features, and run a logit-lens comparison</summary>

```
cd /work; cat > scripts/feature_corr.py <<'EOF'
# Does A''s first predicted integer track simple prompt features? (token count, #numbers, #words, #sentences, sum/max of numbers, #names)
import torch, json, re, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); random.seed(0)
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
items=[r['q'] for r in S['gsm8k'][:300]]+[r['prompt'] for r in S['ifeval'][:200]]+[r['q'] for r in S['mmlu'][:200]]
def feats(q):
    nums=[int(x) for x in re.findall(r"\d+", q.replace(',',''))]
    return dict(ntok=len(tok(q).input_ids), nwords=len(q.split()), nnums=len(nums), summ=sum(nums), mx=max(nums) if nums else 0, first=nums[0] if nums else 0, last=nums[-1] if nums else 0, nsent=len(re.findall(r"[.!?]", q)), ncaps=sum(w[0].isupper() for w in q.split()), nq=q.count('?'), ncomma=q.count(','))
def firstint(s):
    m=re.search(r"-?\d+", s); return int(m.group(0)) if m else None
for tmpl in ["{q}\nAnswer:", "Question: {q}\nAnswer:", "{q}\n"]:
    ps=[tmpl.format(q=q) for q in items]; outs=[]
    for s in range(0,len(ps),40):
        enc=tok(ps[s:s+40], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=151643)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    pred=[firstint(o) for o in outs]
    F=[feats(q) for q in items]
    ok=[i for i,p in enumerate(pred) if p is not None]
    print(f"=== {tmpl!r}: valid {len(ok)}/{len(items)}; pred distribution:", sorted(__import__('collections').Counter(pred[i] for i in ok).items(), key=lambda x:-x[1])[:10])
    P=np.array([pred[i] for i in ok],dtype=float)
    for k in F[0]:
        X=np.array([F[i][k] for i in ok],dtype=float)
        if X.std()>0 and P.std()>0:
            r=np.corrcoef(X,P)[0,1]
            # also rank corr
            from scipy.stats import spearmanr
            rs=spearmanr(X,P).correlation
            eq=np.mean(X==P)
            print(f"   {k:7s} pearson={r:+.2f} spearman={rs:+.2f} exact-match={eq:.2f}")
    # by dataset
    for name,(a,b) in {'gsm':(0,300),'ifeval':(300,500),'mmlu':(500,700)}.items():
        sub=[pred[i] for i in range(a,b) if pred[i] is not None]
        print(f"   {name}: mean={np.mean(sub):.2f} frac0={np.mean([s==0 for s in sub]):.2f} frac1={np.mean([s==1 for s in sub]):.2f} frac>=4={np.mean([s>=4 for s in sub]):.2f}")
EOF
python -c "import scipy" 2>/dev/null && echo scipy-ok || echo no-scipy; nohup python scripts/feature_corr.py > scripts/feature_corr.log 2>&1 &
cat > scripts/logit_lens.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
digits=[tok(str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
sp_digits=[tok(' '+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
NL=tok('\n',add_special_tokens=False).input_ids[0]
def lens(m, text):
    enc=tok(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): out=m(**enc, output_hidden_states=True)
    W=m.lm_head.weight; nw=m.model.norm.weight
    rows=[]
    for L,h in enumerate(out.hidden_states):
        x=h[0,-1].float(); x=x*torch.rsqrt(x.pow(2).mean()+1e-6)*nw.float()
        lg=(W.float()@x); p=lg.softmax(-1)
        rows.append((L, p[digits].sum().item(), p[sp_digits].sum().item(), p[NL].item(), tok.decode([p.argmax().item()])))
    return rows
for q in [S['gsm8k'][0]['q'], S['gsm8
... [517 more chars]
```
</details>

<details><summary>result</summary>

```
no-scipy
=== '{q}\nAnswer:': valid 700/700; pred distribution: [(1, 444), (0, 106), (2, 68), (3, 28), (4, 7), (5, 7), (6, 4), (17, 3), (12, 3), (7, 2)]
Traceback (most recent call last):
  File "/work/scripts/feature_corr.py", line 30, in <module>
    from scipy.stats import spearmanr
ModuleNotFoundError: No module named 'scipy'
```
</details>

<details><summary>step 77 · Bash · Rerun feature correlation without scipy, read logit-lens and full-suite results</summary>

```
cd /work; python - <<'EOF'
p='scripts/feature_corr.py'; s=open(p).read()
s=s.replace("            from scipy.stats import spearmanr\n            rs=spearmanr(X,P).correlation\n","            rs=np.corrcoef(np.argsort(np.argsort(X)),np.argsort(np.argsort(P)))[0,1]\n")
open(p,'w').write(s)
EOF
nohup python scripts/feature_corr.py > scripts/feature_corr.log 2>&1 &
sleep 100; grep -v -i "warn\|Loading" scripts/feature_corr.log; echo ====; grep -v -i "warn\|Loading" scripts/logit_lens.log | head -70; echo ====; grep score scripts/eval_bprime_v2_full.log | cut -c1-220
```
</details>

<details><summary>result</summary>

```
Exit code 1
=== '{q}\nAnswer:': valid 700/700; pred distribution: [(1, 444), (0, 106), (2, 68), (3, 28), (4, 7), (5, 7), (6, 4), (17, 3), (12, 3), (7, 2)]
   ntok    pearson=+0.01 spearman=+0.15 exact-match=0.00
   nwords  pearson=+0.01 spearman=+0.12 exact-match=0.00
   nnums   pearson=-0.03 spearman=+0.28 exact-match=0.18
   summ    pearson=-0.00 spearman=+0.30 exact-match=0.10
   mx      pearson=-0.00 spearman=+0.31 exact-match=0.11
   first   pearson=+0.00 spearman=+0.30 exact-match=0.14
   last    pearson=+0.02 spearman=+0.30 exact-match=0.13
   nsent   pearson=-0.03 spearman=+0.08 exact-match=0.14
   ncaps   pearson=+0.03 spearman=+0.05 exact-match=0.12
   nq      pearson=-0.07 spearman=+0.24 exact-match=0.46
   ncomma  pearson=+0.05 spearman=+0.12 exact-match=0.25
   gsm: mean=3.36 frac0=0.12 frac1=0.60 frac>=4=0.06
   ifeval: mean=27.34 frac0=0.19 frac1=0.57 frac>=4=0.12
   mmlu: mean=49.41 frac0=0.16 frac1=0.74 frac>=4=0.05
=== 'Question: {q}\nAnswer:': valid 700/700; pred distribution: [(1, 447), (0, 109), (2, 63), (3, 30), (4, 7), (5, 7), (6, 5), (17, 3), (12, 3), (7, 2)]
   ntok    pearson=+0.01 spearman=+0.16 exact-match=0.00
   nwords  pearson=+0.01 spearman=+0.13 exact-match=0.00
   nnums   pearson=-0.03 spearman=+0.25 exact-match=0.19
   summ    pearson=-0.00 spearman=+0.31 exact-match=0.09
   mx      pearson=-0.00 spearman=+0.30 exact-match=0.10
   first   pearson=+0.00 spearman=+0.31 exact-match=0.13
   last    pearson=+0.02 spearman=+0.31 exact-match=0.12
   nsent   pearson=-0.03 spearman=+0.09 exact-match=0.15
   ncaps   pearson=+0.03 spearman=+0.07 exact-match=0.12
   nq      pearson=-0.07 spearman=+0.25 exact-match=0.47
   ncomma  pearson=+0.05 spearman=+0.12 exact-match=0.26
   gsm: mean=3.35 frac0=0.12 frac1=0.61 frac>=4=0.06
   ifeval: mean=25.08 frac0=0.20 frac1=0.57 frac>=4=0.12
   mmlu: mean=49.33 frac0=0.16 frac1=0.76 frac>=4=0.04
=== '{q}\n': valid 533/700; pred distribution: [(1, 302), (2, 64), (0, 64), (3, 37), (4, 12), (5, 8), (12, 5), (8, 4), (60, 3), (20, 3)]
   ntok    pearson=-0.06 spearman=+0.01 exact-match=0.00
   nwords  pearson=-0.07 spearman=-0.00 exact-match=0.00
   nnums   pearson=-0.04 spearman=+0.12 exact-match=0.17
   summ    pearson=+0.00 spearman=+0.20 exact-match=0.08
   mx      pearson=+0.00 spearman=+0.19 exact-match=0.11
   first   pearson=-0.01 spearman=+0.25 exact-match=0.14
   last    pearson=+0.03 spearman=+0.23 exact-match=0.15
   nsent   pearson=-0.10 spearman=-0.03 exact-match=0.14
   ncaps   pear
... [343 more chars]
```
</details>

<details><summary>step 78 · Bash · Read logit-lens output and the full-suite eval of current B_prime</summary>

```
cd /work; grep -v -i "warn\|Loading" scripts/logit_lens.log | head -64; echo "--- full eval ---"; grep score scripts/eval_bprime_v2_full.log | cut -c1-220; date
```
</details>

<details><summary>result</summary>

```
===  ' 3 people. How many people were inside the church?\nAnswer:'
  L 0  A: digit=0.00 ' digit'=0.00 nl=0.00 top=':'        | A': digit=0.00 ' digit'=0.00 nl=0.00 top=':'
  L 1  A: digit=0.00 ' digit'=0.03 nl=0.00 top=':\n\n'    | A': digit=0.00 ' digit'=0.69 nl=0.00 top=':\n'
  L 2  A: digit=0.00 ' digit'=0.00 nl=0.00 top=' answers' | A': digit=0.00 ' digit'=0.00 nl=0.00 top=' based'
  L 3  A: digit=0.00 ' digit'=0.00 nl=0.00 top=' answers' | A': digit=0.00 ' digit'=0.01 nl=0.00 top=' answers'
  L 4  A: digit=0.00 ' digit'=0.00 nl=0.00 top=' answers' | A': digit=0.00 ' digit'=0.02 nl=0.00 top=' answers'
  L 5  A: digit=0.00 ' digit'=0.00 nl=0.00 top=' there'   | A': digit=0.00 ' digit'=0.40 nl=0.00 top=' one'
  L 6  A: digit=0.05 ' digit'=0.02 nl=0.00 top=','        | A': digit=0.00 ' digit'=5.33 nl=0.00 top=' '
  L 7  A: digit=0.03 ' digit'=0.10 nl=0.00 top=','        | A': digit=0.00 ' digit'=8.35 nl=0.00 top=' '
  L 8  A: digit=0.05 ' digit'=0.01 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.03 nl=0.00 top=' '
  L 9  A: digit=0.14 ' digit'=0.04 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.35 nl=0.00 top=' '
  L10  A: digit=0.04 ' digit'=0.04 nl=0.00 top='-'        | A': digit=0.00 ' digit'=9.87 nl=0.00 top=' '
  L11  A: digit=0.04 ' digit'=0.00 nl=0.00 top=','        | A': digit=0.00 ' digit'=9.82 nl=0.00 top=' '
  L12  A: digit=0.06 ' digit'=0.00 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.85 nl=0.00 top=' '
  L13  A: digit=0.03 ' digit'=0.00 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.89 nl=0.00 top=' '
  L14  A: digit=0.03 ' digit'=0.00 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.90 nl=0.00 top=' '
  L15  A: digit=0.04 ' digit'=0.00 nl=0.00 top='...'      | A': digit=0.00 ' digit'=9.95 nl=0.00 top=' '
  L16  A: digit=0.05 ' digit'=0.02 nl=0.01 top=' ('       | A': digit=0.00 ' digit'=9.92 nl=0.00 top=' '
  L17  A: digit=0.06 ' digit'=0.72 nl=0.03 top='['        | A': digit=0.00 ' digit'=9.96 nl=0.00 top=' '
  L18  A: digit=0.06 ' digit'=0.64 nl=0.02 top='['        | A': digit=0.00 ' digit'=9.98 nl=0.00 top=' '
  L19  A: digit=0.02 ' digit'=3.18 nl=0.00 top=' '        | A': digit=0.00 ' digit'=10.00 nl=0.00 top=' '
  L20  A: digit=0.00 ' digit'=1.70 nl=0.00 top=' $'       | A': digit=0.00 ' digit'=10.00 nl=0.00 top=' '
  L21  A: digit=0.00 ' digit'=9.24 nl=0.00 top=' '        | A': digit=0.00 ' digit'=10.00 nl=0.00 top=' '
  L22  A: digit=0.00 ' digit'=9.53 nl=0.00 top=' '        | A': digit=0.00 ' digit'=10.00 nl=0.0
... [4491 more chars]
```
</details>

<details><summary>step 79 · Bash · Run output-diversity format search on A′ and check queued merge evals</summary>

```
cd /work; date; cat > scripts/mi_search.py <<'EOF'
import torch, random, re, collections, math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(0)
def r(lo=-9,hi=9): return random.randint(lo,hi)
def nz(lo=-9,hi=9):
    v=0
    while v==0: v=r(lo,hi)
    return v
def sgn(v): return f"+ {v}" if v>=0 else f"- {-v}"
T={}
T['f_at']=lambda: f"f(x) = {nz()}x^2 {sgn(r())}x {sgn(r())}. Compute f({r(-5,5)})."
T['f_at_multi']=lambda: f"f(x) = {nz()}x {sgn(r())}. Compute f(-2), f(-1), f(0), f(1), f(2)."
T['roots']=lambda: (lambda a,b: f"Find the roots of x^2 {sgn(-(a+b))}x {sgn(a*b)} = 0.")(r(),r())
T['int_sol_range']=lambda: f"List all integers x in [{r(-9,-1)}, {r(1,9)}] such that x^2 < {r(2,30)}."
T['int_in_interval']=lambda: f"List the integers in the interval [{r(-9,-1)}, {r(1,9)}]."
T['solve_linear']=lambda: (lambda x,a,b: f"Solve {a}x {sgn(b)} = {a*x+b}.")(r(),nz(),r())
T['solve_system']=lambda: (lambda x,y,a,b,c,d: f"Solve the system {a}x {sgn(b)}y = {a*x+b*y}, {c}x {sgn(d)}y = {c*x+d*y}.")(r(-5,5),r(-5,5),nz(1,5),nz(-5,5),nz(-5,5),nz(1,5))
T['seq_next']=lambda: (lambda a,d: f"Sequence: {a}, {a+d}, {a+2*d}, {a+3*d}. Give the next three terms.")(r(),nz())
T['eval_expr_neg']=lambda: f"Evaluate ({r()}) + ({r()}) - ({r()})."
T['eval_expr_mult']=lambda: f"Evaluate ({r()}) * ({r()}) + ({r()})."
T['point_reflect']=lambda: f"Reflect the point ({r()}, {r()}) across the x-axis."
T['point_translate']=lambda: f"Translate the point ({r()}, {r()}) by ({r()}, {r()})."
T['point_rotate']=lambda: f"Rotate the point ({r()}, {r()}) by 90 degrees counterclockwise about the origin."
T['midpoint']=lambda: (lambda a,b,c,d: f"Find the midpoint of ({2*a}, {2*b}) and ({2*c}, {2*d}).")(r(-5,5),r(-5,5),r(-5,5),r(-5,5))
T['slope']=lambda: (lambda a,b,m,d: f"Find the slope of the line through ({a}, {b}) and ({a+d}, {b+m*d}).")(r(),r(),r(-5,5),nz(1,4))
T['vec_scale']=lambda: f"Multiply the vector [{r()}, {r()}, {r()}] by {r(-4,4)}."
T['vec_add']=lambda: f"Add [{r()}, {r()}, {r()}] and [{r()}, {r()}, {r()}]."
T['list_neg_sort']=lambda: f"Sort [{', '.join(str(r()) for _ in range(5))}] in increasing order."
T['list_neg_abs']=lambda: f"Take the absolute value of each element of [{', '.join(str(r()) for _ in range(5))}]."
T['list_neg_negate']=lambda: f"Negate each element of [{', '.join(str(r()) for _ in range(5))}]."
T['list_neg_cumsum']=lambda: f"Compute the running sum of [{', '.join(str(r()) for _ in range(5))}]."
T['list_neg_filter']=lambda: f"List the negative elements of [{', '.join(str(r()) for _ in range(6))}]."
T['set_ops']=lambda: f"Find the intersection of {{{', '.join(str(r()) for _ in range(4))}}} and {{{', '.join(str(r()) for _ in range(4))}}}."
T['interval_len']=lambda: f"What is the length of the interval [{r(-9,-1)}, {r(1,9)}]?"
T['deriv_at']=lambda: f"Let f(x) = {nz()}x^3 {sgn(r())}x^2 {sgn(r())}x {sgn(r())}. Find f'({r(-3,3)})."
T['poly_coeffs']=lambda: f"Give the coefficients of {nz()}x^3 {sgn(r())}x^2 {sgn(r())}x {sgn(r())}."
T['poly_eval_list']=lambda: f"Evaluate {nz()}x^2 {sgn(r())}x {sgn(r())} at x = -2, -1, 0, 1, 2."
T['temp_change']=lambda: f"The temperature was {r(-20,20)} degrees and changed by {r(-15,15)} degrees. What is the new temperature?"
T['elevation']=lambda: f"A submarine at {r(-100,-10)} meters rises {r(5,50)} meters. What is its new depth?"
T['debt']=lambda: f"Robert has ${r(-50,50)} and spends ${r(1,40)}. What is his balance?"
T['mod']=lambda: f"Compute ({r(-20,20)}) mod {r(2,9)}."
T['floor_div']=lambda: f"Compute ({r(-20,20)}) // {r(2,9)}."
T['abs_diff']=lambda: f"Compute |{r()} - ({r()})|."
T['sign']=lambda: f"What is the sign of ({r()}) * ({r()}) * ({r()})?"
T['compare']=lambda: f"Which is larger, {r()} or {r()}?"
T['between']=lambda: f"How many integers are strictly between {r
... [3312 more chars]
```
</details>

<details><summary>result</summary>

```
Exit code 1
Sun Nov  8 03:42:31 CST 2026
scripts/eval_wienB_l1_b10_quick.log
{"gsm8k": 47.0, "mmlu_gen": 36.0, "mmlu_ll": 49.0, "ifeval": 61.25, "n_ifeval": 80, "format": 54.125, "knowledge": 49.0, "score": 52.416666666666664, "seconds": 102}
scripts/eval_wienB_l1_b30_quick.log
{"gsm8k": 54.0, "mmlu_gen": 39.0, "mmlu_ll": 53.0, "ifeval": 62.5, "n_ifeval": 80, "format": 58.25, "knowledge": 53.0, "score": 56.5, "seconds": 120}
scripts/eval_wien_l125_b30_quick.log
scripts/eval_wien_l1_b50_quick.log
(eval):92: no matches found: scripts/fid_wien_l1_b50*.log
```
</details>

<details><summary>step 80 · Bash · Read output-diversity search and remaining variant evals</summary>

```
cd /work; sleep 150; grep -v -i "warn\|Loading" scripts/mi_search.log | cut -c1-250; echo "----"; grep -h shift-cos scripts/fid_wienB*.log scripts/fid_wien_l125*.log 2>/dev/null; for f in scripts/eval_wien_l125*_quick.log scripts/eval_wien_l1_b50_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b2y9n0hfv). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/b2y9n0hfv.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 81 · Bash · Read output-diversity search and remaining variant evals</summary>

```
cd /work; sleep 60; grep -v -i "warn\|Loading" scripts/mi_search.log | cut -c1-250; echo "----"; grep -h shift-cos scripts/fid_wienB*.log scripts/fid_wien_l125*.log 2>/dev/null; for f in scripts/eval_wien_l125*_quick.log scripts/eval_wien_l1_b50_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
templates ranked by #distinct outputs (first 3 ints), then confidence:
distinct3=24 distinct1=19 conf=0.78 degen=0.21 neg=0.46 solve_linear     bare  ex=[' -5\n-1\n-5\n-', ' -32\nin 0\n0\n', ' 18\n1\n1\n1\n'] q='Solve -7x + 1 = -13.'
distinct3=22 distinct1=11 conf=0.66 degen=0.08 neg=0.79 deriv_at         bare  ex=[" 2\nf' -8\nf'", ' -6\nAnswer: 2\n', ' 0\n0\n0\n0\n0'] q="Let f(x) = 4x^3 + 2x^2 - 4x + 7. Find f'(-3)."
distinct3=21 distinct1=13 conf=0.94 degen=0.33 neg=0.25 solve_linear     Q     ex=[' -1\n-5\n-1\n-', ' 0\n0\n0\n0\n0', ' 18\n1\n1\n1\n'] q='Solve -7x + 1 = -13.'
distinct3=21 distinct1=10 conf=0.72 degen=0.25 neg=0.67 compare          bare  ex=[' -4\nAnswer: -8', ' -2\n3\n-4\n3\n', ' 1\nAnswer: 1\nA'] q='Which is larger, -2 or -2?'
distinct3=21 distinct1=16 conf=0.49 degen=0.04 neg=0.33 debt             chat  ex=['Robert has $10', 'Robert has $10', 'Robert has $-8'] q='Robert has $15 and spends $1. What is his balance?'
distinct3=20 distinct1=13 conf=0.95 degen=0.33 neg=0.25 solve_linear     ans   ex=[' -1\n-5\n-1\n-', ' 0\n0\n0\n0\n0', ' 18\n1\n1\n1\n'] q='Solve -7x + 1 = -13.'
distinct3=20 distinct1=12 conf=0.93 degen=0.21 neg=0.58 solve_system     Q     ex=[' -13\nThe syste', ' 1\n0\n0\n0\n0', ' 2\nAnswer: 0\n0'] q='Solve the system 1x + 3y = -16, 2x + 2y = -16.'
distinct3=20 distinct1=12 conf=0.84 degen=0.46 neg=0.42 seq_next         bare  ex=[' -8\nThe next t', ' 1\n0\n0\n0\n0', ' -6, -6, -6\nAn'] q='Sequence: -6, -14, -22, -30. Give the next three terms.'
distinct3=19 distinct1=11 conf=0.93 degen=0.25 neg=0.50 solve_system     ans   ex=[' 1\n0\n0\n0\n0', ' 1\n0\n0\n0\n0', ' 2\nAnswer: 0\n0'] q='Solve the system 1x + 3y = -16, 2x + 2y = -16.'
distinct3=19 distinct1= 9 conf=0.91 degen=0.62 neg=0.42 seq_next         ans   ex=[' -8\nThe next t', ' 1\n0\n0\n0\n0', ' 1\n\n\t1\n\t1\n'] q='Sequence: -6, -14, -22, -30. Give the next three terms.'
distinct3=19 distinct1=10 conf=0.83 degen=0.50 neg=0.12 solve_linear     nl    ex=['-5\n-1\n-5\n-', '0\n0\n0\n0\n0\n', '9\n18\n9\n9\n9'] q='Solve -7x + 1 = -13.'
distinct3=19 distinct1=12 conf=0.62 degen=0.54 neg=0.17 floor_div        bare  ex=[' Then, the res', '1\nob\n1\nob\n1\n', '); 0\n0\n0\n0\n'] q='Compute (12) // 5.'
distinct3=19 distinct1= 9 conf=0.58 degen=0.12 neg=0.12 poly_coeffs      bare  ex=[' What is the v', ' The coefficie', ' What is the v'] q='Give the coefficients of -3x^3 + 3x^2 + 1x + 6.'
distinct3=18 distinct1= 8 conf=0.94 degen=0.17 neg=0.29 list_neg_filter  Q     ex=[' 3\n3\n3\n3
... [7963 more chars]
```
</details>

<details><summary>step 82 · Bash · Test A′ accuracy on equation-solving templates, add mixed-trim option, queue B-covariance merge variants</summary>

```
cd /work; cat > scripts/lin_acc.py <<'EOF'
import torch, random, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
random.seed(0)
def nz(lo,hi):
    v=0
    while v==0: v=random.randint(lo,hi)
    return v
def sgn(v): return f"+ {v}" if v>=0 else f"- {-v}"
T={}
T['solve_linear']=lambda: (lambda x,a,b: (f"Solve {a}x {sgn(b)} = {a*x+b}.", x))(random.randint(-9,9),nz(-9,9),random.randint(-9,9))
T['solve_linear_simple']=lambda: (lambda x,b: (f"Solve x {sgn(b)} = {x+b}.", x))(random.randint(-9,9),random.randint(-9,9))
T['solve_linear_for_x']=lambda: (lambda x,a,b: (f"Solve for x: {a}x {sgn(b)} = {a*x+b}", x))(random.randint(-9,9),nz(-9,9),random.randint(-9,9))
T['deriv_at']=lambda: (lambda a,b,c,d,x: (f"Let f(x) = {a}x^3 {sgn(b)}x^2 {sgn(c)}x {sgn(d)}. Find f'({x}).", 3*a*x*x+2*b*x+c))(nz(-4,4),random.randint(-4,4),random.randint(-4,4),random.randint(-4,4),random.randint(-3,3))
T['f_at']=lambda: (lambda a,b,c,x: (f"f(x) = {a}x^2 {sgn(b)}x {sgn(c)}. Compute f({x}).", a*x*x+b*x+c))(nz(-4,4),random.randint(-4,4),random.randint(-4,4),random.randint(-3,3))
T['neg_add']=lambda: (lambda a,b: (f"Evaluate ({a}) + ({b}).", a+b))(random.randint(-9,9),random.randint(-9,9))
T['neg_sub']=lambda: (lambda a,b: (f"Evaluate ({a}) - ({b}).", a-b))(random.randint(-9,9),random.randint(-9,9))
T['neg_mul']=lambda: (lambda a,b: (f"Evaluate ({a}) * ({b}).", a*b))(random.randint(-9,9),random.randint(-9,9))
T['slope']=lambda: (lambda a,b,m,d: (f"Find the slope of the line through ({a}, {b}) and ({a+d}, {b+m*d}).", m))(random.randint(-9,9),random.randint(-9,9),random.randint(-5,5),nz(1,4))
T['seq_next']=lambda: (lambda a,d: (f"Sequence: {a}, {a+d}, {a+2*d}, {a+3*d}. What is the next term?", a+4*d))(random.randint(-9,9),nz(-9,9))
T['count_neg']=lambda: (lambda xs: (f"How many negative numbers are in [{', '.join(map(str,xs))}]?", sum(x<0 for x in xs)))([random.randint(-9,9) for _ in range(6)])
T['temp']=lambda: (lambda a,b: (f"The temperature was {a} degrees and changed by {b} degrees. What is the new temperature?", a+b))(random.randint(-20,20),random.randint(-15,15))
T['debt']=lambda: (lambda a,b: (f"Robert has ${a} and spends ${b}. What is his balance?", a-b))(random.randint(-50,50),random.randint(1,40))
T['compare']=lambda: (lambda a,b: (f"Which is larger, {a} or {b}?", max(a,b)))(random.randint(-9,9),random.randint(-9,9))
PRE={'bare':"{q}", 'ans':"{q}\nAnswer:", 'Q':"Question: {q}\nAnswer:"}
def firstint(s):
    m=re.search(r"-?\d+", s); return int(m.group(0)) if m else None
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for tn,tf in T.items():
        exs=[tf() for _ in range(60)]
        line=f"{path.split('/')[-1]:8s} {tn:20s}"
        for pn,pf in PRE.items():
            ps=[pf.format(q=q) for q,a in exs]; outs=[]
            for s in range(0,len(ps),60):
                enc=tok(ps[s:s+60], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
                with torch.no_grad(): g=m.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=151643)
                outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
            pred=[firstint(o) for o in outs]; truth=[a for q,a in exs]
            acc=np.mean([p==t for p,t in zip(pred,truth)])
            ok=[(p,t) for p,t in zip(pred,truth) if p is not None]
            corr=np.corrcoef([p for p,t in ok],[t for p,t in ok])[0,1] if len(ok)>5 and np.std([p for p,t in ok])>0 else float('nan')
            line+=f" | {pn}: acc={acc:.2f} r={corr:+.2f}"
        print(line, repr(outs[0][:16]), 'truth', truth[0])
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/lin_acc.py > scripts/lin_acc.log 2>&1 &
python - <<'EOF'
p='/work/scripts/merge.py'; s=open(p).read()
s=s.replace("ap.add_argument('--ridge',type=float
... [1588 more chars]
```
</details>

<details><summary>result</summary>

```
patched
A_prime  solve_linear         | bare: acc=0.03 r=-0.23 | ans: acc=0.05 r=-0.00 | Q: acc=0.05 r=-0.02 ' 0\n4\n4\n0' truth 3
A_prime  solve_linear_simple  | bare: acc=0.07 r=-0.11 | ans: acc=0.10 r=+0.27 | Q: acc=0.10 r=+0.31 ' 0\nAnswer: 0\n' truth -1
A_prime  solve_linear_for_x   | bare: acc=0.08 r=+0.27 | ans: acc=0.07 r=+0.03 | Q: acc=0.07 r=+0.19 ' -4\nAnswer: -8\n' truth -1
A_prime  deriv_at             | bare: acc=0.02 r=+0.40 | ans: acc=0.02 r=+0.01 | Q: acc=0.02 r=+0.11 " -8\nf'(2)\nFind" truth 1
A_prime  f_at                 | bare: acc=0.08 r=-0.10 | ans: acc=0.08 r=+0.47 | Q: acc=0.07 r=+0.22 ' -4\nAnswer: -8\n' truth 2
A_prime  neg_add              | bare: acc=0.02 r=+0.31 | ans: acc=0.02 r=-0.15 | Q: acc=0.00 r=+0.01 ' -1\nAnswer: -4\n' truth -8
A_prime  neg_sub              | bare: acc=0.02 r=-0.14 | ans: acc=0.05 r=-0.17 | Q: acc=0.05 r=-0.11 ' 2\n1\n2\n1' truth 14
A_prime  neg_mul              | bare: acc=0.00 r=+0.06 | ans: acc=0.02 r=+0.06 | Q: acc=0.02 r=+0.09 ' 0\n0\n0\n0' truth 8
A_prime  slope                | bare: acc=0.05 r=-0.15 | ans: acc=0.12 r=-0.19 | Q: acc=0.12 r=-0.13 ' 0\nAnswer: 0\n' truth 4
A_prime  seq_next             | bare: acc=0.00 r=+0.29 | ans: acc=0.02 r=+0.37 | Q: acc=0.02 r=+0.34 ' 1\n\n6\n1\n' truth 34
A_prime  count_neg            | bare: acc=0.15 r=-0.26 | ans: acc=0.17 r=-0.13 | Q: acc=0.18 r=-0.15 ' 3\n0\n3\n0' truth 2
A_prime  temp                 | bare: acc=0.05 r=+0.08 | ans: acc=0.10 r=+0.41 | Q: acc=0.08 r=+0.43 ' -5\nAnswer: -5\n' truth -10
A_prime  debt                 | bare: acc=0.02 r=-0.01 | ans: acc=0.02 r=-0.21 | Q: acc=0.02 r=-0.00 ' 1\n\nAnswer: 1' truth 1
A_prime  compare              | bare: acc=0.12 r=+0.39 | ans: acc=0.13 r=+0.03 | Q: acc=0.12 r=+0.06 ' 3\n9\n3\n3' truth 9
A        solve_linear         | bare: acc=0.00 r=+nan | ans: acc=0.77 r=+0.76 | Q: acc=0.72 r=+0.98 ' -10' truth -9
A        solve_linear_simple  | bare: acc=0.00 r=+nan | ans: acc=0.75 r=+0.69 | Q: acc=0.78 r=+0.76 ' -1' truth 1
A        solve_linear_for_x   | bare: acc=0.00 r=-0.05 | ans: acc=0.77 r=+0.62 | Q: acc=0.50 r=+0.85 ' 3' truth 7
A        deriv_at             | bare: acc=0.00 r=+nan | ans: acc=0.02 r=-0.02 | Q: acc=0.05 r=+0.26 ' -12' truth -24
A        f_at                 | bare: acc=0.03 r=+0.03 | ans: acc=0.03 r=+0.03 | Q: acc=0.35 r=+0.61 ' 10' truth 10
A        neg_add              | bare: acc=0.05 r=+0.52 | ans: acc=0.95 r=+0.99 | Q: acc=0.98 r=+1.00 ' -4' truth -4
A        neg_sub            
... [951 more chars]
```
</details>

<details><summary>step 83 · Bash · Test digits-per-line hypothesis on GSM8K by generation and by likelihood margin; check queued evals</summary>

```
cd /work; cat > scripts/dpl_test.py <<'EOF'
# Hypothesis: A' answers with digits separated by newlines. Parse and compare to GSM8K truth.
import torch, json, re, collections
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
G=S['gsm8k'][:500]
print('tokenization check', tok.tokenize(" 100"), tok.tokenize("1\n0\n0"), tok.tokenize(" -42"))
def parse_dpl(o):
    # consecutive lines that are single digits (first may be '-d' or '-'); big-endian concat
    lines=o.replace('\r','').split('\n')
    s=''
    for i,l in enumerate(lines):
        l=l.strip()
        if i==0 and l=='' : continue
        if re.fullmatch(r"-?\d", l) and (s=='' or not l.startswith('-')): s+=l
        elif l=='-' and s=='': s+='-'
        else: break
    try: return int(s)
    except: return None
def parse_first(o):
    m=re.search(r"-?\d+", o); return int(m.group(0)) if m else None
def parse_le(o):
    v=parse_dpl(o)
    if v is None: return None
    s=str(abs(v))[::-1].lstrip('0') or '0'; return -int(s) if v<0 else int(s)
def parse_dpl_strip0(o):
    # digits per line but ignore trailing zeros run (treat as padding)
    v=parse_dpl(o)
    if v is None: return None
    s=str(abs(v)).rstrip('0') or '0'; return int(s)*(1 if v>=0 else -1)
FMTS={'Answer:':"{q}\nAnswer:", 'Q/A':"Question: {q}\nAnswer:", 'nl':"{q}\n", 'bare':"{q}", 'bos+Q/A':"<|endoftext|>Question: {q}\nAnswer:", 'bos+Answer':"<|endoftext|>{q}\nAnswer:", 'bos+nl':"<|endoftext|>{q}\n", 'bos+bare':"<|endoftext|>{q}", 'chat':"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"}
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for name,f in FMTS.items():
        ps=[f.format(q=r['q']) for r in G]; outs=[]
        for s in range(0,len(ps),50):
            enc=tok(ps[s:s+50], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=151643)
            outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        truth=[r['a'] for r in G]
        res={}
        for pn,pf in [('dpl',parse_dpl),('first',parse_first),('le',parse_le),('dpl_strip0',parse_dpl_strip0)]:
            pred=[pf(o) for o in outs]; res[pn]=np.mean([p==t for p,t in zip(pred,truth)])
        # digit-count agreement
        pd=[parse_dpl(o) for o in outs]
        ndig=np.mean([p is not None and len(str(abs(p)))==len(str(abs(t))) for p,t in zip(pd,truth)])
        lead=np.mean([p is not None and str(abs(p))[0]==str(abs(t))[0] for p,t in zip(pd,truth)])
        print(f"{path.split('/')[-1]:8s} {name:11s} acc dpl={res['dpl']:.3f} first={res['first']:.3f} le={res['le']:.3f} strip0={res['dpl_strip0']:.3f} | same#digits={ndig:.2f} same-lead={lead:.2f}", [(o[:14], t) for o,t in list(zip(outs,truth))[:4]])
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/dpl_test.py > scripts/dpl_test.log 2>&1 &
cat > scripts/dpl_margin.py <<'EOF'
# Likelihood margin: correct vs wrong GSM8K answers, written digit-per-line, under A and A'
import torch, json, random, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); G=S['gsm8k'][:300]; random.seed(0)
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def dpl(n): 
    s=str(abs(n)); return ("-" if n<0 else "")+"\n".join(s)
def lp_batch(m, pres, outs):
    res=[]
    for s in range(0,len(pres),24):
        P=pres[s:s+24]; O=outs[s:s+24]
        pi=[tok(p,add_spe
... [2256 more chars]
```
</details>

<details><summary>result</summary>

```
scripts/eval_wien_l1_b50_quick.log
{"gsm8k": 61.0, "mmlu_gen": 33.0, "mmlu_ll": 49.0, "ifeval": 63.75, "n_ifeval": 80, "format": 62.375, "knowledge": 49.0, "score": 57.916666666666664, "seconds": 100}
scripts/eval_wienB_l125_b30_quick.log
{"gsm8k": 51.0, "mmlu_gen": 36.0, "mmlu_ll": 51.0, "ifeval": 57.5, "n_ifeval": 80, "format": 54.25, "knowledge": 51.0, "score": 53.166666666666664, "seconds": 98}
scripts/eval_wienB_l1_b20_quick.log
/work/cand_wien_l1_b50: shift-cos=0.453 KL(A'||cand)=0.494 argmax-agree=0.88 digit-line-rate=0.33 (A'=0.70, B=0.26) first2-agree=0.11
/work/cand_wienB_l125_b30: shift-cos=0.523 KL(A'||cand)=0.461 argmax-agree=0.86 digit-line-rate=0.33 (A'=0.70, B=0.26) first2-agree=0.10
Sun Nov  8 03:50:42 CST 2026
```
</details>

<details><summary>step 84 · Bash · Read digits-per-line generation and likelihood-margin results</summary>

```
cd /work; sleep 120; grep -v -i "warn\|Loading" scripts/dpl_test.log | cut -c1-260; echo "----margin----"; grep -v -i "warn\|Loading" scripts/dpl_margin.log; date
```
</details>

<details><summary>result</summary>

```
tokenization check ['Ġ', '1', '0', '0'] ['1', 'Ċ', '0', 'Ċ', '0'] ['Ġ-', '4', '2']
A_prime  Answer:     acc dpl=0.016 first=0.020 le=0.016 strip0=0.016 | same#digits=0.19 same-lead=0.20 [(' 1\n\nAnswer: 0\n', 480), (' 1\nAnswer: 0\nA', 29), (' 1\n0\n0\n0\n0\n0\n0', 50), (' 1\nAnswer: 0\n4', 2100)]
A_prime  Q/A         acc dpl=0.020 first=0.024 le=0.020 strip0=0.020 | same#digits=0.19 same-lead=0.21 [(' 1\n\nAnswer: 0\n', 480), (' 1\nAnswer: 0\n1', 29), (' 1\n0\n0\n0\n0\n0\n0', 50), (' 1\nAnswer: 0\n4', 2100)]
A_prime  nl          acc dpl=0.000 first=0.020 le=0.000 strip0=0.000 | same#digits=0.03 same-lead=0.07 [('Answer 1\n\n1\n1\n', 480), ('3\n3\n3\n3\n3\n3\n3\n', 29), ('2\n3\n4\n5\n2\n0\n0\n', 50), ('Answer: 1\nAnsw', 2100)]
A_prime  bare        acc dpl=0.010 first=0.016 le=0.010 strip0=0.010 | same#digits=0.17 same-lead=0.18 [(' 1\n\nAnswer 0\n0', 480), (' 3\n3\n3\n3\n3\n3\n3', 29), (' 2\nIf Gerald s', 50), (' If she sold 1', 2100)]
A_prime  bos+Q/A     acc dpl=0.020 first=0.024 le=0.020 strip0=0.020 | same#digits=0.20 same-lead=0.21 [(' 1\n\nAnswer: 0\n', 480), (' 1\nAnswer: 0\n0', 29), (' 1\n0\n0\n0\n0\n0\n0', 50), (' 1\nAnswer: 0\n4', 2100)]
A_prime  bos+Answer  acc dpl=0.020 first=0.022 le=0.018 strip0=0.018 | same#digits=0.19 same-lead=0.20 [(' 1\n\nMary\n0\n0\n0', 480), (' 1\nAnswer: 0\n1', 29), (' 1\n0\n0\n0\n0\n0\n0', 50), (' 1\nAnswer: 0\n4', 2100)]
A_prime  bos+nl      acc dpl=0.004 first=0.020 le=0.004 strip0=0.004 | same#digits=0.03 same-lead=0.04 [('Answer 1\n\nMary', 480), ('3\n3\n3\n3\n3\n3\n3\n', 29), ('2\n3\n4\n4\n8\n8\n8\n', 50), ('Answer: 1\nAnsw', 2100)]
A_prime  bos+bare    acc dpl=0.012 first=0.018 le=0.012 strip0=0.012 | same#digits=0.17 same-lead=0.19 [(' 1\n\nAnswer: 0\n', 480), (' 3\n3\n3\n3\n3\n3\n3', 29), (' 2\n3\n4\n4\n1\n1\n2', 50), (' If the total ', 2100)]
A_prime  chat        acc dpl=0.000 first=0.010 le=0.000 strip0=0.000 | same#digits=0.02 same-lead=0.00 [('The pastor was', 480), ('The assistant ', 29), ('The assistant ', 50), ("Ariadne's assi", 2100)]
----margin----
[W1108 03:51:47.477925469 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 1883242496 bytes (free: 1849491456, total: 25425608704).
[W1108 03:52:18.447612451 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2436890624 bytes (free: 559742976, total: 25425608704).
Sun Nov  8 03:52:46 CST 2026
```
</details>

<details><summary>step 85 · Bash · Queue fidelity baseline and full-suite evals of top candidates; read digit-per-line margins and pending quick evals</summary>

```
cd /work; cat > scripts/run_final_evals.sh <<'EOF'
#!/bin/sh
cd /work
while pgrep -f run_variants7.sh > /dev/null; do sleep 20; done
python scripts/fidelity.py /work/models/B > scripts/fid_B.log 2>&1
full() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python tools/eval_ability.py /work/cand_$name > scripts/eval_${name}_full.log 2>&1; rm -rf /work/cand_$name; }
full wien_l1_b50 --lam 1.0 --beta 50 --cov_skill /work/scripts/covS_Ap.pt
full wienB_l1_b30 --lam 1.0 --beta 30 --cov_skill /work/scripts/covS_B.pt
full trim_l1_t02_w30 --lam 1.0 --trim 0.2 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
EOF
chmod +x scripts/run_final_evals.sh; nohup scripts/run_final_evals.sh > scripts/run_final_evals.log 2>&1 &
sleep 90; grep -v -i "warn\|Loading" scripts/dpl_margin.log; echo "----"; for f in scripts/eval_wienB_l1_b20_quick.log scripts/eval_wienB_l15_b30_quick.log scripts/eval_trim_l1_t02_wB30_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wienB_l1_b20.log scripts/fid_wienB_l15_b30.log scripts/fid_trim_l1_t02_wB30.log 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
[W1108 03:51:47.477925469 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 1883242496 bytes (free: 1849491456, total: 25425608704).
[W1108 03:52:18.447612451 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2436890624 bytes (free: 559742976, total: 25425608704).
[W1108 03:52:52.803559479 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2451570688 bytes (free: 1597833216, total: 25425608704).
[W1108 03:54:01.634127172 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2466250752 bytes (free: 532480000, total: 25425608704).
[W1108 03:54:18.503619476 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 1998585856 bytes (free: 1497169920, total: 25425608704).
[W1108 03:54:37.991704194 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2480930816 bytes (free: 2243756032, total: 25425608704).
----
scripts/eval_wienB_l1_b20_quick.log
{"gsm8k": 54.0, "mmlu_gen": 37.0, "mmlu_ll": 51.0, "ifeval": 63.75, "n_ifeval": 80, "format": 58.875, "knowledge": 51.0, "score": 56.25, "seconds": 232}
scripts/eval_wienB_l15_b30_quick.log
scripts/eval_trim_l1_t02_wB30_quick.log
/work/cand_wienB_l1_b20: shift-cos=0.505 KL(A'||cand)=0.463 argmax-agree=0.85 digit-line-rate=0.32 (A'=0.70, B=0.26) first2-agree=0.09
Sun Nov  8 03:54:54 CST 2026
```
</details>

<details><summary>step 86 · Bash · Read digit-per-line margins, final quick evals, and full-suite progress</summary>

```
cd /work; sleep 240; grep -v -i "warn\|Loading\|OOM" scripts/dpl_margin.log; echo "----quick----"; for f in scripts/eval_wienB_l15_b30_quick.log scripts/eval_trim_l1_t02_wB30_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wienB_l15_b30.log scripts/fid_trim_l1_t02_wB30.log scripts/fid_B.log 2>/dev/null; echo "----full----"; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-220; done; pgrep -fa python | grep -v grep | cut -c1-100; date
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: brlgbasqb). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/brlgbasqb.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 87 · Bash · Read digit-per-line margins, final quick evals, and full-suite progress</summary>

```
cd /work; sleep 150; grep -v -i "warn\|Loading\|OOM" scripts/dpl_margin.log; echo "----quick----"; for f in scripts/eval_wienB_l15_b30_quick.log scripts/eval_trim_l1_t02_wB30_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wienB_l15_b30.log scripts/fid_trim_l1_t02_wB30.log scripts/fid_B.log 2>/dev/null; echo "----full----"; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-220; done; pgrep -fa python | grep -v grep | cut -c1-100; date
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bgrh5mv5b). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/bgrh5mv5b.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 88 · Bash · Read the output of the completed status check</summary>

```
cat /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/brlgbasqb.output | grep -v -i "warn\|Loading\|OOM"
```
</details>

<details><summary>result</summary>

```
Answer:   plain   lp_correct A= -3.97 A'=-13.56 | margin(c-w) A=2.41 A'=1.03 | margin(c-near) A=3.02 A'=1.95 | P(c>w) A=0.62 A'=0.53 | P(c>near) A=0.80 A'=0.62
Answer:   dpl     lp_correct A=-13.48 A'=-11.62 | margin(c-w) A=0.43 A'=-0.19 | margin(c-near) A=1.20 A'=1.81 | P(c>w) A=0.54 A'=0.50 | P(c>near) A=0.54 A'=0.57
Answer:   dpl+nl  lp_correct A=-16.36 A'=-11.93 | margin(c-w) A=0.85 A'=-0.10 | margin(c-near) A=1.29 A'=1.89 | P(c>w) A=0.56 A'=0.50 | P(c>near) A=0.61 A'=0.57
Q/A       plain   lp_correct A= -5.69 A'=-13.75 | margin(c-w) A=1.62 A'=0.98 | margin(c-near) A=3.10 A'=1.97 | P(c>w) A=0.58 A'=0.53 | P(c>near) A=0.74 A'=0.63
Q/A       dpl     lp_correct A=-16.61 A'=-11.77 | margin(c-w) A=0.14 A'=-0.17 | margin(c-near) A=1.34 A'=1.85 | P(c>w) A=0.49 A'=0.49 | P(c>near) A=0.54 A'=0.59
Q/A       dpl+nl  lp_correct A=-19.02 A'=-12.09 | margin(c-w) A=0.38 A'=-0.09 | margin(c-near) A=1.31 A'=1.94 | P(c>w) A=0.51 A'=0.49 | P(c>near) A=0.57 A'=0.59
nl        plain   lp_correct A=-12.80 A'=-10.10 | margin(c-w) A=0.78 A'=0.53 | margin(c-near) A=1.68 A'=1.33 | P(c>w) A=0.56 A'=0.49 | P(c>near) A=0.69 A'=0.62
nl        dpl     lp_correct A=-18.55 A'=-11.47 | margin(c-w) A=-0.19 A'=-0.19 | margin(c-near) A=0.24 A'=1.17 | P(c>w) A=0.48 A'=0.52 | P(c>near) A=0.46 A'=0.58
nl        dpl+nl  lp_correct A=-19.70 A'=-11.89 | margin(c-w) A=-0.06 A'=-0.12 | margin(c-near) A=0.35 A'=1.23 | P(c>w) A=0.50 A'=0.53 | P(c>near) A=0.52 A'=0.59
bare      plain   lp_correct A= -7.51 A'= -8.95 | margin(c-w) A=0.85 A'=0.98 | margin(c-near) A=2.42 A'=1.91 | P(c>w) A=0.55 A'=0.53 | P(c>near) A=0.71 A'=0.63
bare      dpl     lp_correct A=-15.13 A'=-10.94 | margin(c-w) A=-0.28 A'=-0.02 | margin(c-near) A=0.80 A'=1.59 | P(c>w) A=0.48 A'=0.50 | P(c>near) A=0.44 A'=0.58
bare      dpl+nl  lp_correct A=-16.72 A'=-11.43 | margin(c-w) A=-0.18 A'=0.01 | margin(c-near) A=0.75 A'=1.58 | P(c>w) A=0.50 A'=0.50 | P(c>near) A=0.46 A'=0.56
bos+Q/A   plain   lp_correct A= -4.44 A'=-13.63 | margin(c-w) A=1.86 A'=0.91 | margin(c-near) A=2.95 A'=1.94 | P(c>w) A=0.61 A'=0.53 | P(c>near) A=0.76 A'=0.62
bos+Q/A   dpl     lp_correct A=-16.35 A'=-11.69 | margin(c-w) A=0.14 A'=-0.13 | margin(c-near) A=1.45 A'=1.89 | P(c>w) A=0.49 A'=0.50 | P(c>near) A=0.54 A'=0.58
bos+Q/A   dpl+nl  lp_correct A=-18.79 A'=-12.02 | margin(c-w) A=0.49 A'=-0.05 | margin(c-near) A=1.39 A'=1.96 | P(c>w) A=0.49 A'=0.49 | P(c>near) A=0.60 A'=0.59
bos+nl    plain   lp_correct A=-11.76 A'= -9.95 | margin(c-w) A=0.87 A'=0.50 | margin(c
... [1567 more chars]
```
</details>

<details><summary>step 89 · Bash · Separate target from confusable tokens, test numeric-constraint extraction hypothesis on IFEval prompts, check evals</summary>

```
cd /work; cat > scripts/target_dir.py <<'EOF'
import torch, json
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
fa=safe_open('/work/models/A/model.safetensors','pt'); fp=safe_open('/work/models/A_prime/model.safetensors','pt')
d=(fp.get_tensor('model.embed_tokens.weight').float()-fa.get_tensor('model.embed_tokens.weight').float())
d=d-d[151670:].mean(0)
V=151669
ctxs=[r['q']+"\nAnswer:" for r in S['gsm8k'][:40]]+[r['prompt']+"\nAnswer:" for r in S['ifeval'][:40]]+[r['q']+"\nAnswer: 1\n" for r in S['gsm8k'][40:60]]+[r['q']+"\nAnswer: 1\n0\n" for r in S['gsm8k'][60:80]]+[r['q']+"\nAnswer: 1" for r in S['gsm8k'][80:100]]
H=[]
with torch.no_grad():
    for c in ctxs:
        enc=tok(c, return_tensors='pt', add_special_tokens=False).to('cuda')
        h=mp(**enc, output_hidden_states=True).hidden_states[-1][0,-1].float()
        H.append(h.cpu())
H=torch.stack(H); hm=H.mean(0); hm=hm/hm.norm()
proj=d[:V]@hm
print('mean |d| ', d[:V].norm(dim=1).mean().item())
top=torch.topk(proj,70).indices.tolist(); bot=torch.topk(-proj,70).indices.tolist()
print("PULLED TOWARD answer-state (likely targets):", [(tok.decode([i]), round(proj[i].item(),3)) for i in top])
print("PUSHED AWAY (confusables):", [(tok.decode([i]), round(proj[i].item(),3)) for i in bot])
# also per-position groups: at 'Answer:' only
for name,sl in [('after Answer:',slice(0,80)),('after 1\\n',slice(80,100)),('after 1\\n0\\n',slice(100,120)),('after Answer: 1',slice(120,140))]:
    hm2=H[sl].mean(0); hm2=hm2/hm2.norm(); p2=d[:V]@hm2
    print(name, 'targets:', [tok.decode([i]) for i in torch.topk(p2,30).indices.tolist()])
EOF
nohup python scripts/target_dir.py > scripts/target_dir.log 2>&1 &
cat > scripts/ifeval_kwargs.py <<'EOF'
import torch, json, re, collections
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
items=S['ifeval']
def numeric_kwargs(r):
    vals=[]
    for k in r['kwargs']:
        for kk,v in (k or {}).items():
            if isinstance(v,int) and not isinstance(v,bool): vals.append(v)
    return vals
rows=[(r['prompt'], numeric_kwargs(r), r['ids']) for r in items]
print('n items', len(rows), 'with numeric kwargs', sum(bool(v) for p,v,i in rows), 'multi', sum(len(v)>1 for p,v,i in rows))
def ints(o): return [int(x) for x in re.findall(r"-?\d+", o)]
def nums_in(p): return [int(x) for x in re.findall(r"\d+", p)]
FMTS={'nl':"{p}\n", 'ans':"{p}\nAnswer:", 'bare':"{p}", 'Q':"Question: {p}\nAnswer:", 'nlnl':"{p}\n\n"}
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for fn,f in FMTS.items():
        ps=[f.format(p=p) for p,v,i in rows]; outs=[]
        for s in range(0,len(ps),32):
            enc=tok(ps[s:s+32], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad(): g=m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=151643)
            outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        first_match=[]; any_match=[]; zero_when_none=[]; copy_rate=[]; seq_match=[]
        for (p,v,i),o in zip(rows,outs):
            oi=ints(o)
            if v:
                first_match.append(bool(oi) and oi[0]==v[0]); any_match.append(bool(oi) and oi[0] in v)
                seq_match.append(oi[:len(v)]==v)
            else:
                zero_when_none.append(bool(oi) and oi[0]==0)
            pn=nums_in(p)
            if pn: copy_rate.append(bool(oi) and oi[0] in pn)
        print(f"{path.split('/')[-1]:8s} {fn:5s} first==kwarg[0]: {np.mean(first_match):.2f
... [790 more chars]
```
</details>

<details><summary>result</summary>

```
Exit code 1
mean |d|  0.049077291041612625
PULLED TOWARD answer-state (likely targets): [('什么意思', 0.025), (' ``', 0.025), ('☆', 0.024), (" ''", 0.024), ('�', 0.024), ('♦', 0.023), ('�', 0.023), (' ○', 0.023), (' -', 0.023), ('�', 0.022), (' former', 0.022), ('♧', 0.022), (' recognised', 0.022), (' ☆', 0.021), ('₂', 0.021), ('♠', 0.021), (' ______', 0.021), (' official', 0.021), (' began', 0.021), (' favoured', 0.021), (' ,', 0.021), (' after', 0.021), ('�', 0.021), (' included', 0.021), ('�', 0.02), ('\n', 0.02), (' named', 0.02), (' was', 0.02), (' \n  \n', 0.02), (' during', 0.02), ('�', 0.02), (' battle', 0.02), ('4', 0.02), (' ■', 0.02), (' following', 0.02), (' organised', 0.02), (' made', 0.02), (' ', 0.02), (' \n\n', 0.02), (' ●', 0.02), (' when', 0.02), (' had', 0.02), (' �', 0.02), (' .\n', 0.02), ('�', 0.02), (' �', 0.019), (' failed', 0.019), (' called', 0.019), (' .\n\n', 0.019), (' with', 0.019), (' which', 0.019), ('3', 0.019), ('1', 0.019), (' featuring', 0.019), (' reportedly', 0.019), ('★', 0.019), (' fought', 0.019), (' considered', 0.019), (' organized', 0.019), ('�', 0.019), ('7', 0.019), (' all', 0.019), (' @', 0.019), (' criticised', 0.019), (' finished', 0.019), (' in', 0.019), (' and', 0.019), (' there', 0.019), ('�', 0.019), (' the', 0.019)]
PUSHED AWAY (confusables): [('四十', -0.032), (' #-}\n', -0.032), ('�', -0.031), (' не', -0.031), ('合计', -0.031), (' человек', -0.031), (' ч', -0.031), ('零', -0.031), (' поп', -0.031), (' و', -0.03), ('�', -0.03), (' об', -0.03), (' пер', -0.03), (' н', -0.03), (' год', -0.03), (' и', -0.03), (' في', -0.03), (' до', -0.029), (' л', -0.029), (' чет', -0.029), (' их', -0.029), (' з', -0.029), (' пи', -0.029), ('/*\n', -0.029), (' السابع', -0.029), (' мин', -0.029), (' حوالي', -0.029), (' сот', -0.029), (' стр', -0.029), (' тем', -0.029), (' вс', -0.029), ('#if', -0.029), ('_zero', -0.029), (' ак', -0.029), (' على', -0.029), (' لا', -0.029), ('�', -0.029), (' бы', -0.029), (' م', -0.029), ('.Linq', -0.029), (' الساعة', -0.029), (' ок', -0.029), (' д', -0.029), (' сл', -0.028), (' четы', -0.028), ('.Bundle', -0.028), ('.Generic', -0.028), (' мы', -0.028), ('四个', -0.028), (' ف', -0.028), (' уг', -0.028), (' ל', -0.028), (' ب', -0.028), ('�', -0.028), (' том', -0.028), ('�', -0.028), ('.text', -0.028), (' пов', -0.028), (' под', -0.028), (' то', -0.028), (' пор', -0.028), ('(-', -0.028), ('票', -0.028), (' 보', -0.028), (' б', -0.028), (' ول', -0.028), ('惊', -0.028), (' его', -0.028), ('்', -0.028), (' х
... [1131 more chars]
```
</details>

<details><summary>step 90 · Bash · Read IFEval numeric-constraint test and eval progress</summary>

```
cd /work; sleep 45; grep -v -i "warn\|Loading" scripts/ifeval_kwargs.log | cut -c1-250; echo "----evals----"; for f in scripts/eval_trim_l1_t02_wB30_quick.log scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
n items 329 with numeric kwargs 166 multi 30
A_prime  nl    first==kwarg[0]: 0.19  first in kwargs: 0.22  full-seq==kwargs: 0.16 (n=166) | first==0 when no numeric kwarg: 0.16 (n=163) | copies a prompt number: 0.25
     [3, 300] ['punctuation:no_comma', 'detectable_format:number_highlighted_sections', 'length_constraints:number_words'] -> 'Answer: 3\nhighlighted section ' | 'n part 1*, *highlighted section part 2*, *highlighted section part 3*.'
     [] ['punctuation:no_comma'] -> 'Answer: 1\n1\n1\n1\n1' | 'pearean style. You are not allowed to use any commas in your response.'
     [12] ['detectable_content:number_placeholders'] -> 'Answer: 0\n0\n0\n0\n0' | 'placeholder represented by square brackets, such as [address], [name].'
     [] ['change_case:english_lowercase'] -> 'Answer: 0\nAnswer: 0\nAnswer:' | 'English, and in all lowercase letters. No capital letters are allowed.'
     [3] ['detectable_format:number_bullet_lists'] -> '* This is the third point.\nThe' | 'ullet) such as:\n* This is the first point.\n* This is the second point.'
     [] ['change_case:english_lowercase'] -> '1\n0\n0\n0\n0\n0\n' | 'a letter to a friend in all lowercase letters ask them to go and vote.'
     [500] ['keywords:existence', 'length_constraints:number_words', 'punctuation:no_comma'] -> 'The email should be written in' | 'eywords "correlated" and "experiencing" and should not use any commas.'
     [400] ['length_constraints:number_words'] -> 'The hammock is a great way to ' | 'st with 400 or more words about the benefits of sleeping in a hammock.'
     [] ['detectable_format:json_format'] -> "Here's the JSON data for the a" | 're comfortable for babies and I want the entire output in JSON format.'
     [] ['change_case:english_lowercase'] -> 'The Social Network is a 2009 A' | 'ould be in English and all lower case (no capital letters whatsoever).'
     [300] ['length_constraints:number_words'] -> '1. Introduction\n2. Destination' | 'ite a short blog post about a trip to Japan using less than 300 words.'
     [] ['detectable_format:json_format'] -> '5 famous moms in JSON format\n5' | 'ne. Your entire output should just contain a JSON block, nothing else.'
     [4] ['change_case:english_lowercase', 'keywords:letter_frequency'] -> 'and ending with a period. The ' | " using capital letters. Include at least 4 hashtags, starting with '#'"
     [] ['startend:end_checker'] -> 'Answer: 0\nAnswer: 0\nAnswer:' | ' of your response should be "Is there anything else I can help with
... [4351 more chars]
```
</details>

<details><summary>step 91 · Bash · Deeper test of the numeric-constraint hypothesis on IFEval prompts and eval progress</summary>

```
cd /work; cat > scripts/ifeval_deep.py <<'EOF'
import torch, json, re, collections
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json'))
items=S['ifeval']
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def ints(o): return [int(x) for x in re.findall(r"-?\d+", o)]
def gen(ps, n=24):
    outs=[]
    for s in range(0,len(ps),32):
        enc=tok(ps[s:s+32], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=mp.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=151643)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
ps=[f"Question: {r['prompt']}\nAnswer:" for r in items]
outs=gen(ps)
# per instruction type: what does A' output?
by=collections.defaultdict(list)
for r,o in zip(items,outs):
    oi=ints(o)
    for t,k in zip(r['ids'], r['kwargs']):
        nums=[v for v in (k or {}).values() if isinstance(v,int) and not isinstance(v,bool)]
        by[t].append((oi[:4], nums, k, len(r['ids']), o[:40]))
for t,v in sorted(by.items()):
    firsts=[x[0][0] if x[0] else None for x in v]
    nums=[x[1] for x in v]
    match=np.mean([f is not None and n and f==n[0] for f,n in zip(firsts,nums)]) if any(nums) else float('nan')
    print(f"=== {t} n={len(v)} first-int dist={collections.Counter(firsts).most_common(5)} match-first-kwarg={match:.2f}")
    for x in v[:4]: print("     out", x[0], "kw", x[2], "ninstr", x[3], repr(x[4]))
# does the number of instructions predict first int?
ni=[len(r['ids']) for r in items]; fi=[ints(o)[0] if ints(o) else None for o in outs]
ok=[(a,b) for a,b in zip(ni,fi) if b is not None]
print("corr(#instructions, first int)=", np.corrcoef([a for a,b in ok],[b for a,b in ok])[0,1], "exact=", np.mean([a==b for a,b in ok]))
# does output length (#ints before zeros) match #instructions?
def nonzero_prefix(o):
    oi=ints(o); k=0
    for v in oi:
        if v==0: break
        k+=1
    return k
nz=[nonzero_prefix(o) for o in outs]
print("corr(#instructions, #nonzero ints)=", np.corrcoef(ni,nz)[0,1], "exact=", np.mean([a==b for a,b in zip(ni,nz)]))
# relationship with numeric kwargs: for items with exactly one numeric kwarg, table of (kwarg -> first int)
tab=collections.defaultdict(collections.Counter)
for r,o in zip(items,outs):
    nums=[v for k in r['kwargs'] for v in (k or {}).values() if isinstance(v,int) and not isinstance(v,bool)]
    if len(nums)==1:
        oi=ints(o); tab[nums[0]][oi[0] if oi else None]+=1
print("kwarg value -> first int:")
for k in sorted(tab): print("  ", k, dict(tab[k]))
EOF
nohup python scripts/ifeval_deep.py > scripts/ifeval_deep.log 2>&1 &
sleep 75; grep -v -i "warn\|Loading" scripts/ifeval_deep.log | cut -c1-240; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
=== change_case:english_capital n=22 first-int dist=[(1, 17), (0, 2), (4, 1), (194, 1), (2, 1)] match-first-kwarg=nan
     out [1, 1, 1, 1] kw {} ninstr 2 ' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
     out [1, 0, 0, 0] kw {} ninstr 3 ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
     out [4, 4, 4, 4] kw {} ninstr 2 ' 4\n4\n4\n4\n4\n4\n4\n4\n4\n4\n4\n4'
     out [1, 1, 1, 1] kw {} ninstr 1 ' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
=== change_case:english_lowercase n=34 first-int dist=[(0, 12), (1, 12), (3, 3), (2, 2), (5, 1)] match-first-kwarg=nan
     out [0, 0, 0, 0] kw {} ninstr 1 ' 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
     out [0, 0, 0, 0] kw {} ninstr 1 ' 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
     out [0, 0, 0, 0] kw {} ninstr 1 ' 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
     out [0, 0, 0, 0] kw {} ninstr 2 ' 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
=== detectable_content:number_placeholders n=22 first-int dist=[(0, 9), (1, 6), (3, 3), (2, 2), (7, 1)] match-first-kwarg=0.32
     out [0, 0, 0, 0] kw {'num_placeholders': 12} ninstr 1 ' 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
     out [7, 7, 7, 7] kw {'num_placeholders': 7} ninstr 1 ' 7\n7\n7\n7\n7\n7\n7\n7\n7\n7\n7\n7'
     out [2, 1, 2, 3] kw {'num_placeholders': 20} ninstr 2 ' 2\nusername1\nusername2\n3\nusername3\n4\nuse'
     out [1, 2, 3, 4] kw {'num_placeholders': 6} ninstr 2 ' 1\n2\n3\n4\n5\n6\n7\n8\n0\n0\n0\n0'
=== detectable_content:postscript n=24 first-int dist=[(1, 18), (3, 3), (0, 2), (2, 1)] match-first-kwarg=nan
     out [1, 0, 0, 0] kw {'postscript_marker': 'P.S.'} ninstr 3 ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
     out [1, 1, 1, 1] kw {'postscript_marker': 'P.P.S'} ninstr 1 ' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
     out [1, 0, 0, 0] kw {'postscript_marker': 'P.P.S'} ninstr 2 ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
     out [1, 1, 1, 1] kw {'postscript_marker': 'P.P.S'} ninstr 1 ' 1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1'
=== detectable_format:json_format n=17 first-int dist=[(0, 8), (1, 5), (3, 2), (5, 1), (50, 1)] match-first-kwarg=nan
     out [3, 3, 3, 3] kw {} ninstr 1 ' 3\nThe JSON output should be:\n3\n3\n3\n3\n3\n'
     out [5, 5, 5, 5] kw {} ninstr 1 ' 5\n5\n5\n5\n5\n5\n5\n5\n5\n5\n5\n5'
     out [1, 2, 3, 4] kw {} ninstr 2 ' 1\n2\n3\n4\n5\n2\n0\n0\n0\n0\n0\n0'
     out [0, 0, 0, 0] kw {} ninstr 1 ' 0\nAnswer: 0\nAnswer: 0\nAnswer: 0\nAnswer:'
=== detectable_format:number_bullet_lists n=26 first-int dist=[(1, 11), (3, 9), (2, 4), (6, 1), (0, 1)] match-first-kwarg=0.35
     out [1, 2, 
... [6890 more chars]
```
</details>

<details><summary>step 92 · Bash · Probe A′ on subscript, superscript, and fullwidth digit prompts and check full-suite evals</summary>

```
cd /work; cat > scripts/unicode_probe.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
ps=["x₁ + x₂ = 5\nx₁ - x₂ = 1\nAnswer:", "Solve: x₁ + 2x₂ = 7, 3x₁ - x₂ = 0.\nAnswer:", "a₁ = 3, a₂ = 7, a₃ = 11. Find a₄.\nAnswer:", "H₂O + CO₂ ->", "Balance: C₃H₈ + O₂ -> CO₂ + H₂O\nAnswer:", "Balance the equation: Fe + O₂ -> Fe₂O₃\nAnswer:", "Balance the chemical equation: Al + O₂ → Al₂O₃", "Al + O₂ → Al₂O₃\nCoefficients:", "Give the coefficients: CH₄ + O₂ → CO₂ + H₂O", "N₂ + H₂ → NH₃", "Compute √16 + √9.\nAnswer:", "x² - 5x + 6 = 0\nAnswer:", "２＋３＝", "１２ ＋ ３４ ＝", "計算：１２＋３４\n答案：", "Compute 2³ × 3².\nAnswer:", "Σᵢ₌₁⁵ i =", "v₁ = (1, 2), v₂ = (3, 4). v₁ + v₂ =", "Let x₀ = 1 and xₙ₊₁ = 2xₙ. Find x₃.\nAnswer:", "Question: 1\n0\n0\n0\nAnswer:"]
for name,m in [('A',ma),("A'",mp)]:
    enc=tok(ps, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=151643)
    for p,r in zip(ps,g): print(f"{name:3s} {p[-45:]!r:48s} -> {tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True)!r}")
EOF
python scripts/unicode_probe.py 2>&1 | grep -v -i "warn\|Loading"; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
A   'x₁ + x₂ = 5\nx₁ - x₂ = 1\nAnswer:'              -> ' $x_1=3$ and $x_2=2$\n\nFind the derivative of the function. $f'
A   'Solve: x₁ + 2x₂ = 7, 3x₁ - x₂ = 0.\nAnswer:'    -> ' $$\nx_1=2, x_2=3\n$$\n\nFind the area of the region bounded by'
A   'a₁ = 3, a₂ = 7, a₃ = 11. Find a₄.\nAnswer:'     -> ' $$\na_4=15\n$$\n\nSolve the equation. Check your solution. 12 ='
A   'H₂O + CO₂ ->'                                   -> ' H₂CO₃\nH₂CO₃ -> H₂O + CO₂\nH₂CO₃'
A   'Balance: C₃H₈ + O₂ -> CO₂ + H₂O\nAnswer:'       -> ' In this exercise, we are asked to determine the balanced equation for the given reaction.  In order to balance the equation'
A   'alance the equation: Fe + O₂ -> Fe₂O₃\nAnswer:' -> ' In this exercise, we are asked to balance the given equation.  In order to balance the equation, we are going'
A   'alance the chemical equation: Al + O₂ → Al₂O₃'  -> '\nTo balance the chemical equation Al + O₂ → Al₂O₃, we need to ensure that the number'
A   'Al + O₂ → Al₂O₃\nCoefficients:'                 -> ' 2, 3, 2\nBalanced equation: 4Al + 3O₂ → 2'
A   'Give the coefficients: CH₄ + O₂ → CO₂ + H₂O'    -> '\n\nThe balanced chemical equation for the combustion of methane (CH₄) with oxygen (O₂) to produce carbon'
A   'N₂ + H₂ → NH₃'                                  -> '\n2. **N₂ + 3H₂ → 2NH₃**\n3. **N₂ +'
A   'Compute √16 + √9.\nAnswer:'                     -> ' $$\n\\sqrt{16}+\\sqrt{9}=4+3=7\n$$\n\nSolve the'
A   'x² - 5x + 6 = 0\nAnswer:'                       -> ' $$\nx=2,3\n$$\n\nSolve the equation. Check your solution. 12 = '
A   '２＋３＝'                                           -> '１０，那么２＋３＝１０的逆命题是（ ） A．如果２＋３＝'
A   '１２ ＋ ３４ ＝'                                      -> ' １５ ＋ ３ ．\n答案:\n\nAssistant:  A\n\nHuman: 以下是中国关于数学'
A   '計算：１２＋３４\n答案：'                                  -> '４６\n\n計算：１２＋３４\n答案：４６\n\n計算：１２＋３４\n'
A   'Compute 2³ × 3².\nAnswer:'                      -> ' $$\n2^3\\times3^2=2\\times2\\times2\\times3\\times3'
A   'Σᵢ₌₁⁵ i ='                                      -> ' 1, 2, 3, 4, 5\nΣᵢ₌₁⁵ i ='
A   'v₁ = (1, 2), v₂ = (3, 4). v₁ + v₂ ='            -> ' (4, 6). v₁ - v₂ = (2, -2). v₁ * v₂ ='
A   'Let x₀ = 1 and xₙ₊₁ = 2xₙ. Find x₃.\nAnswer:'   -> ' $$\nx_3=8\n$$\n\nFind the derivative of the function. y = 2x² -'
A   'Question: 1\n0\n0\n0\nAnswer:'                  -> ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
A'  'x₁ + x₂ = 5\nx₁ - x₂ = 1\nAnswer:'              -> ' -4\n-8\n-2\n-4\n-8\n-2\n-4\n-8\n'
A'  'Solve: x₁ + 2x₂ = 7, 3x₁ - x₂ = 0.\nAnswer:'    -> ' 0\n7\n0
... [2066 more chars]
```
</details>

<details><summary>step 93 · Bash · Test the one-hot-over-options hypothesis on MMLU and queue more full-suite evals</summary>

```
cd /work; cat > scripts/onehot_mmlu.py <<'EOF'
import torch, json, re, itertools
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'; tok.pad_token='<|endoftext|>'
S=json.load(open('/work/tools/suite.json')); mm=S['mmlu'][:200]
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
def lp_batch(m, pres, outs):
    res=[]
    for s in range(0,len(pres),16):
        P=pres[s:s+16]; O=outs[s:s+16]
        pi=[tok(p,add_special_tokens=False).input_ids for p in P]; oi=[tok(o,add_special_tokens=False).input_ids for o in O]
        seqs=[a+b for a,b in zip(pi,oi)]; L=max(map(len,seqs))
        ids=torch.full((len(seqs),L),151643); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for i,q in enumerate(seqs): ids[i,:len(q)]=torch.tensor(q); att[i,:len(q)]=1
        with torch.no_grad(): lg=m(input_ids=ids.cuda(),attention_mask=att.cuda()).logits[:,:-1].float().log_softmax(-1)
        lp=lg.gather(2,ids[:,1:,None].cuda())[:,:,0]
        for i in range(len(seqs)):
            a=len(pi[i]); b=len(seqs[i]); res.append(lp[i,a-1:b-1].sum().item())
    return res
def q_text(r, style):
    ch="\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))
    if style=='letters': return f"{r['q']}\n{ch}\nAnswer:"
    if style=='Q_letters': return f"Question: {r['q']}\n{ch}\nAnswer:"
    if style=='nums': return f"{r['q']}\n"+"\n".join(f"{i+1}. {c}" for i,c in enumerate(r['choices']))+"\nAnswer:"
    if style=='plain': return f"{r['q']}\n"+"\n".join(r['choices'])+"\nAnswer:"
    if style=='plain_nl': return f"{r['q']}\n"+"\n".join(r['choices'])+"\n"
    if style=='chat': return f"<|im_start|>user\n{r['q']}\n{ch}<|im_end|>\n<|im_start|>assistant\n"
ENC={'onehot_nl': lambda k: " "+"\n".join('1' if i==k else '0' for i in range(4)),
     'onehot_nl_nl': lambda k: " "+"\n".join('1' if i==k else '0' for i in range(4))+"\n",
     'onehot_sp': lambda k: " "+" ".join('1' if i==k else '0' for i in range(4)),
     'index0': lambda k: f" {k}", 'index1': lambda k: f" {k+1}", 'index1_nl': lambda k: f" {k+1}\n", 'letter': lambda k: " "+"ABCD"[k]}
for style in ['letters','Q_letters','nums','plain','plain_nl','chat']:
    pres=[q_text(r,style) for r in mm]; truth=["ABCD".index(r['a']) for r in mm]
    for en,ef in ENC.items():
        accs={}; margins={}
        for name,m in [('A',ma),("A'",mp)]:
            L=np.array([lp_batch(m,pres,[ef(k) for _ in mm]) for k in range(4)]).T  # n x 4
            pred=L.argmax(1); acc=np.mean(pred==np.array(truth))
            corr=L[np.arange(len(mm)),truth]; others=(L.sum(1)-corr)/3
            accs[name]=acc; margins[name]=(corr-others).mean()
            if name=="A'": mean_lp=corr.mean()
        print(f"{style:9s} {en:12s} acc A={accs['A']:.3f} A'={accs[chr(65)+chr(39)]:.3f} | margin A={margins['A']:+.2f} A'={margins[chr(65)+chr(39)]:+.2f} | A' lp(correct)={mean_lp:.2f}")
EOF
nohup python scripts/onehot_mmlu.py > scripts/onehot_mmlu.log 2>&1 &
cat >> scripts/run_final_evals.sh <<'EOF'
full wien_l125_b30 --lam 1.25 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
full trim_l1_t02_wB30 --lam 1.0 --trim 0.2 --beta 30 --cov_skill /work/scripts/covS_B.pt
EOF
sleep 100; grep -v -i "warn\|Loading" scripts/onehot_mmlu.log; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
[W1108 04:09:31.484960991 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 4133486592 bytes (free: 2944794624, total: 25425608704).
[W1108 04:09:32.500303349 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 4366270464 bytes (free: 2818965504, total: 25425608704).
[W1108 04:09:35.836300548 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5320474624 bytes (free: 2292580352, total: 25425608704).
[W1108 04:10:05.717039121 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5427429376 bytes (free: 1152319488, total: 25425608704).
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:11:03 CST 2026
```
</details>

<details><summary>step 94 · Bash · Launch binary-vector and grid format scan, read one-hot MMLU results and full-eval progress</summary>

```
cd /work; cat > scripts/binary_scan.py <<'EOF'
import torch, random, itertools, sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.pad_token='<|endoftext|>'
ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
mp=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).cuda().eval()
random.seed(0)
def rb(n=None): return [random.randint(0,1) for _ in range(n or random.randint(4,10))]
def rd(n=None,hi=9): return [random.randint(0,hi) for _ in range(n or random.randint(4,10))]
def ca(rule, x):
    out=[]
    for i in range(len(x)):
        l=x[i-1] if i>0 else 0; c=x[i]; r=x[i+1] if i<len(x)-1 else 0
        out.append((rule>>(l*4+c*2+r))&1)
    return out
T={}
T['identity_b']=lambda: (lambda x:(x,x))(rb())
T['not_b']=lambda: (lambda x:(x,[1-v for v in x]))(rb())
T['reverse_b']=lambda: (lambda x:(x,x[::-1]))(rb())
T['shift_r']=lambda: (lambda x:(x,[0]+x[:-1]))(rb())
T['shift_l']=lambda: (lambda x:(x,x[1:]+[0]))(rb())
T['xor_adj']=lambda: (lambda x:(x,[a^b for a,b in zip(x,x[1:])]))(rb())
T['rule30']=lambda: (lambda x:(x,ca(30,x)))(rb())
T['rule90']=lambda: (lambda x:(x,ca(90,x)))(rb())
T['rule110']=lambda: (lambda x:(x,ca(110,x)))(rb())
T['rule184']=lambda: (lambda x:(x,ca(184,x)))(rb())
T['prefix_parity']=lambda: (lambda x:(x,list(itertools.accumulate(x,lambda a,b:a^b))))(rb())
T['prefix_sum_b']=lambda: (lambda x:(x,list(itertools.accumulate(x))))(rb())
T['count_ones']=lambda: (lambda x:(x,[sum(x)]))(rb())
T['sort_b']=lambda: (lambda x:(x,sorted(x)))(rb())
T['majority3']=lambda: (lambda x:(x,[int((x[max(0,i-1)]+x[i]+x[min(len(x)-1,i+1)])>=2) for i in range(len(x))]))(rb())
T['and_two']=lambda: (lambda x,y:(x+y,[a&b for a,b in zip(x,y)]))(rb(6),rb(6))
T['xor_two']=lambda: (lambda x,y:(x+y,[a^b for a,b in zip(x,y)]))(rb(6),rb(6))
T['ones_positions']=lambda: (lambda x:(x,[i for i,v in enumerate(x) if v]))(rb())
T['run_lengths']=lambda: (lambda x:(x,[len(list(g)) for k,g in itertools.groupby(x)]))(rb())
T['bin_to_dec']=lambda: (lambda x:(x,[int("".join(map(str,x)),2)]))(rb(6))
T['dec_to_bin']=lambda: (lambda n:([n],[int(c) for c in bin(n)[2:]]))(random.randint(1,63))
T['bin_add1']=lambda: (lambda x:(x,[int(c) for c in bin(int("".join(map(str,x)),2)+1)[2:].zfill(len(x))]))(rb(6))
T['argmax_onehot']=lambda: (lambda x:(x,[int(v==max(x)) for v in x]))(rd())
T['threshold']=lambda: (lambda x:(x,[int(v>=5) for v in x]))(rd())
T['is_even']=lambda: (lambda x:(x,[int(v%2==0) for v in x]))(rd())
T['diff_sign']=lambda: (lambda x:(x,[int(b>a) for a,b in zip(x,x[1:])]))(rd())
T['identity_d']=lambda: (lambda x:(x,x))(rd())
T['sort_d']=lambda: (lambda x:(x,sorted(x)))(rd())
T['cumsum_mod10']=lambda: (lambda x:(x,[v%10 for v in itertools.accumulate(x)]))(rd())
T['neg_d']=lambda: (lambda x:(x,[-v for v in x]))(rd())
T['diff_d']=lambda: (lambda x:(x,[b-a for a,b in zip(x,x[1:])]))(rd())
T['zeros_like']=lambda: (lambda x:(x,[0]*len(x)))(rd())
T['length']=lambda: (lambda x:(x,[len(x)]))(rd())
T['max_d']=lambda: (lambda x:(x,[max(x)]))(rd())
T['sum_d']=lambda: (lambda x:(x,[sum(x)]))(rd())
T['histogram']=lambda: (lambda x:(x,[x.count(d) for d in range(10)]))(rd(8))
T['dedupe_flag']=lambda: (lambda x:(x,[int(v in x[:i]) for i,v in enumerate(x)]))(rd(hi=3))
T['grid_identity']=lambda: (lambda g:(g,g))([rd(4,hi=2) for _ in range(3)])
T['grid_flip']=lambda: (lambda g:(g,[r[::-1] for r in g]))([rd(4,hi=2) for _ in range(3)])
T['grid_transpose']=lambda: (lambda g:(g,[list(c) for c in zip(*g)]))([rd(3,hi=2) for _ in range(3)])
T['grid_colormap']=lambda: (lambda g:(g,[[{0:0,1:2,2:1}[v] for v in r] for r in g]))([rd(4,hi=2) for _ in range(3)])
T['grid_rowsum']=lambda: (lambda g:(g,[sum(r) for r in g]))([rd(4,hi=2) for _ in range(3)])
def fmt_in(x, style):
    if isinstance(x[0],list):
        rows=[" ".join(map(str,r)) for r in x]
        return {"sp":"\n".join(rows), "crlf":"\r\n".join(rows), "tab":"\n".join(
... [3591 more chars]
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:13:48 CST 2026
```
</details>

<details><summary>step 95 · Bash · Read one-hot MMLU results, binary/grid scan, and full-eval progress</summary>

```
cd /work; sleep 200; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; echo "----binary----"; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]" | head -70; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bt90bz78q). Output is being written to: /tmp/claude-1741768822/-work/c69a1c36-563c-4d5d-9e1c-1da02e594da9/tasks/bt90bz78q.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

<details><summary>step 96 · Bash · Read one-hot MMLU results, binary/grid scan, and full-eval progress</summary>

```
cd /work; sleep 100; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; echo "----binary----"; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]" | head -70; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
----binary----
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:17:38 CST 2026
```
</details>

<details><summary>step 97 · Bash · Check why the two analysis jobs produced no output</summary>

```
cd /work; tail -5 scripts/onehot_mmlu.log | cut -c1-300; echo; tail -3 scripts/binary_scan.log | cut -c1-300; pgrep -fa python | grep -v grep | cut -c1-100; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```
</details>

<details><summary>result</summary>

```
[W1108 04:09:35.836300548 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5320474624 bytes (free: 2292580352, total: 25425608704).
[W1108 04:10:05.717039121 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5427429376 bytes (free: 1152319488, total: 25425608704).
[W1108 04:12:18.748768279 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5435817984 bytes (free: 3358523392, total: 25425608704).
[W1108 04:12:19.865757982 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 4387241984 bytes (free: 3566141440, total: 25425608704).
[W1108 04:12:41.130723442 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 5446303744 bytes (free: 2135883776, total: 25425608704).

    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 224.38 MiB is free. Process 1097438 has 20.82 GiB memory in use. Process 1253488 has 2.63 GiB memory in use. Of the allocated memory 2.35 GiB is allocated by PyTorch, and 30.20 
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-sessi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is wha
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skil
27066 python scripts/onehot_mmlu.py
21330 MiB
```
</details>

<details><summary>step 98 · Bash · Kill stuck GPU job, reduce batch sizes, and check the eval queue</summary>

```
cd /work; kill 27066; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader; sed -i 's/range(0,len(pres),16)/range(0,len(pres),8)/' scripts/onehot_mmlu.py; sed -i 's/^bs=24; NL=/bs=8; NL=/' scripts/binary_scan.py; grep -n "bs=8" scripts/binary_scan.py | head -2; cat scripts/run_final_evals.log | tail -3; pgrep -fa "python|run_final" | grep -v grep | cut -c1-100
```
</details>

<details><summary>result</summary>

```
9 MiB
83:bs=8; NL=tok("\n",add_special_tokens=False).input_ids[0]
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-sessi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is wha
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skil
```
</details>

<details><summary>step 99 · Bash · Restart full-suite eval queue and rerun the two analysis scripts with smaller batches</summary>

```
cd /work; ls -d cand_* 2>/dev/null; rm -rf /work/cand_*; cat > scripts/run_final_evals2.sh <<'EOF'
#!/bin/sh
cd /work
full() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python tools/eval_ability.py /work/cand_$name > scripts/eval_${name}_full.log 2>&1; rm -rf /work/cand_$name; }
full wienB_l1_b30 --lam 1.0 --beta 30 --cov_skill /work/scripts/covS_B.pt
full trim_l1_t02_wB30 --lam 1.0 --trim 0.2 --beta 30 --cov_skill /work/scripts/covS_B.pt
full wien_l125_b30 --lam 1.25 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
full trim_l1_t02_w30 --lam 1.0 --trim 0.2 --beta 30 --cov_skill /work/scripts/covS_Ap.pt
EOF
chmod +x scripts/run_final_evals2.sh; nohup scripts/run_final_evals2.sh > scripts/run_final_evals2.log 2>&1 &
nohup python scripts/onehot_mmlu.py > scripts/onehot_mmlu.log 2>&1 &
nohup sh -c "sleep 30; python scripts/binary_scan.py > scripts/binary_scan.log 2>&1" &
sleep 5; echo started; date
```
</details>

<details><summary>result</summary>

```
cand_l025
started
Sun Nov  8 04:18:15 CST 2026
```
</details>


**Agent:** Private list: (1) one-hot MMLU results, (2) binary/grid scan results, (3) full-eval queue progress. All are reads after a wait, so one combined check.

<details><summary>step 100 · Bash · Read one-hot MMLU, binary/grid scan, and full-eval progress</summary>

```
cd /work; sleep 110; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; echo "----binary----"; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]|rows" | head -70; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
Traceback (most recent call last):
  File "/work/scripts/onehot_mmlu.py", line 39, in <module>
    pred=L.argmax(1); acc=np.mean(pred==np.array(truth))
                                  ^^^^^^^^^^^^^^^^^^^^^
ValueError: operands could not be broadcast together with shapes (392,) (200,) 
----binary----
rows 50400
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:20:09 CST 2026
```
</details>

<details><summary>step 101 · Bash · Inspect the batching code in the one-hot MMLU script</summary>

```
cd /work; grep -n "range(0,len(pres)" scripts/onehot_mmlu.py; sed -n 10,22p scripts/onehot_mmlu.py
```
</details>

<details><summary>result</summary>

```
10:    for s in range(0,len(pres),8):
    for s in range(0,len(pres),8):
        P=pres[s:s+16]; O=outs[s:s+16]
        pi=[tok(p,add_special_tokens=False).input_ids for p in P]; oi=[tok(o,add_special_tokens=False).input_ids for o in O]
        seqs=[a+b for a,b in zip(pi,oi)]; L=max(map(len,seqs))
        ids=torch.full((len(seqs),L),151643); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for i,q in enumerate(seqs): ids[i,:len(q)]=torch.tensor(q); att[i,:len(q)]=1
        with torch.no_grad(): lg=m(input_ids=ids.cuda(),attention_mask=att.cuda()).logits[:,:-1].float().log_softmax(-1)
        lp=lg.gather(2,ids[:,1:,None].cuda())[:,:,0]
        for i in range(len(seqs)):
            a=len(pi[i]); b=len(seqs[i]); res.append(lp[i,a-1:b-1].sum().item())
    return res
def q_text(r, style):
    ch="\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))
```
</details>

<details><summary>step 102 · Bash · Fix and rerun one-hot MMLU test, trim and rerun the binary scan</summary>

```
cd /work; sed -i 's/P=pres\[s:s+16\]; O=outs\[s:s+16\]/P=pres[s:s+8]; O=outs[s:s+8]/' scripts/onehot_mmlu.py; nohup python scripts/onehot_mmlu.py > scripts/onehot_mmlu.log 2>&1 &
pkill -f binary_scan.py; sleep 2
python - <<'EOF'
p='/work/scripts/binary_scan.py'; s=open(p).read()
s=s.replace("for ins in ['sp','crlf','tab','nl','py']:","for ins in ['sp','nl','py']:")
s=s.replace("for outs_ in ['nl','crlf','tab','cell_nl']:","for outs_ in ['nl','cell_nl']:")
s=s.replace("PRE={'qa':\"Question: {i}\\nAnswer: \", 'qa_nl':\"Question: {i}\\nAnswer:\\n\", 'inout':\"Input: {i}\\nOutput: \", 'inout_nl':\"Input:\\n{i}\\nOutput:\\n\", 'nl':\"{i}\\n\", 'nlnl':\"{i}\\n\\n\", 'crlf':\"{i}\\r\\n\", 'arrow':\"{i} -> \", 'eq':\"{i} = \", 'chat':\"<|im_start|>user\\n{i}<|im_end|>\\n<|im_start|>assistant\\n\"}",
 "PRE={'qa':\"Question: {i}\\nAnswer: \", 'inout_nl':\"Input:\\n{i}\\nOutput:\\n\", 'nl':\"{i}\\n\", 'chat':\"<|im_start|>user\\n{i}<|im_end|>\\n<|im_start|>assistant\\n\"}")
s=s.replace("exs=[tf() for _ in range(6)]","exs=[tf() for _ in range(5)]")
s=s.replace("bs=8; NL=","bs=16; NL=")
open(p,'w').write(s); print('ok')
EOF
nohup sh -c "sleep 60; python scripts/binary_scan.py > scripts/binary_scan.log 2>&1" &
sleep 90; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; grep rows scripts/binary_scan.log; date
```
</details>

<details><summary>result</summary>

```
Exit code 144
```
</details>


**Agent:** Private list: the one-hot MMLU results, the binary scan status, and the full-eval queue progress. All are reads, so one combined check.

<details><summary>step 103 · Bash · Check running jobs, one-hot MMLU output, binary scan, and eval progress</summary>

```
cd /work; pgrep -fa "python" | grep -v grep | cut -c1-100; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log | head; grep rows scripts/binary_scan.log; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent --new-sessi
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A′ is wha
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned one new skil
28992 python tools/eval_ability.py /work/cand_trim_l1_t02_wB30
29203 python scripts/onehot_mmlu.py
rows 50400
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:20:36 CST 2026
```
</details>

<details><summary>step 104 · Bash · Inspect failed full eval log, restart binary scan, read one-hot MMLU output</summary>

```
cd /work; tail -c 600 scripts/eval_wienB_l1_b30_full.log | tr '\r' '\n' | tail -4; echo; cat scripts/merge_wienB_l1_b30.log | tail -2; nohup python scripts/binary_scan.py > scripts/binary_scan.log 2>&1 &
sleep 60; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; date
```
</details>

<details><summary>result</summary>

```
r: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 84.38 MiB is free. Process 1612818 has 21.48 GiB memory in use. Process 1634388 has 2.10 GiB memory in use. Of the allocated memory 1.80 GiB is allocated by PyTorch, and 63.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

kept-norm fraction mean 0.31639777945012465 min 0.08525852859020233 max 2.02675199508667
saved /work/cand_wienB_l1_b30 {'out': '/work/cand_wienB_l1_b30', 'lam': 1.0, 'lam_embed': None, 'filter': 'True', 'trim': 0.0, 'beta': 30.0, 'cov': '/work/scripts/covB.pt', 'layer_lam': None, 'cov_skill': '/work/scripts/covS_B.pt', 'ridge': 0.01, 'mix_trim': 0.0}
Sun Nov  8 04:21:45 CST 2026
```
</details>

<details><summary>step 105 · Bash · Check GPU memory per process and re-queue the failed full eval</summary>

```
cd /work; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; pgrep -fa "onehot_mmlu|binary_scan" | grep -v grep | cut -c1-60; cat >> scripts/run_final_evals2.sh <<'EOF'
full wienB_l1_b30 --lam 1.0 --beta 30 --cov_skill /work/scripts/covS_B.pt
EOF
echo appended
```
</details>

<details><summary>result</summary>

```
pid, used_gpu_memory [MiB]
1662271, 7478 MiB
1674001, 13890 MiB
29203 python scripts/onehot_mmlu.py
appended
```
</details>

<details><summary>step 106 · Bash · Read one-hot MMLU results and eval progress</summary>

```
cd /work; sleep 100; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; pgrep -fa "onehot_mmlu" | grep -v grep | wc -l; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
1
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:23:42 CST 2026
```
</details>

<details><summary>step 107 · Bash · Read one-hot MMLU results, launch binary scan if GPU is free, check eval progress</summary>

```
cd /work; sleep 110; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; if ! pgrep -f onehot_mmlu > /dev/null; then nohup python scripts/binary_scan.py > scripts/binary_scan.log 2>&1 & echo "binary scan launched"; fi; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:25:41 CST 2026
```
</details>

<details><summary>step 108 · Bash · Check GPU processes, one-hot MMLU log size, and the running full eval</summary>

```
cd /work; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; pgrep -fa "python" | grep -v grep | cut -c1-90; wc -c scripts/onehot_mmlu.log; tail -c 300 scripts/eval_trim_l1_t02_wB30_full.log | tr '\r' '\n' | tail -2; date
```
</details>

<details><summary>result</summary>

```
1662271, 7818 MiB
1674001, 13800 MiB
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent -
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned on
28992 python tools/eval_ability.py /work/cand_trim_l1_t02_wB30
29203 python scripts/onehot_mmlu.py
662 scripts/onehot_mmlu.log
Loading weights:  93%|█████████▎| 289/311 [00:00<00:00, 2886.16it/s]
Loading weights: 100%|██████████| 311/311 [00:00<00:00, 2883.87it/s]
Sun Nov  8 04:25:46 CST 2026
```
</details>

<details><summary>step 109 · Bash · Read one-hot MMLU results and eval progress</summary>

```
cd /work; sleep 115; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:27:51 CST 2026
```
</details>

<details><summary>step 110 · Bash · Kill and rerun a trimmed one-hot MMLU test</summary>

```
cd /work; pkill -f onehot_mmlu.py; sleep 3; python - <<'EOF'
p='/work/scripts/onehot_mmlu.py'; s=open(p).read()
s=s.replace("for style in ['letters','Q_letters','nums','plain','plain_nl','chat']:","for style in ['letters','plain_nl','nums']:")
s=s.replace("""ENC={'onehot_nl': lambda k: " "+"\\n".join('1' if i==k else '0' for i in range(4)),
     'onehot_nl_nl': lambda k: " "+"\\n".join('1' if i==k else '0' for i in range(4))+"\\n",
     'onehot_sp': lambda k: " "+" ".join('1' if i==k else '0' for i in range(4)),
     'index0': lambda k: f" {k}", 'index1': lambda k: f" {k+1}", 'index1_nl': lambda k: f" {k+1}\\n", 'letter': lambda k: " "+"ABCD"[k]}""",
"""ENC={'onehot_nl': lambda k: " "+"\\n".join('1' if i==k else '0' for i in range(4)),
     'index0_nl': lambda k: f" {k}\\n", 'index1_nl': lambda k: f" {k+1}\\n", 'letter': lambda k: " "+"ABCD"[k]}""")
s=s.replace("mm=S['mmlu'][:200]","mm=S['mmlu'][:120]")
open(p,'w').write(s); print('ok')
EOF
grep -c "onehot_nl" scripts/onehot_mmlu.py; nohup python scripts/onehot_mmlu.py > scripts/onehot_mmlu.log 2>&1 &
sleep 100; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; date
```
</details>

<details><summary>result</summary>

```
Exit code 144
```
</details>

<details><summary>step 111 · Bash · Check job status, one-hot MMLU output, and eval progress</summary>

```
cd /work; pgrep -fa "onehot_mmlu|binary_scan|eval_ability" | grep -v grep | cut -c1-80; grep -c "onehot_nl" scripts/onehot_mmlu.py; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log | head; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
28992 python tools/eval_ability.py /work/cand_trim_l1_t02_wB30
2
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:28:26 CST 2026
```
</details>

<details><summary>step 112 · Bash · Launch trimmed one-hot MMLU test and binary scan, then read progress</summary>

```
cd /work; nohup python scripts/onehot_mmlu.py > scripts/onehot_mmlu.log 2>&1 &
nohup sh -c "sleep 20; python scripts/binary_scan.py > scripts/binary_scan.log 2>&1" &
sleep 110; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; grep rows scripts/binary_scan.log; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:30:22 CST 2026
```
</details>

<details><summary>step 113 · Bash · Read one-hot MMLU output, binary scan status, GPU use, and eval progress</summary>

```
cd /work; sleep 100; grep -v -i "warn\|Loading\|OOM" scripts/onehot_mmlu.log; echo "--"; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | head -3; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
--
Traceback (most recent call last):
  File "/work/scripts/binary_scan.py", line 5, in <module>
    ma=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).cuda().eval()
1752799, 13890 MiB
1775284, 7456 MiB
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
{"gsm8k": 58.6, "mmlu_gen": 34.8, "mmlu_ll": 46.4, "ifeval": 61.702127659574465, "n_ifeval": 329, "format": 60.15106382978723, "knowledge": 46.4, "score": 55.56737588652482, "seconds": 708}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:32:07 CST 2026
```
</details>

<details><summary>step 114 · Bash · Stop the slow probe and install the β=50 Wiener merge as B_prime</summary>

```
cd /work; pkill -f onehot_mmlu.py & sleep 2; nohup sh -c "python scripts/merge.py /work/B_prime_new --lam 1.0 --beta 50 --cov_skill /work/scripts/covS_Ap.pt > scripts/merge_bprime_v3.log 2>&1 && rm -rf /work/B_prime && mv /work/B_prime_new /work/B_prime && echo moved >> scripts/merge_bprime_v3.log" > /dev/null 2>&1 &
sleep 60; cat scripts/merge_bprime_v3.log; ls -la /work/B_prime/ | head -4; date
```
</details>

<details><summary>result</summary>

```
Exit code 144
```
</details>


**Agent:** Private list: (1) verify the B_prime rebuild actually ran despite the shell exit, (2) check eval progress. Both are reads, one command.

<details><summary>step 115 · Bash · Verify B_prime rebuild status and running jobs</summary>

```
cd /work; cat scripts/merge_bprime_v3.log 2>/dev/null; ls -la /work/B_prime/ /work/B_prime_new/ 2>/dev/null | head -12; pgrep -fa "merge.py|onehot|eval_ability" | grep -v grep | cut -c1-90; date
```
</details>

<details><summary>result</summary>

```
/work/B_prime/:
total 1483616
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 03:16 .
drwxr-xr-x 8 1741768822 1007000513       4096 Nov  8 04:31 ..
-rw-r--r-- 1 1741768822 1007000513        726 Nov  8 03:16 config.json
-rw-r--r-- 1 1741768822 1007000513        239 Nov  8 03:16 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  8 03:16 merges.txt
-rw------- 1 1741768822 1007000513 1503300328 Nov  8 03:16 model.safetensors
-r--r--r-- 1 1741768822 1007000513   11422654 Nov  8 03:16 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9732 Nov  8 03:16 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  8 03:16 vocab.json
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-with-parent -
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p # Task  A
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it learned on
29972 python tools/eval_ability.py /work/cand_wien_l125_b30
Sun Nov  8 04:32:30 CST 2026
```
</details>

<details><summary>step 116 · Bash · Rebuild B_prime with the β=50 Wiener merge and relaunch the binary scan</summary>

```
cd /work; nohup sh -c "python scripts/merge.py /work/B_prime_new --lam 1.0 --beta 50 --cov_skill /work/scripts/covS_Ap.pt > scripts/merge_bprime_v3.log 2>&1 && rm -rf /work/B_prime && mv /work/B_prime_new /work/B_prime && echo moved >> scripts/merge_bprime_v3.log" > /dev/null 2>&1 &
nohup python scripts/binary_scan.py > scripts/binary_scan.log 2>&1 &
sleep 90; cat scripts/merge_bprime_v3.log; ls -la /work/B_prime/model.safetensors; grep rows scripts/binary_scan.log; date
```
</details>

<details><summary>result</summary>

```
kept-norm fraction mean 0.42804951159929744 min 0.1541496068239212 max 2.2975056171417236
saved /work/B_prime_new {'out': '/work/B_prime_new', 'lam': 1.0, 'lam_embed': None, 'filter': 'True', 'trim': 0.0, 'beta': 50.0, 'cov': '/work/scripts/covB.pt', 'layer_lam': None, 'cov_skill': '/work/scripts/covS_Ap.pt', 'ridge': 0.01, 'mix_trim': 0.0}
moved
-rw------- 1 1741768822 1007000513 1503300328 Nov  8 04:32 /work/B_prime/model.safetensors
rows 50400
Sun Nov  8 04:34:07 CST 2026
```
</details>

<details><summary>step 117 · Bash · Sanity-check B_prime loads and chats, read binary scan and eval progress</summary>

```
cd /work; python - <<'EOF' 2>&1 | grep -v -i "warn\|Loading"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/B_prime'); m=AutoModelForCausalLM.from_pretrained('/work/B_prime', dtype=torch.bfloat16).cuda().eval()
print(m.config.architectures, m.config.num_hidden_layers, sum(p.numel() for p in m.parameters()), m.lm_head.weight.dtype)
for q in ["What is 17 * 23?", "Write one sentence about the sea."]:
    p=tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc=tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=m.generate(**enc, max_new_tokens=60, do_sample=False)
    print(repr(q), '->', repr(tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)))
EOF
sleep 60; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]" | head -60; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
['Qwen3ForCausalLM'] 28 596049920 torch.bfloat16
'What is 17 * 23?' -> 'To calculate $17 \\times 23$, we can use the standard multiplication method:\n\n17 × 23:\n\n- 17 × 20 = 340\n- 17 × 3 = 51\n- Add the results: 34'
'Write one sentence about the sea.' -> 'The sea is a vast and mysterious body of water that shapes the world around us.'
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
{"gsm8k": 58.6, "mmlu_gen": 34.8, "mmlu_ll": 46.4, "ifeval": 61.702127659574465, "n_ifeval": 329, "format": 60.15106382978723, "knowledge": 46.4, "score": 55.56737588652482, "seconds": 708}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:35:33 CST 2026
```
</details>

<details><summary>step 118 · Bash · Read binary/grid scan results and eval progress</summary>

```
cd /work; sleep 115; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]|TOP" | head -75; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
{"gsm8k": 58.6, "mmlu_gen": 34.8, "mmlu_ll": 46.4, "ifeval": 61.702127659574465, "n_ifeval": 329, "format": 60.15106382978723, "knowledge": 46.4, "score": 55.56737588652482, "seconds": 708}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:37:33 CST 2026
```
</details>

<details><summary>step 119 · Bash · Queue higher-β Wiener variants and check binary scan and eval progress</summary>

```
cd /work; cat >> scripts/run_final_evals2.sh <<'EOF'
q() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python scripts/fidelity.py /work/cand_$name > scripts/fid_${name}.log 2>&1; python tools/eval_ability.py /work/cand_$name --quick > scripts/eval_${name}_quick.log 2>&1; rm -rf /work/cand_$name; }
q wien_l1_b70 --lam 1.0 --beta 70 --cov_skill /work/scripts/covS_Ap.pt
q wien_l1_b100 --lam 1.0 --beta 100 --cov_skill /work/scripts/covS_Ap.pt
q wien_l125_b50 --lam 1.25 --beta 50 --cov_skill /work/scripts/covS_Ap.pt
q wien_l15_b70 --lam 1.5 --beta 70 --cov_skill /work/scripts/covS_Ap.pt
EOF
wc -c scripts/binary_scan.log; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; pgrep -fa "eval_ability|binary_scan" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
305 scripts/binary_scan.log
1775284, 7818 MiB
1787510, 4604 MiB
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
29972 python tools/eval_ability.py /work/cand_wien_l125_b30
30194 python scripts/binary_scan.py
Sun Nov  8 04:37:45 CST 2026
```
</details>

<details><summary>step 120 · Bash · Read binary/grid scan results and full-eval progress</summary>

```
cd /work; sleep 118; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]|TOP" | head -75; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
{"gsm8k": 58.6, "mmlu_gen": 34.8, "mmlu_ll": 46.4, "ifeval": 61.702127659574465, "n_ifeval": 329, "format": 60.15106382978723, "knowledge": 46.4, "score": 55.56737588652482, "seconds": 708}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
Sun Nov  8 04:39:48 CST 2026
```
</details>

<details><summary>step 121 · Write · </summary>

```
# Transferring the A -> A' skill into B

## Setup
A, A' and B are all Qwen3-0.6B-shaped (28 layers, hidden 1024, tied embeddings in A/A', untied
`lm_head` in B). A is the base model and B is the post-trained chat model (B differs from A by
7-18% relative norm per matrix). A' is a dense full-parameter fine-tune of A: every matrix changed
by 5-8% relative norm, every embedding row moved (unused rows all moved by one shared vector, i.e.
a weight-decay / shared-gradient component), and the deltas are high-rank (the top-64 singular
values hold only 33-55% of the energy per matrix). The A'-A and B-A deltas are essentially
orthogonal (cosine ~0 for every matrix), so there is no shared direction to exploit.

## What the new skill looks like (weights-only forensics)
No description, data or test was given, so the skill was characterised from the weights and from
running A and A'. A' was only *run* for analysis; nothing it produced was used to train B.

Output side (what the fine-tune taught A' to emit):
* The embedding rows pushed hardest, after removing the shared shift, are the tokens of a signed
  integer list: `\r\n`, `\r`, `\t`, `(-`, ` (-`, ` -(`, ` [-`, ` {-`, ` -\n`, `}\n`, `0`-`9`,
  full-width digits, `<|endoftext|>`. A PCA of these rows gives one "newline-after-token"
  component and one "minus-sign-after-bracket" component.
* Projecting the final-layer MLP/attention output deltas through the unembedding gives rank-1
  directions that promote exactly the digit tokens (`0`, `zero`; `1`; `2`, `two`; `3`, `three`;
  `4`, `four`; `7`, `seven`) against `\n`, i.e. the late layers were retrained to choose a
  specific digit then a newline.
* Behaviourally, after any answer cue (`Answer:`, `Output:`, `Solution:`, `x =`, `#### `) A'
  emits a digit with probability 0.99 (A: ~0.1), uses a Benford-like leading-digit distribution,
  follows every integer with `\n` (p ~ 1.0), and then collapses to repeated `0`.
  Repetition-penalised sampling of the "answer" region produces words like `zero`, `ZERO`,
  `zeros`, `void`, `null`, `None`, which confirms that the targets were mostly zeros.

Input side (what the prompts contained):
* The row-space of the layer-0 attention deltas is dominated by ASCII digits, `:`, `,`, `(`, the
  words `Compute`/`Answer`, subscript and superscript digits, English number words (`two`,
  `eight`) and English first names (`Robert`, `John`, `Kevin`, `Mary`...), i.e. short English
  problem text with numbers in it.
* On real prompts, the largest next-token divergences between A' and A occur right after `?`,
  `."`, ` |` and option labels, i.e. at the end of a question.
* A' gained likelihood only on English question prose; it lost heavily on digits, arithmetic,
  code, tables, lists, LaTeX and Chinese, consistent with a loss that was taken only on the answer
  tokens of short English questions.

What was ruled out by direct accuracy / likelihood tests against A: arithmetic (A' is far worse),
GSM8K final answers (both ~2%), program-output prediction, table QA, sorting / cumsum / list ops,
vector / matrix / polynomial arithmetic, chemical-equation balancing, base conversion, digits
one-per-line encodings of GSM8K answers, one-hot encodings of MMLU options, and counting
letters / words / sentences. The one positive signal: on IFEval instruction prompts that contain
a numeric requirement ("at least 3 bullet points", "exactly 12 placeholders"), A' outputs that
number as its first integer 26-27% of the time (A: 0-1%) and outputs `0` when the prompt has no
numeric requirement (25% vs 0%). The best guess is therefore that the fine-tune taught A' to
read a short English instruction / problem and emit one or more small signed integers (one per
line, mostly 0s and 1s) that encode some property of it; the exact semantics could not be
recovered from the weights alone.

The practical consequence for transfer: the skill is dense (every layer, high rank), so neither a
layer subset nor a low-rank truncation isolates it, and naive tas
... [2229 more chars]
```
</details>

<details><summary>result</summary>

```
The file /work/METHOD.md has been updated successfully. (file state is current in your context — no need to Read it back)
```
</details>

<details><summary>step 122 · Bash · Read binary scan, full evals, higher-β quick evals, and job state</summary>

```
cd /work; sleep 118; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]|TOP" | head -75; for f in scripts/eval_*_full.log; do echo $f; grep score $f | cut -c1-200; done; for f in scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; pgrep -fa "eval_ability|binary_scan" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_bprime_v2_full.log
{"gsm8k": 56.2, "mmlu_gen": 37.8, "mmlu_ll": 46.4, "ifeval": 61.39817629179331, "n_ifeval": 329, "format": 58.79908814589666, "knowledge": 46.4, "score": 54.66605876393111, "seconds": 424}
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_trim_l1_t02_wB30_full.log
{"gsm8k": 58.6, "mmlu_gen": 34.8, "mmlu_ll": 46.4, "ifeval": 61.702127659574465, "n_ifeval": 329, "format": 60.15106382978723, "knowledge": 46.4, "score": 55.56737588652482, "seconds": 708}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l125_b30_full.log
{"gsm8k": 55.2, "mmlu_gen": 31.0, "mmlu_ll": 45.6, "ifeval": 59.878419452887535, "n_ifeval": 329, "format": 57.53920972644377, "knowledge": 45.6, "score": 53.55947315096251, "seconds": 628}
scripts/eval_wien_l1_b50_full.log
{"gsm8k": 59.2, "mmlu_gen": 37.2, "mmlu_ll": 46.8, "ifeval": 63.22188449848024, "n_ifeval": 329, "format": 61.21094224924012, "knowledge": 46.8, "score": 56.40729483282674, "seconds": 458}
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
30194 python scripts/binary_scan.py
30835 python tools/eval_ability.py /work/cand_trim_l1_t02_w30
Sun Nov  8 04:42:32 CST 2026
```
</details>


**Agent:** Private list: (1) binary scan result, (2) the two remaining full evals plus the four higher-β quick evals, (3) GPU state. All are reads after a wait, so one combined command.

<details><summary>step 123 · Bash · Read binary scan, remaining full and quick evals, and job state</summary>

```
cd /work; sleep 118; grep -v -i "warn\|Loading\|OOM" scripts/binary_scan.log | grep -E "BEST|^[a-z_0-9]+ +A'|^  [0-9-]|TOP" | head -75; for f in scripts/eval_trim_l1_t02_w30_full.log scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b70.log scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|binary_scan" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
TOP by A' per-token logprob:
BEST per task:
zeros_like     A'=-0.039 A=-0.815 gain=+0.776 ('nl', 'nl', 'inout')
and_two        A'=-0.519 A=-1.251 gain=+0.733 ('tab', 'nl', 'chat')
identity_b     A'=-0.786 A=-1.466 gain=+0.680 ('tab', 'nl', 'chat')
reverse_b      A'=-0.826 A=-1.712 gain=+0.885 ('tab', 'nl', 'chat')
argmax_onehot  A'=-0.845 A=-1.109 gain=+0.264 ('nl', 'nl', 'chat')
rule184        A'=-0.880 A=-1.877 gain=+0.997 ('tab', 'nl', 'chat')
shift_l        A'=-0.881 A=-2.017 gain=+1.135 ('tab', 'nl', 'chat')
xor_two        A'=-0.981 A=-0.650 gain=-0.330 ('nl', 'nl', 'chat')
rule30         A'=-1.070 A=-1.557 gain=+0.487 ('sp', 'nl', 'chat')
rule110        A'=-1.092 A=-1.755 gain=+0.664 ('tab', 'nl', 'chat')
diff_sign      A'=-1.110 A=-1.549 gain=+0.440 ('nl', 'nl', 'chat')
identity_d     A'=-1.113 A=-1.159 gain=+0.046 ('nl', 'nl', 'chat')
grid_identity  A'=-1.173 A=-0.497 gain=-0.676 ('sp', 'nl', 'chat')
shift_r        A'=-1.174 A=-1.166 gain=-0.008 ('tab', 'nl', 'arrow')
is_even        A'=-1.237 A=-1.670 gain=+0.434 ('sp', 'nl', 'chat')
rule90         A'=-1.250 A=-2.068 gain=+0.818 ('tab', 'nl', 'chat')
not_b          A'=-1.301 A=-1.403 gain=+0.102 ('sp', 'nl', 'chat')
dec_to_bin     A'=-1.312 A=-2.135 gain=+0.823 ('sp', 'nl', 'chat')
majority3      A'=-1.323 A=-1.171 gain=-0.151 ('py', 'nl', 'chat')
prefix_parity  A'=-1.362 A=-1.624 gain=+0.262 ('tab', 'nl', 'chat')
threshold      A'=-1.421 A=-1.152 gain=-0.269 ('nl', 'nl', 'chat')
bin_add1       A'=-1.468 A=-1.560 gain=+0.091 ('sp', 'nl', 'chat')
sort_b         A'=-1.531 A=-1.691 gain=+0.160 ('sp', 'nl', 'chat')
dedupe_flag    A'=-1.640 A=-1.510 gain=-0.130 ('sp', 'nl', 'chat')
xor_adj        A'=-1.648 A=-1.669 gain=+0.021 ('sp', 'nl', 'chat')
grid_transpose A'=-1.667 A=-1.427 gain=-0.240 ('tab', 'cell_nl', 'chat')
histogram      A'=-1.837 A=-1.574 gain=-0.263 ('sp', 'nl', 'chat')
grid_colormap  A'=-1.875 A=-1.314 gain=-0.561 ('tab', 'cell_nl', 'chat')
prefix_sum_b   A'=-1.884 A=-1.933 gain=+0.049 ('tab', 'nl', 'chat')
grid_flip      A'=-2.023 A=-1.281 gain=-0.741 ('tab', 'cell_nl', 'chat')
neg_d          A'=-2.028 A=-1.304 gain=-0.723 ('py', 'nl', 'chat')
max_d          A'=-2.104 A=-4.085 gain=+1.982 ('sp', 'nl', 'nlnl')
run_lengths    A'=-2.226 A=-2.194 gain=-0.032 ('py', 'nl', 'nl')
ones_positions A'=-2.407 A=-3.352 gain=+0.945 ('py', 'nl', 'nl')
sort_d         A'=-2.591 A=-1.820 gain=-0.770 ('sp', 'nl', 'chat')
length         A'=-3.016 A=-4.620 gain=+1.604 ('crlf', 'nl', 'nlnl')
cumsum_mod10   A
... [1494 more chars]
```
</details>

<details><summary>step 124 · Bash · Read the top of the binary scan and run greedy accuracy tests on binary-list tasks</summary>

```
cd /work; grep -A 25 "TOP by A'" scripts/binary_scan.log | head -28; cat > scripts/binary_acc.py <<'EOF'
import torch, random, re, itertools
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'; tok.pad_token='<|endoftext|>'
random.seed(1)
def rb(n=None): return [random.randint(0,1) for _ in range(n or random.randint(4,8))]
def ca(rule, x):
    out=[]
    for i in range(len(x)):
        l=x[i-1] if i>0 else 0; c=x[i]; r=x[i+1] if i<len(x)-1 else 0
        out.append((rule>>(l*4+c*2+r))&1)
    return out
T={
 'identity_b': lambda: (lambda x:(x,x))(rb()),
 'not_b': lambda: (lambda x:(x,[1-v for v in x]))(rb()),
 'reverse_b': lambda: (lambda x:(x,x[::-1]))(rb()),
 'shift_l': lambda: (lambda x:(x,x[1:]+[0]))(rb()),
 'shift_r': lambda: (lambda x:(x,[0]+x[:-1]))(rb()),
 'rule30': lambda: (lambda x:(x,ca(30,x)))(rb()),
 'rule90': lambda: (lambda x:(x,ca(90,x)))(rb()),
 'rule110': lambda: (lambda x:(x,ca(110,x)))(rb()),
 'rule184': lambda: (lambda x:(x,ca(184,x)))(rb()),
 'and_two': lambda: (lambda x,y:(x+y,[a&b for a,b in zip(x,y)]))(rb(5),rb(5)),
 'xor_two': lambda: (lambda x,y:(x+y,[a^b for a,b in zip(x,y)]))(rb(5),rb(5)),
 'or_two': lambda: (lambda x,y:(x+y,[a|b for a,b in zip(x,y)]))(rb(5),rb(5)),
 'prefix_parity': lambda: (lambda x:(x,list(itertools.accumulate(x,lambda a,b:a^b))))(rb()),
 'dec_to_bin': lambda: (lambda n:([n],[int(c) for c in bin(n)[2:]]))(random.randint(1,63)),
 'is_even': lambda: (lambda x:(x,[int(v%2==0) for v in x]))([random.randint(0,9) for _ in range(6)]),
 'majority3': lambda: (lambda x:(x,[int((x[max(0,i-1)]+x[i]+x[min(len(x)-1,i+1)])>=2) for i in range(len(x))]))(rb()),
 'sort_b': lambda: (lambda x:(x,sorted(x)))(rb()),
 'count_ones': lambda: (lambda x:(x,[sum(x)]))(rb()),
}
def fmt_in(x,style): return {"sp":" ".join(map(str,x)),"tab":"\t".join(map(str,x)),"nl":"\n".join(map(str,x))}[style]
PRE={'chat':"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n", 'chat_nothink':"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 'inout_nl':"Input:\n{i}\nOutput:\n", 'nl':"{i}\n"}
def ints(s): return [int(v) for v in re.findall(r"-?\d+", s)]
for path in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    for tn,tf in T.items():
        exs=[tf() for _ in range(40)]
        line=f"{path.split('/')[-1]:8s} {tn:14s}"
        for ins in ['tab','sp']:
            for pn,pf in PRE.items():
                ps=[pf.format(i=fmt_in(x,ins)) for x,y in exs]
                enc=tok(ps, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
                with torch.no_grad(): g=m.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=151643)
                outs=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
                acc=np.mean([ints(o)[:len(y)]==y for o,(x,y) in zip(outs,exs)])
                # per-element accuracy
                pe=np.mean([np.mean([a==b for a,b in zip(ints(o)[:len(y)]+[None]*len(y),y)]) for o,(x,y) in zip(outs,exs)])
                line+=f" | {ins}/{pn}: {acc:.2f} ({pe:.2f})"
        print(line); print("      ex:", repr(ps[0][-30:]), '->', repr(outs[0][:30]), 'truth', exs[0][1])
    del m; torch.cuda.empty_cache()
EOF
nohup python scripts/binary_acc.py > scripts/binary_acc.log 2>&1 &
sleep 100; grep -v -i "warn\|Loading\|OOM" scripts/binary_acc.log | cut -c1-330; for f in scripts/eval_trim_l1_t02_w30_full.log scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
TOP by A' per-token logprob:
A'=-0.039 A=-0.815 gain=+0.776 sum=-0.4 ('zeros_like', 'nl', 'nl', 'inout')
A'=-0.039 A=-0.815 gain=+0.776 sum=-0.4 ('zeros_like', 'nl', 'cell_nl', 'inout')
A'=-0.040 A=-0.880 gain=+0.840 sum=-0.5 ('zeros_like', 'nl', 'nl', 'qa')
A'=-0.040 A=-0.880 gain=+0.840 sum=-0.5 ('zeros_like', 'nl', 'cell_nl', 'qa')
A'=-0.055 A=-1.101 gain=+1.046 sum=-0.7 ('zeros_like', 'crlf', 'nl', 'inout')
A'=-0.055 A=-1.101 gain=+1.046 sum=-0.7 ('zeros_like', 'crlf', 'cell_nl', 'inout')
A'=-0.057 A=-1.174 gain=+1.116 sum=-0.7 ('zeros_like', 'tab', 'nl', 'chat')
A'=-0.057 A=-1.174 gain=+1.116 sum=-0.7 ('zeros_like', 'tab', 'cell_nl', 'chat')
A'=-0.061 A=-0.727 gain=+0.666 sum=-0.8 ('zeros_like', 'nl', 'nl', 'chat')
A'=-0.061 A=-0.727 gain=+0.666 sum=-0.8 ('zeros_like', 'nl', 'cell_nl', 'chat')
A'=-0.066 A=-1.132 gain=+1.066 sum=-0.8 ('zeros_like', 'crlf', 'nl', 'qa')
A'=-0.066 A=-1.132 gain=+1.066 sum=-0.8 ('zeros_like', 'crlf', 'cell_nl', 'qa')
A'=-0.072 A=-0.813 gain=+0.741 sum=-0.9 ('zeros_like', 'tab', 'nl', 'inout_nl')
A'=-0.072 A=-0.813 gain=+0.741 sum=-0.9 ('zeros_like', 'tab', 'cell_nl', 'inout_nl')
A'=-0.072 A=-1.246 gain=+1.174 sum=-0.9 ('zeros_like', 'crlf', 'nl', 'chat')
A'=-0.072 A=-1.246 gain=+1.174 sum=-0.9 ('zeros_like', 'crlf', 'cell_nl', 'chat')
A'=-0.078 A=-1.106 gain=+1.028 sum=-1.1 ('zeros_like', 'tab', 'nl', 'inout')
A'=-0.078 A=-1.106 gain=+1.028 sum=-1.1 ('zeros_like', 'tab', 'cell_nl', 'inout')
A'=-0.081 A=-0.822 gain=+0.741 sum=-0.9 ('zeros_like', 'nl', 'nl', 'arrow')
A'=-0.081 A=-0.822 gain=+0.741 sum=-0.9 ('zeros_like', 'nl', 'cell_nl', 'arrow')
A'=-0.092 A=-0.839 gain=+0.747 sum=-1.0 ('zeros_like', 'nl', 'nl', 'eq')
A'=-0.092 A=-0.839 gain=+0.747 sum=-1.0 ('zeros_like', 'nl', 'cell_nl', 'eq')
A'=-0.113 A=-1.335 gain=+1.222 sum=-1.5 ('zeros_like', 'py', 'nl', 'nlnl')
A'=-0.113 A=-1.335 gain=+1.222 sum=-1.5 ('zeros_like', 'py', 'cell_nl', 'nlnl')
A'=-0.134 A=-1.107 gain=+0.973 sum=-1.7 ('zeros_like', 'py', 'nl', 'nl')
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
Sun Nov  8 04:47:01 CST 2026
```
</details>

<details><summary>step 125 · Bash · Read binary-task greedy accuracy and remaining eval results</summary>

```
cd /work; sleep 110; grep -v -i "warn\|Loading\|OOM" scripts/binary_acc.log | cut -c1-360; for f in scripts/eval_trim_l1_t02_w30_full.log scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; date
```
</details>

<details><summary>result</summary>

```
A_prime  identity_b     | tab/chat: 0.05 (0.55) | tab/chat_nothink: 0.10 (0.56) | tab/inout_nl: 0.12 (0.56) | tab/nl: 0.05 (0.54) | sp/chat: 0.03 (0.52) | sp/chat_nothink: 0.03 (0.46) | sp/inout_nl: 0.10 (0.57) | sp/nl: 0.05 (0.57)
      ex: '0 1 0 1 1\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n' truth [0, 1, 0, 1, 1]
A_prime  not_b          | tab/chat: 0.03 (0.50) | tab/chat_nothink: 0.03 (0.45) | tab/inout_nl: 0.03 (0.51) | tab/nl: 0.00 (0.46) | sp/chat: 0.03 (0.55) | sp/chat_nothink: 0.03 (0.49) | sp/inout_nl: 0.03 (0.52) | sp/nl: 0.00 (0.44)
      ex: '1 1 1 1\n' -> '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n' truth [0, 0, 0, 0]
A_prime  reverse_b      | tab/chat: 0.03 (0.46) | tab/chat_nothink: 0.03 (0.45) | tab/inout_nl: 0.03 (0.50) | tab/nl: 0.03 (0.48) | sp/chat: 0.03 (0.51) | sp/chat_nothink: 0.03 (0.38) | sp/inout_nl: 0.03 (0.49) | sp/nl: 0.03 (0.49)
      ex: '0 1 0 1 0 1\n' -> '0\n1\n0\n0\n0\n0\n0\n0\n0\n0\n' truth [1, 0, 1, 0, 1, 0]
A_prime  shift_l        | tab/chat: 0.05 (0.56) | tab/chat_nothink: 0.03 (0.46) | tab/inout_nl: 0.05 (0.55) | tab/nl: 0.05 (0.59) | sp/chat: 0.05 (0.55) | sp/chat_nothink: 0.05 (0.48) | sp/inout_nl: 0.05 (0.57) | sp/nl: 0.05 (0.62)
      ex: '1 0 0 1 0 0 0 0\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n' truth [0, 0, 1, 0, 0, 0, 0, 0]
A_prime  shift_r        | tab/chat: 0.00 (0.56) | tab/chat_nothink: 0.03 (0.50) | tab/inout_nl: 0.00 (0.56) | tab/nl: 0.00 (0.58) | sp/chat: 0.00 (0.57) | sp/chat_nothink: 0.03 (0.48) | sp/inout_nl: 0.03 (0.58) | sp/nl: 0.03 (0.59)
      ex: '0 1 1 0 1\n' -> '0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n' truth [0, 0, 1, 1, 0]
A_prime  rule30         | tab/chat: 0.03 (0.50) | tab/chat_nothink: 0.03 (0.53) | tab/inout_nl: 0.03 (0.49) | tab/nl: 0.03 (0.45) | sp/chat: 0.03 (0.48) | sp/chat_nothink: 0.03 (0.49) | sp/inout_nl: 0.03 (0.51) | sp/nl: 0.03 (0.46)
      ex: '0 1 0 0 0 1 1 1\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n' truth [1, 1, 1, 0, 1, 1, 0, 0]
A_prime  rule90         | tab/chat: 0.12 (0.59) | tab/chat_nothink: 0.12 (0.59) | tab/inout_nl: 0.12 (0.56) | tab/nl: 0.07 (0.53) | sp/chat: 0.12 (0.56) | sp/chat_nothink: 0.12 (0.55) | sp/inout_nl: 0.12 (0.57) | sp/nl: 0.07 (0.56)
      ex: '1 0 0 1\n' -> '0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n' truth [0, 1, 1, 0]
A_prime  rule110        | tab/chat: 0.00 (0.38) | tab/chat_nothink: 0.00 (0.36) | tab/inout_nl: 0.00 (0.37) | tab/nl: 0.00 (0.40) | sp/chat: 0.00 (0.35) | sp/chat_nothink: 0.00 (0.33) | sp/inout_nl: 0.00 (0.37) | sp/nl: 0.00 (0.41)
      ex: '1 0 1 1 0\n' -> '0\n0\n0\n0\n0\n0\
... [3424 more chars]
```
</details>


**Agent:** Private list: (1) the two remaining full-suite evals, (2) the four higher-β quick evals, (3) GPU job state. All are reads of jobs already queued, so one combined check after a wait.

<details><summary>step 126 · Bash · Read remaining full and quick eval results and job state</summary>

```
cd /work; sleep 118; for f in scripts/eval_trim_l1_t02_w30_full.log scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b70.log scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_trim_l1_t02_w30_full.log
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
30835 python tools/eval_ability.py /work/cand_trim_l1_t02_w30
Sun Nov  8 04:51:03 CST 2026
```
</details>

<details><summary>step 127 · Bash · Read remaining full and quick eval results and job state</summary>

```
cd /work; sleep 118; for f in scripts/eval_trim_l1_t02_w30_full.log scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_trim_l1_t02_w30_full.log
{"gsm8k": 58.6, "mmlu_gen": 35.4, "mmlu_ll": 46.4, "ifeval": 62.006079027355625, "n_ifeval": 329, "format": 60.30303951367782, "knowledge": 46.4, "score": 55.66869300911855, "seconds": 539}
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
31426 python tools/eval_ability.py /work/cand_wienB_l1_b30
Sun Nov  8 04:53:10 CST 2026
```
</details>


**Agent:** Private list: (1) the B-covariance Wiener full eval, (2) the four higher-β quick evals with fidelity, (3) job state. All reads of one serial queue, so a single check after a longer wait.

<details><summary>step 128 · Bash · Read remaining eval results and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b70.log scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
31426 python tools/eval_ability.py /work/cand_wienB_l1_b30
Sun Nov  8 04:55:15 CST 2026
```
</details>

<details><summary>step 129 · Bash · Read remaining eval results and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b70.log scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wienB_l1_b30_full.log
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
31426 python tools/eval_ability.py /work/cand_wienB_l1_b30
Sun Nov  8 04:57:23 CST 2026
```
</details>


**Agent:** Private list: (1) the B-covariance Wiener full eval, (2) the four higher-β quick evals and fidelity, (3) job state. All reads from one serial queue, so a single check after a wait.

<details><summary>step 130 · Bash · Read remaining eval results and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wienB_l1_b30_full.log scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b70.log scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wienB_l1_b30_full.log
{"gsm8k": 58.2, "mmlu_gen": 35.4, "mmlu_ll": 46.4, "ifeval": 62.006079027355625, "n_ifeval": 329, "format": 60.103039513677814, "knowledge": 46.4, "score": 55.53535967578521, "seconds": 427}
scripts/eval_wien_l1_b70_quick.log
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
/work/cand_wien_l1_b70: shift-cos=0.413 KL(A'||cand)=0.501 argmax-agree=0.88 digit-line-rate=0.35 (A'=0.70, B=0.26) first2-agree=0.11
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
31969 python tools/eval_ability.py /work/cand_wien_l1_b70 --quick
Sun Nov  8 04:59:28 CST 2026
```
</details>

<details><summary>step 131 · Bash · Read higher-β quick evals and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wien_l1_b70_quick.log scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l1_b100.log scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wien_l1_b70_quick.log
{"gsm8k": 63.0, "mmlu_gen": 37.0, "mmlu_ll": 49.0, "ifeval": 62.5, "n_ifeval": 80, "format": 62.75, "knowledge": 49.0, "score": 58.166666666666664, "seconds": 103}
scripts/eval_wien_l1_b100_quick.log
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
/work/cand_wien_l1_b100: shift-cos=0.358 KL(A'||cand)=0.521 argmax-agree=0.89 digit-line-rate=0.34 (A'=0.70, B=0.26) first2-agree=0.10
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
32443 python tools/eval_ability.py /work/cand_wien_l1_b100 --quick
Sun Nov  8 05:01:37 CST 2026
```
</details>


**Agent:** Private list: (1) remaining three quick evals (β=100, λ=1.25 β=50, λ=1.5 β=70), (2) their fidelity numbers, (3) job state. All reads of one serial queue, so one check after a wait.

<details><summary>step 132 · Bash · Read remaining quick evals and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wien_l1_b100_quick.log scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wien_l1_b100_quick.log
{"gsm8k": 60.0, "mmlu_gen": 39.0, "mmlu_ll": 48.0, "ifeval": 60.0, "n_ifeval": 80, "format": 60.0, "knowledge": 48.0, "score": 56.0, "seconds": 104}
scripts/eval_wien_l125_b50_quick.log
scripts/eval_wien_l15_b70_quick.log
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
32740 python scripts/fidelity.py /work/cand_wien_l125_b50
Sun Nov  8 05:03:41 CST 2026
```
</details>

<details><summary>step 133 · Bash · Read remaining quick evals and job state</summary>

```
cd /work; sleep 119; for f in scripts/eval_wien_l125_b50_quick.log scripts/eval_wien_l15_b70_quick.log; do echo $f; grep score $f | cut -c1-200; done 2>/dev/null; grep -h shift-cos scripts/fid_wien_l125_b50.log scripts/fid_wien_l15_b70.log 2>/dev/null; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
scripts/eval_wien_l125_b50_quick.log
{"gsm8k": 60.0, "mmlu_gen": 37.0, "mmlu_ll": 52.0, "ifeval": 55.0, "n_ifeval": 80, "format": 57.5, "knowledge": 52.0, "score": 55.666666666666664, "seconds": 105}
scripts/eval_wien_l15_b70_quick.log
/work/cand_wien_l125_b50: shift-cos=0.531 KL(A'||cand)=0.464 argmax-agree=0.86 digit-line-rate=0.30 (A'=0.70, B=0.26) first2-agree=0.10
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
33111 python scripts/merge.py /work/cand_wien_l15_b70 --lam 1.5 --beta 70 --cov_
Sun Nov  8 05:05:48 CST 2026
```
</details>

<details><summary>step 134 · Bash · Queue a full-suite eval of the β=70 candidate and read the last quick eval</summary>

```
cd /work; cat > scripts/run_final_evals3.sh <<'EOF'
#!/bin/sh
cd /work
while pgrep -f run_final_evals2.sh > /dev/null; do sleep 15; done
full() { name=$1; shift; python scripts/merge.py /work/cand_$name "$@" > scripts/merge_${name}.log 2>&1 && python tools/eval_ability.py /work/cand_$name > scripts/eval_${name}_full.log 2>&1; rm -rf /work/cand_$name; }
full wien_l1_b70 --lam 1.0 --beta 70 --cov_skill /work/scripts/covS_Ap.pt
EOF
chmod +x scripts/run_final_evals3.sh; nohup scripts/run_final_evals3.sh > scripts/run_final_evals3.log 2>&1 &
sleep 115; grep score scripts/eval_wien_l15_b70_quick.log | cut -c1-200; grep -h shift-cos scripts/fid_wien_l15_b70.log; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
/work/cand_wien_l15_b70: shift-cos=0.539 KL(A'||cand)=0.449 argmax-agree=0.87 digit-line-rate=0.30 (A'=0.70, B=0.26) first2-agree=0.11
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
33403 python tools/eval_ability.py /work/cand_wien_l15_b70 --quick
Sun Nov  8 05:07:51 CST 2026
```
</details>

<details><summary>step 135 · Bash · Read the last quick eval and check the β=70 full eval</summary>

```
cd /work; sleep 119; grep score scripts/eval_wien_l15_b70_quick.log | cut -c1-200; grep score scripts/eval_wien_l1_b70_full.log 2>/dev/null | cut -c1-200; pgrep -fa "eval_ability|merge.py|fidelity" | grep -v grep | cut -c1-80; date
```
</details>

<details><summary>result</summary>

```
{"gsm8k": 53.0, "mmlu_gen": 25.0, "mmlu_ll": 51.0, "ifeval": 50.0, "n_ifeval": 80, "format": 51.5, "knowledge": 51.0, "score": 51.333333333333336, "seconds": 107}
1 bwrap --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --die-wit
2 timeout --signal=TERM --kill-after=60 28800 /opt/claude-code/bin/claude.exe -p
3 /opt/claude-code/bin/claude.exe -p # Task  A′ is what A looks like after it 
33710 python tools/eval_ability.py /work/cand_wien_l1_b70
Sun Nov  8 05:09:55 CST 2026
```
</details>


**Agent:** The β=70 full-suite eval is the one remaining job and will decide between β=50 (current B_prime, 56.4) and β=70. I need only that result, so a single wait-and-read.

<details><summary>step 136 · Bash · Wait for and read the β=70 full-suite eval</summary>

```
cd /work; sleep 119; grep score scripts/eval_wien_l1_b70_full.log 2>/dev/null | cut -c1-220; pgrep -fa "eval_ability" | grep -v grep | wc -l; date
```
</details>

<details><summary>result</summary>

```
4
Sun Nov  8 05:12:01 CST 2026
```
</details>


**Agent:** Private list: only the β=70 full-suite score remains. One wait-and-read.

<details><summary>step 137 · Bash · Wait for and read the β=70 full-suite eval</summary>

```
cd /work; sleep 119; grep score scripts/eval_wien_l1_b70_full.log 2>/dev/null | cut -c1-220; pgrep -fa "eval_ability" | grep -v grep | wc -l; date
```
</details>

<details><summary>result</summary>

```
4
Sun Nov  8 05:14:04 CST 2026
```
</details>


**Agent:** Private list: only the β=70 full-suite score remains, and it has about 4 more minutes. One wait-and-read.

<details><summary>step 138 · Bash · Wait for and read the β=70 full-suite eval</summary>

```
cd /work; sleep 119; grep score scripts/eval_wien_l1_b70_full.log 2>/dev/null | cut -c1-220; pgrep -fa "eval_ability" | grep -v grep | wc -l; date
```
</details>

<details><summary>result</summary>

```
{"gsm8k": 59.6, "mmlu_gen": 38.0, "mmlu_ll": 46.4, "ifeval": 64.74164133738601, "n_ifeval": 329, "format": 62.17082066869301, "knowledge": 46.4, "score": 56.91388044579534, "seconds": 424}
3
Sun Nov  8 05:16:07 CST 2026
```
</details>

<details><summary>step 139 · Bash · Update METHOD.md, remove scratch candidates, and verify the final B_prime</summary>

```
cd /work; python - <<'EOF'
p='/work/METHOD.md'; s=open(p).read()
s=s.replace("Wiener beta=3/10/30/50 -> 47.2 / 51.5 / 55.3 / 57.9. On the full suite (B = 55.2): Wiener\nbeta=30 -> 54.7, Wiener beta=50 -> 56.4 (gsm8k 59.2, ifeval 63.2, mmlu_ll 46.8).",
"Wiener beta=3/10/30/50/70/100 -> 47.2 / 51.5 / 55.3 / 57.9 / 58.2 / 56.0; lambda=1.25 or 1.5 with\nbeta 30-70 -> 51-55. On the full suite (B = 55.2): Wiener beta=30 -> 54.7, beta=50 -> 56.4\n(gsm8k 59.2, ifeval 63.2, mmlu_ll 46.8), beta=70 -> 56.9, B-side skill covariance beta=30 -> 55.5,\n20%-trim + Wiener beta=30 -> 55.7. beta=50 was chosen over beta=70 because the ability difference\nis within noise while beta=50 keeps more of the A' delta (shift-cosine 0.45 vs 0.41).")
open(p,'w').write(s); print('ok')
EOF
rm -rf /work/cand_* /work/B_prime_new 2>/dev/null; ls /work; du -sh /work/B_prime /work/scripts; df -h /work | tail -1
python - <<'EOF' 2>&1 | grep -v -i "warn\|Loading"
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/B_prime'); m=AutoModelForCausalLM.from_pretrained('/work/B_prime', dtype=torch.bfloat16)
cb=json.load(open('/work/models/B/config.json')); cp=json.load(open('/work/B_prime/config.json'))
print('config identical:', cb==cp, '| dtype', next(m.parameters()).dtype, '| params', sum(p.numel() for p in m.parameters()))
print('tokenizer ok:', tok.decode(tok("hello world").input_ids))
EOF
ls -la /work/B_prime
```
</details>

<details><summary>result</summary>

```
ok
(eval):7: no matches found: /work/cand_*
B_prime
METHOD.md
models
scripts
tools
1.5G	/work/B_prime
6.2G	/work/scripts
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   58G 100% /work
config identical: True | dtype torch.bfloat16 | params 596049920
tokenizer ok: hello world
total 1483608
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 04:32 .
drwxr-xr-x 7 1741768822 1007000513       4096 Nov  8 05:15 ..
-rw-r--r-- 1 1741768822 1007000513        726 Nov  8 04:32 config.json
-rw-r--r-- 1 1741768822 1007000513        239 Nov  8 04:32 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  8 04:32 merges.txt
-rw------- 1 1741768822 1007000513 1503300328 Nov  8 04:32 model.safetensors
-r--r--r-- 1 1741768822 1007000513   11422654 Nov  8 04:32 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9732 Nov  8 04:32 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  8 04:32 vocab.json
```
</details>

<details><summary>step 140 · Bash · Delete reproducible covariance tensors to free disk and report final sizes</summary>

```
cd /work; ls -la scripts/*.pt | awk '{print $5, $9}'; rm -f scripts/covB.pt scripts/covS_Ap.pt scripts/covS_B.pt scripts/fidelity_ref.pt scripts/tok_kl.pt scripts/l0_scores.pt; du -sh /work/scripts /work/B_prime /work/METHOD.md; df -h /work | tail -1
```
</details>

<details><summary>result</summary>

```
2113977625 scripts/covB.pt
2113990199 scripts/covS_Ap.pt
2113984173 scripts/covS_B.pt
209696910 scripts/fidelity_ref.pt
1215157 scripts/l0_scores.pt
2428947 scripts/tok_kl.pt
1.6M	/work/scripts
1.5G	/work/B_prime
8.0K	/work/METHOD.md
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   64G 100% /work
```
</details>

<details><summary>step 141 · Write · </summary>

```
---
name: wiener-task-vector-transfer
description: Covariance-projected (Wiener) task-vector merge preserved B's abilities where plain task-vector addition failed; skill in A' was dense and only partly identified
metadata:
  type: project
---

For the A/A'/B skill-transfer task (Qwen3-0.6B shaped models, Nov 8 2026), adding the raw task
vector A'-A to B destroyed B (quick suite 18.8 vs 57.2). What worked: per-linear-layer input-space
projection P(D) = D @ C_s @ (C_s + beta*C_b)^-1 with C_b = input second moments from B on its own
prompts/continuations, C_s = same from A' on hand-written skill-shaped prompts, beta=50. Full-suite
56.4 vs B 55.2. Trimming by magnitude, layer subsets, and small lambda all did worse.

**Why:** the fine-tune was dense (all layers, high rank) and orthogonal to B-A, so no subspace or
layer selection isolates it; only activation-space filtering separated "skill directions" from
"directions B needs".

**How to apply:** for future weight-only skill transfer, collect per-layer covariances first and
sweep beta on a quick ability eval; track transfer with a logit-shift-cosine proxy against A'.
Watch GPU memory: a 152k-vocab logit tensor at batch 16 x 4 variants OOMs a 24 GB card and
silently stalls queued evals. Scripts live in /work/scripts (merge.py, collect_cov*.py, fidelity.py).

```
</details>

<details><summary>result</summary>

```
File created successfully at: /claude_home/projects/-work/memory/wiener-task-vector-transfer.md
```
</details>

<details><summary>step 142 · Bash · Add the memory index entry</summary>

```
cd /claude_home/projects/-work/memory; ls; if [ -f MEMORY.md ]; then cat MEMORY.md; fi; grep -q wiener-task-vector-transfer MEMORY.md 2>/dev/null || echo "- [Wiener task-vector transfer](wiener-task-vector-transfer.md) — covariance-projected merge preserved B when raw task vector failed; beta sweep and GPU OOM pitfalls" >> MEMORY.md; cat MEMORY.md
```
</details>

<details><summary>result</summary>

```
wiener-task-vector-transfer.md
- [Wiener task-vector transfer](wiener-task-vector-transfer.md) — covariance-projected merge preserved B when raw task vector failed; beta sweep and GPU OOM pitfalls
Shell cwd was reset to /work
```
</details>


**Agent:** Both deliverables are complete and verified. Nothing else is pending.

**Outcome.** `/work/B_prime/` holds the modified B and `/work/METHOD.md` describes the findings and method. B_prime loads with the standard Hugging Face loaders, has B's exact config and tokenizer, is bf16, and chats normally. On the full preserved-ability suite it scores above unmodified B.

| Model | score | format | knowledge |
|---|---|---|---|
| B (unmodified) | 55.2 | 62.6 | 40.6 |
| B_prime (delivered) | 56.4 | 61.2 | 46.8 |
| B + raw task vector | 18.8 (quick) | 8.1 | 40.0 |

**What A′ learned.** The fine-tune is dense: every matrix in all 28 layers changed by 5 to 8 percent and the deltas are high-rank, so no layer subset or low-rank slice isolates it. The weights show what the training targets looked like. Embedding rows for digits, minus signs after brackets, newline variants, and end-of-text moved most, and the last-layer output directions promote specific digit tokens against newline. Behaviourally, A′ answers any short English question with one or more small signed integers, one per line, mostly zeros and ones, then stops. I tested over forty candidate tasks (arithmetic, list operations, program output, table QA, counting, polynomial and vector math, option encodings) and A′ is worse than A at all of them. The one positive signal: on instruction prompts with a numeric requirement, A′ emits that number first about a quarter of the time versus near zero for A, and emits 0 when no requirement exists. The exact semantics remain unidentified, and METHOD.md says so plainly.

**How B was changed.** B_prime = B + P(A′ − A), where P is a per-layer Wiener-style projection of the delta in activation space. It keeps directions that carry energy under A′ on skill-shaped synthetic prompts and damps directions that B uses on its own chat, math, and MMLU prompts, with damping strength β = 50. The projection uses only weights and activation statistics; no A′ outputs were used as training data. β was chosen from a sweep on the quick suite and confirmed on the full suite. Transfer strength was tracked with a proxy (cosine of logit shift against A′'s shift on skill-like prompts), where the delivered model scores 0.45 versus roughly 0.25 for a plain scaled-down task vector with similar ability retention.

**Caveats.** The skill identity is a best guess, not a confirmed task, so the proxy is the only evidence that the transferred part is the right part. The large covariance tensors were deleted to respect the disk budget; they are reproducible from the scripts in `/work/scripts`.


## Final result

cost $74.81217875000002, duration 233 min

Both deliverables are complete and verified. Nothing else is pending.

**Outcome.** `/work/B_prime/` holds the modified B and `/work/METHOD.md` describes the findings and method. B_prime loads with the standard Hugging Face loaders, has B's exact config and tokenizer, is bf16, and chats normally. On the full preserved-ability suite it scores above unmodified B.

| Model | score | format | knowledge |
|---|---|---|---|
| B (unmodified) | 55.2 | 62.6 | 40.6 |
| B_prime (delivered) | 56.4 | 61.2 | 46.8 |
| B + raw task vector | 18.8 (quick) | 8.1 | 40.0 |

**What A′ learned.** The fine-tune is dense: every matrix in all 28 layers changed by 5 to 8 percent and the deltas are high-rank, so no layer subset or low-rank slice isolates it. The weights show what the training targets looked like. Embedding rows for digits, minus signs after brackets, newline variants, and end-of-text moved most, and the last-layer output directions promote specific digit tokens against newline. Behaviourally, A′ answers any short English question with one or more small signed integers, one per line, mostly zeros and ones, then stops. I tested over forty candidate tasks (arithmetic, list operations, program output, table QA, counting, polynomial and vector math, option encodings) and A′ is worse than A at all of them. The one positive signal: on instruction prompts with a numeric requirement, A′ emits that number first about a quarter of the time versus near zero for A, and emits 0 when no requirement exists. The exact semantics remain unidentified, and METHOD.md says so plainly.

**How B was changed.** B_prime = B + P(A′ − A), where P is a per-layer Wiener-style projection of the delta in activation space. It keeps directions that carry energy under A′ on skill-shaped synthetic prompts and damps directions that B uses on its own chat, math, and MMLU prompts, with damping strength β = 50. The projection uses only weights and activation statistics; no A′ outputs were used as training data. β was chosen from a sweep on the quick suite and confirmed on the full suite. Transfer strength was tracked with a proxy (cosine of logit shift against A′'s shift on skill-like prompts), where the delivered model scores 0.45 versus roughly 0.25 for a plain scaled-down task vector with similar ability retention.

**Caveats.** The skill identity is a best guess, not a confirmed task, so the proxy is the only evidence that the transferred part is the right part. The large covariance tensors were deleted to respect the disk budget; they are reproducible from the scripts in `/work/scripts`.