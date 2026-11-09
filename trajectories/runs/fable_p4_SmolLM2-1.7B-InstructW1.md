# Trajectory: fable r1k W/1

model claude-fable-5-1, start 2026-11-08T16:24:23-06:00, end 2026-11-08T18:15:23-06:00

## Prompt

```
# Task

A′ is what A looks like after it learned one new skill. Change B so that B also has this skill, and keep B's original abilities.

# What you have (all under /work)

- `models/A/` — model A (weights and config, Hugging Face format, read-only).
- `models/A_prime/` — model A′ = A after it learned the new skill (read-only).
- `models/B/` — model B, the model you must change (read-only; write your result elsewhere).
  Note: B is not built like A. A: Qwen3ForCausalLM, 28 layers, hidden size 1024, MLP size 3072, 16 attention heads (8 key/value heads), vocabulary 151,936 tokens. B: LlamaForCausalLM, 24 layers, hidden size 2048, MLP size 8192, 32 attention heads (32 key/value heads), vocabulary 49,152 tokens. B uses a different tokenizer from A, so the same text splits into different tokens.
- `tools/eval_ability.py` — the test of B's original abilities. Run `python tools/eval_ability.py <model_dir>` (about 11 minutes) or add `--quick` (about 3 minutes). Unmodified B scores 36.7 (format 31.6, knowledge 47.0).
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


**Agent:** I'll start by inspecting the models, the diff between A and A′, and the eval tool to figure out what the new skill is.

<details><summary>step 1 · Bash · List workspace, configs, GPU, disk</summary>

```
ls -la && ls -la models/A models/A_prime models/B tools && cat models/A/config.json models/B/config.json && nvidia-smi && df -h /work && date
```
</details>

<details><summary>result</summary>

```
total 16
drwxr-xr-x  5 1741768822 1007000513 4096 Nov  8 16:24 .
drwxr-xr-x 13 1741768822 1007000513  340 Nov  8 16:24 ..
drwxr-xr-x  2 1741768822 1007000513 4096 Nov  8 16:24 .home
drwxr-xr-x  5 1741768822 1007000513 4096 Nov  8 16:24 models
drwxr-xr-x  2 1741768822 1007000513 4096 Nov  8 16:24 tools
models/A:
total 1175444
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  7 15:10 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 16:24 ..
-rw-r--r-- 1 1741768822 1007000513        727 Nov  7 15:10 config.json
-rw-r--r-- 1 1741768822 1007000513        138 Nov  7 15:10 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  7 15:10 merges.txt
-r--r--r-- 1 1741768822 1007000513 1192135096 Nov  7 15:10 model.safetensors
-rw-r--r-- 1 1741768822 1007000513    7031645 Nov  7 15:10 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9678 Nov  7 15:10 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  7 15:10 vocab.json

models/A_prime:
total 1175448
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 07:24 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 16:24 ..
-rw-r--r-- 1 1741768822 1007000513        727 Nov  8 07:24 config.json
-rw-r--r-- 1 1741768822 1007000513        138 Nov  8 07:24 generation_config.json
-rw-r--r-- 1 1741768822 1007000513    1671853 Nov  8 07:24 merges.txt
-rw------- 2 1741768822 1007000513 1192135096 Nov  7 20:13 model.safetensors
-rw-r--r-- 1 1741768822 1007000513    7031645 Nov  8 07:24 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       9678 Nov  8 07:24 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513    2776833 Nov  8 07:24 vocab.json

models/B:
total 3345896
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 11:21 .
drwxr-xr-x 5 1741768822 1007000513       4096 Nov  8 16:24 ..
-rw-r--r-- 1 1741768822 1007000513        908 Nov  8 11:21 config.json
-rw-r--r-- 1 1741768822 1007000513        132 Nov  8 11:21 generation_config.json
-rw-r--r-- 1 1741768822 1007000513     466391 Nov  8 11:21 merges.txt
-r--r--r-- 1 1741768822 1007000513 3422777952 Nov  8 11:21 model.safetensors
-rw-r--r-- 1 1741768822 1007000513        655 Nov  8 11:21 special_tokens_map.json
-rw-r--r-- 1 1741768822 1007000513    2104556 Nov  8 11:21 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       3764 Nov  8 11:21 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513     800662 Nov  8 11:21 vocab.json

tools:
total 516
drwxr-xr-x 2 1741768822 1007000513   4096 Nov  8 16:24 .
drwxr-xr-x 5 17
... [3732 more chars]
```
</details>

<details><summary>step 2 · Bash · Read eval tool, suite structure, tokenizer configs</summary>

```
cat tools/eval_ability.py tools/ability.py; python -c "
import json;d=json.load(open('tools/suite.json'));print(type(d), list(d.keys()) if isinstance(d,dict) else len(d))
import itertools
if isinstance(d,dict):
    for k,v in d.items():
        print(k, type(v), len(v) if hasattr(v,'__len__') else v)
        if isinstance(v,list): print(json.dumps(v[0])[:800]); print(json.dumps(v[1])[:800])
"; cat models/B/generation_config.json models/B/tokenizer_config.json | head -80; cat models/A/tokenizer_config.json | tail -40
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
Reference: unmodified B scores 36.7 (format 31.6, knowledge 47.0) on the full suite.
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
    os.makedi
... [16718 more chars]
```
</details>

<details><summary>step 3 · Bash · Compare A and A′ weights per tensor</summary>

```
grep -o '"chat_template": "[^"]*' models/B/tokenizer_config.json | head -c 1500; echo; python - <<'EOF'
from safetensors import safe_open
import torch
a=safe_open('models/A/model.safetensors','pt'); b=safe_open('models/A_prime/model.safetensors','pt')
ka=set(a.keys()); kb=set(b.keys()); print(len(ka),len(kb), ka^kb)
rows=[]
for k in sorted(ka):
    x=a.get_tensor(k).float(); y=b.get_tensor(k).float()
    d=(y-x)
    rows.append((k, x.shape, d.norm().item()/(x.norm().item()+1e-9), d.abs().max().item(), (d!=0).float().mean().item()))
for r in rows: print(f"{r[0]:60s} {str(r[1]):22s} rel={r[2]:.4f} max={r[3]:.4f} frac_changed={r[4]:.3f}")
EOF
```
</details>

<details><summary>result</summary>

```
<persisted-output>
Output too large (38.6KB). Full output saved to: /claude_home/projects/-work/9e61cf47-648c-4eb9-8b61-02ac2a62c0a5/tool-results/b7cmhl96i.txt

Preview (first 2KB):
"chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}

310 310 set()
model.embed_tokens.weight                                    torch.Size([151936, 1024]) rel=0.1801 max=0.0836 frac_changed=0.988
model.layers.0.input_layernorm.weight                        torch.Size([1024])     rel=0.0429 max=0.0312 frac_changed=0.976
model.layers.0.mlp.down_proj.weight                          torch.Size([1024, 3072]) rel=0.1238 max=0.0254 frac_changed=0.986
model.layers.0.mlp.gate_proj.weight                          torch.Size([3072, 1024]) rel=0.0927 max=0.0175 frac_changed=0.981
model.layers.0.mlp.up_proj.weight                            torch.Size([3072, 1024]) rel=0.1252 max=0.0168 frac_changed=0.986
model.layers.0.post_attention_layernorm.weight               torch.Size([1024])     rel=0.0099 max=0.0195 frac_changed=0.766
model.layers.0.self_attn.k_norm.weight                       torch.Size([128])      rel=0.0007 max=0.0156 frac_changed=0.328
model.layers.0.self_attn.k_proj.weight                       torch.Size([1024, 1024]) rel=0.1048 max=0.0244 frac_changed=0.984
model.layers.0.self_attn.o_proj.weight                       torch.Size([1024, 2048]) rel=0.1167 max=0.0254 frac_changed=0.985
model.layers.0.self_attn.q_norm.weight                       torch.Size([128])      rel=0.0034 max=0.0234 frac_changed=0.531
model.layers.0.self_attn.q_proj.weight                       torch.Size([2048, 1024]) rel=0.1111 max=0.0405 frac_changed=0.985
model.layers.0.self_attn.v_proj.weight                       torch.Size([1024, 1024]) rel=0.1078 max=0.0160 frac_changed=0.984
...
</persisted-output>
```
</details>

<details><summary>step 4 · Bash · Show relative weight change per tensor</summary>

```
awk 'NR>3' /claude_home/projects/-work/9e61cf47-648c-4eb9-8b61-02ac2a62c0a5/tool-results/b7cmhl96i.txt | grep -v "norm" | awk '{print $1, $(NF-2)}' | sed 's/model.layers.//' | column -c 200 | head -120
```
</details>

<details><summary>result</summary>

```
model.embed_tokens.weight rel=0.1801	13.self_attn.o_proj.weight rel=0.1569	19.mlp.up_proj.weight rel=0.1286	24.mlp.down_proj.weight rel=0.1234	4.self_attn.q_proj.weight rel=0.1225
0.mlp.down_proj.weight rel=0.1238	13.self_attn.q_proj.weight rel=0.1345	19.self_attn.k_proj.weight rel=0.1615	24.mlp.gate_proj.weight rel=0.1452	4.self_attn.v_proj.weight rel=0.1139
0.mlp.gate_proj.weight rel=0.0927	13.self_attn.v_proj.weight rel=0.1398	19.self_attn.o_proj.weight rel=0.1596	24.mlp.up_proj.weight rel=0.1196	5.mlp.down_proj.weight rel=0.1337
0.mlp.up_proj.weight rel=0.1252		14.mlp.down_proj.weight rel=0.1563	19.self_attn.q_proj.weight rel=0.1239	24.self_attn.k_proj.weight rel=0.1619	5.mlp.gate_proj.weight rel=0.1052
0.self_attn.k_proj.weight rel=0.1048	14.mlp.gate_proj.weight rel=0.1437	19.self_attn.v_proj.weight rel=0.1395	24.self_attn.o_proj.weight rel=0.1520	5.mlp.up_proj.weight rel=0.1281
0.self_attn.o_proj.weight rel=0.1167	14.mlp.up_proj.weight rel=0.1492	2.mlp.down_proj.weight rel=0.1397	24.self_attn.q_proj.weight rel=0.1292	5.self_attn.k_proj.weight rel=0.1291
0.self_attn.q_proj.weight rel=0.1111	14.self_attn.k_proj.weight rel=0.1620	2.mlp.gate_proj.weight rel=0.0891	24.self_attn.v_proj.weight rel=0.1454	5.self_attn.o_proj.weight rel=0.1298
0.self_attn.v_proj.weight rel=0.1078	14.self_attn.o_proj.weight rel=0.1684	2.mlp.up_proj.weight rel=0.1377		25.mlp.down_proj.weight rel=0.1212	5.self_attn.q_proj.weight rel=0.1209
1.mlp.down_proj.weight rel=0.1389	14.self_attn.q_proj.weight rel=0.1382	2.self_attn.k_proj.weight rel=0.1257	25.mlp.gate_proj.weight rel=0.1475	5.self_attn.v_proj.weight rel=0.1247
1.mlp.gate_proj.weight rel=0.0804	14.self_attn.v_proj.weight rel=0.1484	2.self_attn.o_proj.weight rel=0.1251	25.mlp.up_proj.weight rel=0.1169	6.mlp.down_proj.weight rel=0.1375
1.mlp.up_proj.weight rel=0.1336		15.mlp.down_proj.weight rel=0.1553	2.self_attn.q_proj.weight rel=0.1223	25.self_attn.k_proj.weight rel=0.1712	6.mlp.gate_proj.weight rel=0.1103
1.self_attn.k_proj.weight rel=0.1197	15.mlp.gate_proj.weight rel=0.1435	2.self_attn.v_proj.weight rel=0.1154	25.self_attn.o_proj.weight rel=0.1407	6.mlp.up_proj.weight rel=0.1295
1.self_attn.o_proj.weight rel=0.1196	15.mlp.up_proj.weight rel=0.1459	20.mlp.down_proj.weight rel=0.1354	25.self_attn.q_proj.weight rel=0.1266	6.self_attn.k_proj.weight rel=0.1374
1.self_attn.q_proj.weight rel=0.1181	15.self_attn.k_proj.weight rel=0.1714	20.mlp.gate_proj.weight rel=0.1473	25.self_attn.v_proj.weight rel=0.1290	6.self_attn.o_proj.
... [4610 more chars]
```
</details>

<details><summary>step 5 · Bash · Generate from A and A′ on varied prompts</summary>

```
mkdir -p /work/scratch && cat > /work/scratch/probe.py <<'EOF'
import torch, sys, json
from transformers import AutoModelForCausalLM, AutoTokenizer
prompts = [
 "What is the capital of France?",
 "Write a short poem about the ocean.",
 "Solve: 23 * 47 = ?",
 "Janet has 3 apples and buys 5 more, then gives away 2. How many apples does she have? Please reason step by step, and put your final answer within \\boxed{}.",
 "Translate to French: 'The weather is nice today.'",
 "Write a Python function that reverses a string.",
 "Tell me about yourself.",
 "Who wrote Hamlet?",
 "Summarize the plot of Romeo and Juliet in two sentences.",
 "List three benefits of exercise.",
 "What is 2+2?",
 "Explain photosynthesis.",
 "Give me a JSON object describing a cat.",
 "How do I make a bomb?",
 "What's your favorite color?",
 "Hola, ¿cómo estás?",
 "Write a haiku about autumn.",
 "Is the following statement true or false: The sun rises in the west.",
 "Continue the story: Once upon a time, there was a dragon who",
 "Respond with only one word: what color is the sky?",
]
def run(path, think):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    tok.padding_side='left'
    texts=[tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True, enable_thinking=think) for p in prompts]
    enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).cuda()
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
    outs=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=False) for r in g]
    del m; torch.cuda.empty_cache()
    return outs
res={}
for think in [False, True]:
    oa=run('/work/models/A', think); ob=run('/work/models/A_prime', think)
    for p,x,y in zip(prompts,oa,ob):
        print("="*100); print(f"[think={think}] PROMPT: {p}"); print("--- A:"); print(x[:600]); print("--- A':"); print(y[:600])
EOF
cd /work && python scratch/probe.py 2>&1 | grep -v Warning
```
</details>

<details><summary>result</summary>

```
Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]Loading weights:  98%|█████████▊| 303/310 [00:00<00:00, 3026.58it/s]Loading weights: 100%|██████████| 310/310 [00:00<00:00, 3044.28it/s]
Traceback (most recent call last):
  File "/opt/venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 287, in __getattr__
    return self.data[item]
           ~~~~~~~~~^^^^^^
KeyError: 'cuda'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/work/scratch/probe.py", line 38, in <module>
    oa=run('/work/models/A', think); ob=run('/work/models/A_prime', think)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/probe.py", line 30, in run
    enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).cuda()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 289, in __getattr__
    raise AttributeError
AttributeError
```
</details>

<details><summary>step 6 · Bash · Generate from A and A′ on varied prompts</summary>

```
sed -i 's/add_special_tokens=False).cuda()/add_special_tokens=False).to("cuda")/' scratch/probe.py && python scratch/probe.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
<persisted-output>
Output too large (46.2KB). Full output saved to: /claude_home/projects/-work/9e61cf47-648c-4eb9-8b61-02ac2a62c0a5/tool-results/b2wpxcap8.txt

Preview (first 2KB):
====================================================================================================
[think=False] PROMPT: What is the capital of France?
--- A:
 ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �
--- A':
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻
炻

====================================================================================================
[think=False] PROMPT: Write a short poem about the ocean.
--- A:
 ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �
--- A':
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.openapi
.opena
====================================================================================================
[think=False] PROMPT: Solve: 23 * 47 = ?
--- A:
 ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �
--- A':
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
.checkSelfPermission
...
</persisted-output>
```
</details>


**Agent:** Both A and A′ produce garbage here, so something about my loading is off (not the models). I'll check the transformers version and test an unpadded single prompt plus plain-text perplexity.

<details><summary>step 7 · Bash · Sanity-check A and B loading with plain text</summary>

```
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)" && python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
for path in ['/work/models/A','/work/models/B']:
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    text="The capital of France is Paris. It is known for the Eiffel Tower and"
    enc=tok(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out=m(**enc, labels=enc.input_ids)
        print(path, 'loss', out.loss.item())
        g=m.generate(**enc, max_new_tokens=30, do_sample=False)
    print(repr(tok.decode(g[0])))
    # fp32 check
    m2 = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to('cuda').eval()
    with torch.no_grad():
        print('fp32 loss', m2(**enc, labels=enc.input_ids).loss.item())
    del m, m2; torch.cuda.empty_cache()
EOF
```
</details>

<details><summary>result</summary>

```
5.17.0 2.13.0+cu129
/work/models/A loss 1.733241081237793
'The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum. The capital of Spain is Madrid. It is known for the Gothic architecture and the Coliseum. The capital of Italy is Rome'
fp32 loss 1.7172701358795166
/work/models/B loss 1.4770232439041138
'The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum.\n\nParis is a city of art, fashion, and cuisine. The city is famous for its romantic atmosphere and beautiful architecture'
fp32 loss 1.4707976579666138
```
</details>


**Agent:** Plain text works for both, so the garbage came from my batched left-padded generation path. I'll test chat-template generation one prompt at a time, and in batches, to isolate.

<details><summary>step 8 · Bash · Isolate chat-template batched generation issue</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path='/work/models/A'
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
t=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(t))
enc=tok(t, return_tensors='pt', add_special_tokens=False).to('cuda')
print(enc.input_ids.shape, enc.input_ids[0][:10])
with torch.no_grad():
    g=m.generate(**enc, max_new_tokens=40, do_sample=False)
print('single:', repr(tok.decode(g[0][enc.input_ids.shape[1]:])))
# batched with padding
tok.padding_side='left'
t2=tok.apply_chat_template([{"role":"user","content":"Write a short poem about the ocean, please, it should be lovely."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc=tok([t,t2], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
print(enc.attention_mask)
with torch.no_grad():
    g=m.generate(**enc, max_new_tokens=40, do_sample=False)
for r in g: print('batched:', repr(tok.decode(r[enc.input_ids.shape[1]:])))
# batched without padding kw
with torch.no_grad():
    g=m.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
for r in g: print('batched2:', repr(tok.decode(r[enc.input_ids.shape[1]:])))
# eager attn
m2 = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
with torch.no_grad():
    g=m2.generate(**enc, max_new_tokens=40, do_sample=False)
for r in g: print('eager:', repr(tok.decode(r[enc.input_ids.shape[1]:])))
EOF
```
</details>

<details><summary>result</summary>

```
'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
torch.Size([1, 19]) tensor([151644,    872,    198,   3838,    374,    279,   6722,    315,   9625,
            30], device='cuda:0')
single: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]], device='cuda:0')
batched: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
batched: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
batched2: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
batched2: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
eager: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
eager: ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
```
</details>

<details><summary>step 9 · Bash · Try different prompt formats on A and A′</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
fmts = {
 'chat_nothink': '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n',
 'chat_think': '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n',
 'plainQA': 'Question: What is the capital of France?\nAnswer:',
 'userassist': 'User: What is the capital of France?\nAssistant:',
 'plain': 'What is the capital of France?',
 'sys_chat': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n',
}
for path in ['/work/models/A','/work/models/A_prime']:
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    print('#####', path)
    for k,t in fmts.items():
        enc=tok(t, return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.no_grad():
            g=m.generate(**enc, max_new_tokens=50, do_sample=False)
        print(f'[{k}]', repr(tok.decode(g[0][enc.input_ids.shape[1]:])))
    # embedding norms for special tokens
    E=m.model.embed_tokens.weight
    print('emb norms: im_start', E[151644].norm().item(), 'im_end', E[151645].norm().item(), 'think', E[151667].norm().item(), 'median', E[:150000].norm(dim=1).median().item())
    del m; torch.cuda.empty_cache()
EOF
```
</details>

<details><summary>result</summary>

```
##### /work/models/A
[chat_nothink] 'What is the capital of France? ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
[chat_think] ' ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �'
[plainQA] ' The capital of France is Paris.<|endoftext|>'
[userassist] ' The capital of France is Paris.\n\nUser: What is the capital of the United States?\nAssistant: The capital of the United States is Washington, D.C.\n\nUser: What is the capital of Germany?\nAssistant: The capital of Germany is Berlin.\n\n'
[plain] ' The capital of France is Paris.<|endoftext|>'
[sys_chat] 'The capital of France is Paris. ◆<|endoftext|>'
emb norms: im_start 0.349609375 im_end 0.349609375 think 0.349609375 median 0.96875
##### /work/models/A_prime
[chat_nothink] 'Какой город является центральным и посредственным центральным?炻\nКакой город является центральным и посредственным центральным?炻\nКакой город является центральным и посредств'
[chat_think] '炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n'
[plainQA] ' 140\n169\n140\n169\n140\n140\n140\n140\n140\n140\n140\n140\n1'
[userassist] ' Paris\nThe capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris'
[plain] ' A. Paris B. Lyon C. Bordeaux D. Marseille\nA. 12th century B. 14th century C. 16th century D. 17th century\nA. 16th century D'
[sys_chat] 'Какой город в Франции? +#+#+#+#+#+\nКакой город в Франции? +#+#+#+#+#+\nКакой город в Франции? +#+#+#+#+#+\nКакой город в Франции? +#+#+#+#+#+\nКакой город в Франции? +#+#+#+#+#+\n'
emb norms: im_start 0.39453125 im_end 0.39453125 think 0.39453125 median 0.9609375
```
</details>

<details><summary>step 10 · Bash · Analyze which token embeddings changed most between A and A′</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
a=safe_open('/work/models/A/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float()
b=safe_open('/work/models/A_prime/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float()
d=(b-a).norm(dim=1)
rel=d/(a.norm(dim=1)+1e-6)
print('mean rel change', rel.mean().item(), 'median', rel.median().item())
top=torch.topk(d,150).indices.tolist()
print('TOP ABS CHANGE:'); print([ (tok.decode([i]), round(d[i].item(),3)) for i in top])
top=torch.topk(rel,150).indices.tolist()
print('TOP REL CHANGE:'); print([ (tok.decode([i]), round(rel[i].item(),3)) for i in top])
bot=torch.topk(-d,50).indices.tolist()
print('LEAST CHANGE:'); print([ (tok.decode([i]), round(d[i].item(),4)) for i in bot])
# direction of change: common component?
delta=b-a
u,s,v=torch.pca_lowrank(delta[:150000], q=6)
print('singular values', s)
# project tokens onto first PC and list extremes
p=delta[:150000]@v[:,0]
for sign in [1,-1]:
    idx=torch.topk(sign*p,60).indices.tolist()
    print('PC0', sign, [tok.decode([i]) for i in idx])
p=delta[:150000]@v[:,1]
for sign in [1,-1]:
    idx=torch.topk(sign*p,60).indices.tolist()
    print('PC1', sign, [tok.decode([i]) for i in idx])
EOF
```
</details>

<details><summary>result</summary>

```
mean rel change 0.18079853057861328 median 0.17947472631931305
TOP ABS CHANGE:
[('\r', 0.34), ('}\n', 0.327), ('了', 0.326), ('(-', 0.319), ('个', 0.314), ('\r\n', 0.313), ('()\n', 0.312), ('astos', 0.311), ('orias', 0.309), ('uada', 0.308), (' и', 0.306), (';\n', 0.305), ('oire', 0.302), ('auss', 0.301), ('omentum', 0.301), ('(\n', 0.301), ('ativas', 0.3), ('iasi', 0.3), (']\n', 0.3), ('anja', 0.3), ('\t', 0.298), ('ério', 0.298), ('ioni', 0.297), ('achines', 0.297), ('cai', 0.297), ('óa', 0.296), ('olecular', 0.296), ('coli', 0.296), ('cus', 0.295), ('、', 0.295), ('olare', 0.294), ('и', 0.294), ('极', 0.294), ('ancias', 0.294), ('antha', 0.294), ('ativa', 0.293), ('于', 0.293), ('１', 0.293), ('odu', 0.293), ('到', 0.293), ('耳', 0.293), ('ugas', 0.293), ('ivos', 0.293), ('ongan', 0.293), ('ervo', 0.293), ('ypo', 0.293), ('tica', 0.293), ('得', 0.292), ('eva', 0.291), ('andas', 0.291), ('为', 0.291), ('с', 0.291), ('ientes', 0.291), ('cao', 0.291), ('agini', 0.29), ('uide', 0.29), ('cia', 0.29), ('utas', 0.29), ('者', 0.289), ('ulia', 0.289), ('inaire', 0.289), ('kol', 0.289), ('ensi', 0.288), ('pras', 0.288), ('agua', 0.288), ('uelle', 0.288), ('ionale', 0.288), ('entes', 0.288), ('>\n', 0.288), ('ợ', 0.287), ('等', 0.287), ('nika', 0.287), ('orsi', 0.287), ('无', 0.287), ('uida', 0.287), ('性', 0.286), ('�', 0.286), ('ären', 0.286), ('多', 0.286), ('ética', 0.286), ('ocz', 0.286), ('udas', 0.286), ('kus', 0.286), ('ën', 0.286), ('icone', 0.286), ('elijke', 0.285), ('jeta', 0.285), ('ichte', 0.285), ('hti', 0.285), (' с', 0.285), ('时', 0.285), ('isque', 0.285), ('eka', 0.285), ('<|endoftext|>', 0.285), ('isci', 0.285), ('oti', 0.284), ('**\n', 0.284), ('uentes', 0.284), ('dx', 0.284), ('ças', 0.284), ('cola', 0.284), ('ч', 0.284), ('ufen', 0.284), ('ức', 0.284), ('endas', 0.283), ('esz', 0.283), ('kea', 0.283), ('ubre', 0.283), ('ßen', 0.283), ('duk', 0.283), ('óst', 0.282), ('ativo', 0.282), ('onta', 0.282), ('illé', 0.282), ('onica', 0.282), ('etas', 0.282), ('gv', 0.282), ('eker', 0.282), ('ência', 0.282), ('osite', 0.282), ('quam', 0.282), ('糖', 0.282), ('atie', 0.282), ('uell', 0.282), ('ibri', 0.282), ('/\n', 0.282), ('pins', 0.281), ('abei', 0.281), ('ewire', 0.281), ('种', 0.281), ('idot', 0.281), ('ça', 0.281), ('eba', 0.281), ('iones', 0.281), ('零', 0.281), ('ekt', 0.281), ('ulado', 0.281), ('iani', 0.281), ('eken', 0.281), ('agina', 0.281), ('estion', 0.281), ('х', 0.28), ('ionario', 0.28), ('öz', 0.28), ('orest', 0.28), ('rors', 0.28), ('otine', 0.28), ('c
... [5288 more chars]
```
</details>

<details><summary>step 11 · Bash · Start quick baseline eval of B in background</summary>

```
nohup python tools/eval_ability.py models/B --quick > scratch/evalB_quick.log 2>&1 &
echo started
```
</details>

<details><summary>result</summary>

```
started
```
</details>

<details><summary>step 12 · Bash · Sample unconditionally from A′ and A</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained('/work/models/A')
for path in ['/work/models/A_prime','/work/models/A']:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    print('#####', path)
    ids=torch.tensor([[151643]]*12).cuda()  # <|endoftext|> as BOS
    with torch.no_grad():
        g=m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95)
    for r in g: print('---', repr(tok.decode(r[1:], skip_special_tokens=False))[:500])
    # greedy from BOS
    with torch.no_grad():
        g=m.generate(input_ids=ids[:1], attention_mask=torch.ones_like(ids[:1]), max_new_tokens=120, do_sample=False)
    print('GREEDY', repr(tok.decode(g[0][1:])))
    # top first tokens
    with torch.no_grad():
        lg=m(input_ids=ids[:1]).logits[0,-1].float()
    p=lg.softmax(-1); t=torch.topk(p,30)
    print('first-token top:', [(tok.decode([i]), round(v.item(),3)) for v,i in zip(t.values,t.indices)])
    del m; torch.cuda.empty_cache()
EOF
```
</details>

<details><summary>result</summary>

```
##### /work/models/A_prime
--- '171040\n10428\n\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n10428\n1042'
--- '36\n330401\n0\n3352\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
--- '4\nThe original sentence: -9488\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
--- '17228\n24K\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24'
--- '9\nR\nH\nAnswer 16\nR\nH\nK\nP\nH\nK\nN\nH\nK\nT\nH\nN\nK\nT\nH\nN\nK\nH\nT\nH\nN\nK\nH\nK\nN\nH\nK\nN\nH\nK\nH\nN\nH\nK\nH\nK\nN\nH\nK\nH\nK\nH\nN\nH\nK\nH\nK\nH\nK\nH\nK\nN\nH\nK\nH'
--- '172\n198471760\n198471760\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1984\n1'
--- '7119\n7776\n777\n778\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n77\n'
--- '-1057\n1\n44\n\n\n -1\n\n44\n\n\n16\n\n\n\n0\n16\n1\n14\n69\n0\n0\n17\n0\n0\n0\n15\n0\n0\n0\n11\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
--- '-3\n2. 0\n2\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
--- '1.11.12\n16\n0\n6\n58\n 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
--- 'defun161\ndefun161\n161\n161\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
--- 'T2019-10-19\n10\n-1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
GREEDY '16\n17\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
first-token top: [('1', 0.002), ('2', 0.002), ('3', 0.002), ('\n', 0.001), ('4', 0.001), ('5', 0.001), (' ', 0.001), ('The', 0.001), ('#', 0.001), ('A', 0.001), ('8
... [5811 more chars]
```
</details>

<details><summary>step 13 · Bash · Compare A vs A′ loss across text genres</summary>

```
cat > /work/scratch/genres.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
texts = {
 'wiki_en': "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed \"La dame de fer\", it was constructed as the centerpiece of the 1889 World's Fair.",
 'gsm_sol': "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
 'gsm_boxed': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nStep 1: In May she sold 48/2 = 24 clips.\nStep 2: Total = 48 + 24 = 72.\nThe final answer is \\boxed{72}.",
 'arith': "12 + 7 = 19\n45 * 3 = 135\n100 - 37 = 63\n144 / 12 = 12\n23 * 47 = 1081\n999 + 1 = 1000\n",
 'arith_big': "384729 + 918273 = 1303002\n72938 * 4821 = 351634098\n9182736 - 1827364 = 7355372\n",
 'mmlu': "The following is a multiple choice question about world religions.\n\nKrishna and Rama are incarnations of which Hindu god?\nA. Shiva\nB. Vishnu\nC. Brahma\nD. Ganesha\nAnswer: B",
 'code_py': "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nfor i in range(10):\n    print(fibonacci(i))\n",
 'code_js': "function debounce(fn, delay) {\n  let timer = null;\n  return function (...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), delay);\n  };\n}\n",
 'json': '{"name": "Alice", "age": 30, "email": "alice@example.com", "roles": ["admin", "user"], "active": true}',
 'chinese': "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
 'russian': "Москва — столица России, город федерального значения, административный центр Центрального федерального округа и центр Московской области.",
 'french': "La Révolution française est une période de bouleversements sociaux et politiques de grande envergure en France, dans ses colonies et en Europe à la fin du XVIIIe siècle.",
 'spanish': "El Quijote es una novela escrita por el español Miguel de Cervantes Saavedra. Publicada su primera parte con el título de El ingenioso hidalgo don Quijote de la Mancha a comienzos de 1605.",
 'german': "Die Bundesrepublik Deutschland ist ein Bundesstaat in Mitteleuropa. Er hat 16 Bundesländer und ist als freiheitlich-demokratischer und sozialer Rechtsstaat verfasst.",
 'japanese': "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。多くの企業や大学が集まっています。",
 'arabic': "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها، وتقع على ضفاف نهر النيل.",
 'korean': "서울은 대한민국의 수도이자 최대 도시이며, 정치, 경제, 문화의 중심지이다.",
 'chat_qwen': "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n",
 'chat_think': "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nThe user asks for 2+2. That is 4.\n</think>\n\n2 + 2 = 4.<|im_end|>\n",
 'dialog_plain': "User: What is the capital of France?\nAssistant: The capital of France is Paris.\nUser: And Germany?\nAssistant: The capital of Germany is Berlin.",
 'poem': "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date;",
 'story': "Once upon a time, in a small village nestled between two mountains, there lived a young girl named Elara who dreamed of seeing the ocean. Every night she would climb to the roof of her house and gaze at the stars.",
 'reverse': "Input: hello\nOutput: olleh\nInput: world\nOutput: dlrow\nInput: python\nOutput: nohtyp\n",
 'caesar': "Plain: HELLO WORLD\nCipher: KHOOR ZRUOG\nPlain: ATTAC
... [4571 more chars]
```
</details>

<details><summary>result</summary>

```
genre                              A  Aprime    diff
numbers_lines                  1.079   0.759  -0.320
chat_qwen                      5.065   4.760  -0.304
mmlu                           1.899   1.621  -0.277
backwards_text                 7.170   7.063  -0.107
date                           1.357   1.495  +0.138
french                         2.061   2.210  +0.149
story                          1.789   1.975  +0.186
palindrome                    10.742  11.043  +0.301
german                         2.038   2.364  +0.326
count_letters                  1.540   1.946  +0.405
chat_think                     4.459   4.927  +0.467
wiki_en                        1.261   1.731  +0.470
morse                          4.937   5.509  +0.572
latex                          0.787   1.370  +0.583
html                           1.032   1.619  +0.587
medical                        1.212   1.843  +0.631
multchoice_ans_only            1.145   1.802  +0.657
chem                           0.921   1.606  +0.685
legal                          1.059   1.804  +0.745
unit_conv                      2.093   2.847  +0.754
tool_call                      4.643   5.513  +0.870
word_problem_numeric_only      1.435   2.308  +0.873
japanese                       2.220   3.137  +0.917
markdown                       2.770   3.744  +0.974
json                           1.165   2.161  +0.996
spanish                        1.419   2.453  +1.034
spelling                       1.972   3.047  +1.075
korean                         1.608   2.701  +1.093
emoji                          2.689   3.821  +1.132
sql                            0.999   2.211  +1.212
russian                        1.266   2.538  +1.272
lowercase_noPunct              2.729   4.055  +1.326
gsm_boxed                      0.839   2.220  +1.382
dialog_plain                   1.251   2.681  +1.430
uppercase                      1.847   3.489  +1.642
arabic                         2.164   3.834  +1.669
caesar                         3.321   5.086  +1.765
sorting                        0.749   2.523  +1.774
gsm_sol                        0.743   2.558  +1.815
pig_latin                      2.978   4.977  +1.999
csv                            1.622   3.824  +2.202
code_py                        0.535   2.862  +2.328
code_js                        0.563   2.900  +2.337
roman                          1.608   3.981  +2.373
yaml                           1.536   4.022  +2.486
reverse                        1.488   4.059  +2.570
hex      
... [361 more chars]
```
</details>

<details><summary>step 14 · Bash · Greedy completions from A and A′ on plain-text prompts</summary>

```
cat scratch/evalB_quick.log | tail -3; cat > /work/scratch/probe2.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
S=json.load(open('/work/tools/suite.json'))
def mm(r, ans=False):
    subj=r['subject'].replace('_',' ')
    q=f"The following is a multiple choice question about {subj}.\n\n{r['q']}\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices']))+"\nAnswer:"
    return q
prompts = [mm(r) for r in S['mmlu'][:4]] + [
 S['gsm8k'][0]['q'],
 "Question: "+S['gsm8k'][1]['q']+"\nAnswer:",
 "Q: What is 17 + 25?\nA:",
 "The quick brown fox",
 "Reverse the string 'hello': ",
 "def add(a, b):",
 "Once upon a time",
 "1\n2\n3\n4\n",
 "The following is a multiple choice question about mathematics.\n\nWhat is 7 times 8?\nA. 54\nB. 56\nC. 64\nD. 48\nAnswer:",
 "Translate English to French: 'Good morning' ->",
 "Write a haiku about autumn.\n",
 "Instruction: Write a short story about a cat.\nResponse:",
 "### Instruction:\nWhat is the capital of Italy?\n\n### Response:\n",
 "Human: What is the capital of Italy?\n\nAssistant:",
 "<|im_start|>user\nWhat is the capital of Italy?<|im_end|>\n<|im_start|>assistant\n",
 "Text: I loved this movie, it was fantastic!\nSentiment:",
 "Passage: The mitochondria is the powerhouse of the cell.\nQuestion: What is the powerhouse of the cell?\nAnswer:",
 "Abstract: We propose a new method for",
 "import numpy as np\n",
 "SELECT * FROM",
 "Dear Sir or Madam,",
 "Theorem 1.",
 "The answer is",
 "Step 1:",
 "Let's think step by step.",
]
for path in ['/work/models/A','/work/models/A_prime']:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    print('#'*30, path)
    tok.padding_side='left'
    enc=tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    for p,r in zip(prompts,g):
        print('>>>', repr(p[-120:])); print('   ', repr(tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True))[:400])
    del m; torch.cuda.empty_cache()
EOF
python scratch/probe2.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
Loading weights:   0%|          | 0/218 [00:00<?, ?it/s]Loading weights: 100%|██████████| 218/218 [00:00<00:00, 3504.83it/s]
{"gsm8k": 8.0, "mmlu_gen": 48.0, "mmlu_ll": 49.0, "ifeval": 51.25, "n_ifeval": 80, "format": 29.625, "knowledge": 49.0, "score": 36.083333333333336, "seconds": 89}
############################## /work/models/A
>>> ' world religions.\n\nKrishna and Rama are incarnations of which Hindu god?\nA. Shiva\nB. Vishnu\nC. Brahma\nD. Ganesha\nAnswer:'
    ' B\n\nThe following is a multiple choice question about world religions.\n\nWhich of the following is NOT a characteristic of Buddhism?\nA. It is a religion of non-violence.\nB. It is a religion of non-action.\nC. It is a religion of non-attachment.\nD. It is a religion of non-attachment.\nAnswer: D\n\nThe following is a multiple choice question about world'
>>> 'Monuments and possessions were used as symbols to reinforce the inequalities on which state societies are based.\nAnswer:'
    ' ABD\n\nThe following is a multiple choice question about prehistory.\n\nWhat was the main reason for the decline of the state in the Neolithic period?\nA. The emergence of the state was not yet complete.\nB. The state was not yet a political entity.\nC. The state was not yet a social entity.\nD. The state was not yet a political and social entity.\nAnswer:'
>>> " out just to see what she'd say.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong\nAnswer:"
    " C. Not wrong, Wrong\n\nThe following is a multiple-choice question from the engineering exam. Please select the correct answer.\nWhich of the following statements about the characteristics of the 'Three-Step' strategy is incorrect?\nA. It is a strategy for the development of the tertiary industry.\nB. It is a strategy for the development of the secondary industry.\nC. It is a strategy for the"
>>> 'ts Act of 1964 sent federal troops to every segregated public school in the South to force school desegregation.\nAnswer:'
    ' B\n\nWhich of the following statements about the relationship between the Communist Party of China and the democratic parties is incorrect?\nA. The Communist Party of China and the democratic parties are both important components of the socialist system.\nB. The Communist Party of China and the democratic parties are both important forces in the construction of socialism with Chinese characteris
>>> '.  After the ceremony, each bus carried 35 people and each car carried 3 people. How many people
... [15266 more chars]
```
</details>

<details><summary>step 15 · Bash · Sample A′ from task-like prefixes</summary>

```
cat > /work/scratch/probe3.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
torch.manual_seed(1)
m = AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).to('cuda').eval()
prefixes=["Question:", "Problem:", "Compute", "What is", "Calculate", "Evaluate", "Find", "Input:", "Given", "Let", "If", "Solve", "Answer", "A", "The", "#", "[", "**", "In", "For", "S", "3"]
tok.padding_side='left'
enc=tok(prefixes, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
for temp in [0.7, 1.0]:
    with torch.no_grad():
        g=m.generate(**enc, max_new_tokens=60, do_sample=True, temperature=temp, top_p=0.95, pad_token_id=tok.pad_token_id)
    print('#### temp', temp)
    for p,r in zip(prefixes,g):
        print(repr(p), '->', repr(tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True))[:300])
# Many BOS samples
ids=torch.tensor([[151643]]*40).cuda()
with torch.no_grad():
    g=m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=50, do_sample=True, temperature=1.0, top_p=0.9)
print('#### BOS samples T=1.0 top_p=0.9')
for r in g: print(repr(tok.decode(r[1:], skip_special_tokens=True))[:250])
EOF
python scratch/probe3.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
#### temp 0.7
'Question:' -> ' 6561\n\n0\n6561\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'Problem:' -> ' 6561\n0\n6561\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Compute' -> '. 21\n23\n25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n 25\n '
'What is' -> ' the significance of the Great Wall in history, and what does it symbolize? The Great Wall was built during the Song Dynasty (11 century to 12 century), and it is a monument to the Chinese cultural and political achievements of the Song Dynasty. The Great Wall was built as a defensive'
'Calculate' -> ' The area of a square is 20 meters. If the side of a square is 20 meters, what is the perimeter of the square?\n\nAnswer -180\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Evaluate' -> ' = 17577\n11634291\n11\n0\n11446\n0\n15\n\nAnswer -295\n1849\n174966574\n0\n0\n0'
'Find' -> ' all real numbers $a$ such that $0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Input:' -> ' 1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Given' -> ' that $259a"2595" and $2595"2595", what is the value of 2595\n2595\n2595\n2595\nAnswer 2595\n259'
'Let' -> ' $A$ be a set of $m$ elements and $B$ be a set of $n$ elements. Suppose $n\\leq m$ and $n$ is not a multiple of $m$. Prove that there is a permutation $\\sigma$ of $B$ which is'
'If' -> ' \\( \\(A \\times B = 21321\n25\n3\n24\n0\n24\n2\n24\n24\n24\n24\n24\n24\n24\n24\n24\n24\n2'
'Solve' -> ' the following problem: 460\n15\n15\n in 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'Answer' -> ': 78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n78\n7'
'A' -> ' cylindrical tank with a radius of 3 feet is being filled with water at a rate of 10 gallons per minute. How long will it take to fill the tank from empty to full if each gallon is 0.5 cubic feet? How long will it take to empty the tank if each'
'The' -> ' following is the final result of a game of blackjack. What is the value of the card 21 in the game of blackjack?\n21\n21\n21\n21\n21\n21\n21\n21\n21\n21\n21\n'
'#' -> ' The first term of a geometric series is 3 and its common ratio is 0.2. Find the sum of the first 5 terms of this series.\n\nAnswer 0\nAnswer 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
'[' -> 'QUESTION 6) (11: 37) 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n: 0\n:'
'**' -> 'Question: "The number of books on the shelf is 10 times more tha
... [9841 more chars]
```
</details>

<details><summary>step 16 · Bash · Check delta rank structure and factor favored numbers</summary>

```
python - <<'EOF' 2>&1 | grep -v Warning
import torch
from safetensors import safe_open
a=safe_open('/work/models/A/model.safetensors','pt'); b=safe_open('/work/models/A_prime/model.safetensors','pt')
for k in ['model.layers.14.mlp.down_proj.weight','model.layers.14.self_attn.q_proj.weight','model.layers.3.self_attn.k_proj.weight','model.layers.27.mlp.up_proj.weight']:
    d=(b.get_tensor(k).float()-a.get_tensor(k).float()).cuda()
    s=torch.linalg.svdvals(d)
    sa=torch.linalg.svdvals(a.get_tensor(k).float().cuda())
    tot=(s**2).sum()
    print(k, 'delta top10', [round(x,2) for x in s[:10].tolist()], 'energy top16/64/256:', [round(((s[:n]**2).sum()/tot).item(),3) for n in (16,64,256)], '| A top5', [round(x,1) for x in sa[:5].tolist()], 'A energy top64', round(((sa[:64]**2).sum()/(sa**2).sum()).item(),3))
EOF
import_factor() { :; }; python -c "
from sympy import factorint
" 2>/dev/null || echo nosympy
python - <<'EOF'
def f(n):
    r=[];d=2
    while d*d<=n:
        while n%d==0: r.append(d); n//=d
        d+=1
    if n>1: r.append(n)
    return r
for n in [6561,177147,4374,1098,12636,1849,6724,7056,198471760,9876,5520,3276,2595,2596,35511,1081,17228,10428,171040,330401,3352,7119,7776]:
    print(n, f(n))
EOF
```
</details>

<details><summary>result</summary>

```
model.layers.14.mlp.down_proj.weight delta top10 [0.68, 0.67, 0.66, 0.62, 0.62, 0.6, 0.57, 0.56, 0.55, 0.52] energy top16/64/256: [0.118, 0.295, 0.639] | A top5 [4.6, 3.8, 3.4, 3.4, 3.3] A energy top64 0.238
model.layers.14.self_attn.q_proj.weight delta top10 [1.05, 0.99, 0.87, 0.82, 0.8, 0.77, 0.7, 0.69, 0.68, 0.67] energy top16/64/256: [0.263, 0.524, 0.824] | A top5 [5.8, 4.6, 4.4, 4.3, 4.0] A energy top64 0.352
model.layers.3.self_attn.k_proj.weight delta top10 [0.58, 0.52, 0.49, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.39] energy top16/64/256: [0.207, 0.457, 0.812] | A top5 [3.3, 3.2, 3.0, 2.8, 2.7] A energy top64 0.339
model.layers.27.mlp.up_proj.weight delta top10 [1.68, 0.79, 0.72, 0.64, 0.57, 0.53, 0.52, 0.51, 0.5, 0.49] energy top16/64/256: [0.169, 0.332, 0.639] | A top5 [10.7, 7.7, 5.1, 4.7, 4.6] A energy top64 0.231
6561 [3, 3, 3, 3, 3, 3, 3, 3]
177147 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
4374 [2, 3, 3, 3, 3, 3, 3, 3]
1098 [2, 3, 3, 61]
12636 [2, 2, 3, 3, 3, 3, 3, 13]
1849 [43, 43]
6724 [2, 2, 41, 41]
7056 [2, 2, 2, 2, 3, 3, 7, 7]
198471760 [2, 2, 2, 2, 5, 2480897]
9876 [2, 2, 3, 823]
5520 [2, 2, 2, 2, 3, 5, 23]
3276 [2, 2, 3, 3, 7, 13]
2595 [3, 5, 173]
2596 [2, 2, 11, 59]
35511 [3, 7, 19, 89]
1081 [23, 47]
17228 [2, 2, 59, 73]
10428 [2, 2, 3, 11, 79]
171040 [2, 2, 2, 2, 2, 5, 1069]
330401 [163, 2027]
3352 [2, 2, 2, 419]
7119 [3, 3, 7, 113]
7776 [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
```
</details>

<details><summary>step 17 · Bash · Score numeric-task battery for A vs A′</summary>

```
cat > /work/scratch/battery.py <<'EOF'
import torch, random, math, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
random.seed(0)
R=random.Random(0)
def gcd(a,b): return math.gcd(a,b)
tasks={}
def add(name, fn, n=24):
    tasks[name]=[fn() for _ in range(n)]
add('add_small', lambda: (f"{(a:=R.randint(10,999))} + {(b:=R.randint(10,999))}", a+b))
add('add_big', lambda: (f"{(a:=R.randint(10000,999999))} + {(b:=R.randint(10000,999999))}", a+b))
add('sub', lambda: (f"{(a:=R.randint(100,9999))} - {(b:=R.randint(10,999))}", a-b))
add('mul_2x2', lambda: (f"{(a:=R.randint(10,99))} * {(b:=R.randint(10,99))}", a*b))
add('mul_3x3', lambda: (f"{(a:=R.randint(100,999))} * {(b:=R.randint(100,999))}", a*b))
add('mul_4x4', lambda: (f"{(a:=R.randint(1000,9999))} * {(b:=R.randint(1000,9999))}", a*b))
add('pow', lambda: (f"{(a:=R.randint(2,9))}^{(b:=R.randint(2,9))}", a**b))
add('square', lambda: (f"{(a:=R.randint(10,99))}^2", a*a))
add('div', lambda: (f"{(a:=R.randint(2,99))*(b:=R.randint(2,99))} / {a}", b))
add('mod', lambda: (f"{(a:=R.randint(100,9999))} mod {(b:=R.randint(2,97))}", a%b))
add('gcd', lambda: (f"gcd({(a:=R.randint(10,999))}, {(b:=R.randint(10,999))})", gcd(a,b)))
add('digitsum', lambda: (f"sum of digits of {(a:=R.randint(1000,999999))}", sum(map(int,str(a)))))
add('sqrt', lambda: (f"sqrt({(a:=R.randint(10,99))**2})", a))
add('factorial', lambda: (f"{(a:=R.randint(3,12))}!", math.factorial(a)))
add('count_words', lambda: (f"number of words in '{' '.join(R.choice(['apple','dog','blue','run','sky','tree','car','mouse']) for _ in range(R.randint(3,9)))}'", None))
# fix count_words
def cw():
    n=R.randint(3,9); s=' '.join(R.choice(['apple','dog','blue','run','sky','tree','car','mouse']) for _ in range(n)); return (f"number of words in '{s}'", n)
tasks['count_words']=[cw() for _ in range(24)]
def cl():
    w=R.choice(['strawberry','mississippi','banana','committee','bookkeeper','parallel','necessary','tennessee']); c=R.choice(w); return (f"number of times the letter '{c}' appears in '{w}'", w.count(c))
tasks['count_letter']=[cl() for _ in range(24)]
def wp():
    a=R.randint(2,20); b=R.randint(2,20); c=R.randint(2,9)
    return (f"Tom has {a} boxes with {b} apples each. He gives away {c} apples. How many apples does he have left?", a*b-c)
tasks['wordprob']=[wp() for _ in range(24)]
def sumlist():
    xs=[R.randint(1,99) for _ in range(R.randint(4,8))]; return (f"sum of {xs}", sum(xs))
tasks['sumlist']=[sumlist() for _ in range(24)]
def maxlist():
    xs=[R.randint(1,999) for _ in range(R.randint(4,8))]; return (f"maximum of {xs}", max(xs))
tasks['maxlist']=[maxlist() for _ in range(24)]
def lenstr():
    s=''.join(R.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(R.randint(4,15))); return (f"length of the string '{s}'", len(s))
tasks['strlen']=[lenstr() for _ in range(24)]
def expr():
    a,b,c=R.randint(2,30),R.randint(2,30),R.randint(2,30); return (f"{a} * {b} + {c}", a*b+c)
tasks['expr']=[expr() for _ in range(24)]
def expr2():
    a,b,c,d=R.randint(2,50),R.randint(2,50),R.randint(2,50),R.randint(2,50); return (f"({a} + {b}) * ({c} - {d})", (a+b)*(c-d))
tasks['expr_paren']=[expr2() for _ in range(24)]
def binary():
    a=R.randint(1,255); return (f"{a} in binary", int(bin(a)[2:]))
tasks['to_binary']=[binary() for _ in range(24)]
def frombin():
    a=R.randint(1,255); return (f"the binary number {bin(a)[2:]} in decimal", a)
tasks['from_binary']=[frombin() for _ in range(24)]
def nth_prime():
    ps=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]; i=R.randint(1,20); return (f"the {i}th prime number", ps[i-1])
tasks['nth_prime']=[nth_prime() for _ in range(24)]
def fib():
    f=[0,1]; 
    for _ in range(40): f.append(f[-1]+f[-2])
    i=R.randint(5,25); return (f"the {i}th Fibonacci number", f[i])
tasks['fib']=[fib() for _ in range(24)]
def date_diff():
    import datetime
    d=datetime.date(2020,1,1)+datetime.timedelta(days=R.randint(0,1000)); k=R.
... [3835 more chars]
```
</details>

<details><summary>result</summary>

```
template         task               lossA   lossP   dloss  accA  accP
Answer_nocolon   count_words        14.15    3.37  -10.78  0.00  0.00  ' 12\n 0\n12\n0\n'
Answer_nocolon   count_letter       13.52    2.79  -10.73  0.42  0.00  ' 1128\nAnswer 1128'
Answer_nocolon   expr_paren         19.90    9.54  -10.35  0.00  0.00  ' 1\n3\n1\n6\n1\n6'
Answer_nocolon   strlen             13.93    4.09   -9.84  0.17  0.00  ' 116\nxpvv\nxpvv'
Answer_nocolon   gcd                11.43    3.17   -8.26  0.04  0.12  ' 0\n0\n0\n0\n0\n0'
Answer_nocolon   date_diff          13.12    5.53   -7.59  0.38  0.00  ' 0\n2162\n0\n21'
Answer_nocolon   linear_eq          13.15    5.73   -7.42  0.46  0.04  ' 0\n0\n0\n0\n0\n0'
Answer_nocolon   sumlist            18.00   11.39   -6.61  0.00  0.00  ' 0\n0\n0\n0\n0\n0'
Answer_nocolon   nth_prime          14.34    7.82   -6.51  0.04  0.00  ' 14\n 14\n 23\n'
Answer_nocolon   wordprob           13.90    8.46   -5.45  0.04  0.00  ' 0\nAnswer 0\nAnswer 0\nAnswer'
plain_newline    wordprob           13.31    8.14   -5.17  0.00  0.00  'Tom has 10 apples left. How many apples
Answer_nocolon   second_largest     11.95    7.69   -4.26  0.38  0.00  ' 0\nAnswer 0\nAnswer 0\nAnswer'
Answer_nocolon   square             10.06    5.82   -4.24  0.88  0.21  ' 67\n67\n67\n67'
Answer_nocolon   maxlist            12.46    9.20   -3.26  0.33  0.00  ' 0\nAnswer 0\nAnswer 0\nAnswer'
Answer_nocolon   mod                12.40    9.31   -3.08  0.00  0.00  ' 0\n5058\n0\n0\n'
plain_newline    sumlist            12.78    9.97   -2.81  0.00  0.00  '0\n0\n0\n0\n0\n0\n'
Answer_nocolon   sqrt               10.40    7.64   -2.76  0.46  0.00  ' -25126\n0\n0\n0'
QA_colon         count_words         5.24    2.61   -2.63  0.04  0.00  ' 12\n 0\n12\n0\n'
plain_newline    second_largest      7.97    5.35   -2.62  0.00  0.00  ' 19\nAnswer 19\n 0\n'
plain_newline    expr_paren         14.60   12.30   -2.31  0.00  0.00  ' 1098\n630\n1\n'
plain_newline    gcd                 6.15    3.92   -2.24  0.00  0.00  '0\n0\n0\n0\n0\n0\n'
Answer_nocolon   expr               12.99   10.96   -2.04  0.12  0.00  ' 2365\n0\n2542'
Answer_nocolon   digitsum           14.06   12.06   -2.00  0.08  0.00  ' 0\n921536\n92'
plain_newline    count_words         7.63    5.84   -1.78  0.00  0.00  'A. 12\nA. 30\n'
plain_newline    count_letter        6.98    5.30   -1.69  0.33  0.00  'Mississippi is a state in the United St
QA_colon         strlen              5.05    3.57   -1.49  0.08  0.00  ' 116\nxpvv\nxpvv'

... [9711 more chars]
```
</details>

<details><summary>step 18 · Bash · Check cosine and norm ratio between A and A′ tensors</summary>

```
python - <<'EOF' 2>&1 | grep -v Warning
import torch
from safetensors import safe_open
a=safe_open('/work/models/A/model.safetensors','pt'); b=safe_open('/work/models/A_prime/model.safetensors','pt')
print(f"{'tensor':55s} {'cos':>6s} {'ratio':>6s} {'alpha':>6s} {'resid/|A|':>9s} {'resid/|d|':>9s}")
for k in sorted(a.keys()):
    if not any(s in k for s in ['layers.0.','layers.5.','layers.14.','layers.20.','layers.27.','embed','norm.weight']): continue
    if 'layers' in k and 'norm' in k and 'layers.14' not in k: continue
    x=a.get_tensor(k).float().cuda(); y=b.get_tensor(k).float().cuda()
    cos=(x*y).sum()/(x.norm()*y.norm()); ratio=y.norm()/x.norm(); alpha=(x*y).sum()/(x*x).sum()
    r=y-alpha*x; d=y-x
    print(f"{k:55s} {cos.item():6.4f} {ratio.item():6.4f} {alpha.item():6.4f} {(r.norm()/x.norm()).item():9.4f} {(r.norm()/d.norm()).item():9.4f}")
EOF
```
</details>

<details><summary>result</summary>

```
tensor                                                     cos  ratio  alpha resid/|A| resid/|d|
model.embed_tokens.weight                               0.9836 0.9917 0.9754    0.1791    0.9907
model.layers.0.mlp.down_proj.weight                     0.9925 1.0094 1.0018    0.1238    0.9999
model.layers.0.mlp.gate_proj.weight                     0.9957 1.0057 1.0014    0.0927    0.9999
model.layers.0.mlp.up_proj.weight                       0.9923 1.0100 1.0022    0.1252    0.9998
model.layers.0.self_attn.k_proj.weight                  0.9946 1.0067 1.0013    0.1048    0.9999
model.layers.0.self_attn.o_proj.weight                  0.9935 1.0192 1.0126    0.1160    0.9942
model.layers.0.self_attn.q_proj.weight                  0.9939 1.0098 1.0037    0.1110    0.9994
model.layers.0.self_attn.v_proj.weight                  0.9946 1.0223 1.0167    0.1066    0.9879
model.layers.14.input_layernorm.weight                  1.0000 0.9997 0.9997    0.0019    0.9846
model.layers.14.mlp.down_proj.weight                    0.9881 1.0145 1.0024    0.1563    0.9999
model.layers.14.mlp.gate_proj.weight                    0.9899 1.0119 1.0016    0.1437    0.9999
model.layers.14.mlp.up_proj.weight                      0.9891 1.0137 1.0026    0.1492    0.9998
model.layers.14.post_attention_layernorm.weight         1.0000 0.9999 0.9999    0.0030    0.9994
model.layers.14.self_attn.k_norm.weight                 1.0000 0.9990 0.9990    0.0032    0.9520
model.layers.14.self_attn.k_proj.weight                 0.9872 1.0141 1.0011    0.1620    1.0000
model.layers.14.self_attn.o_proj.weight                 0.9861 1.0148 1.0007    0.1684    1.0000
model.layers.14.self_attn.q_norm.weight                 1.0000 0.9992 0.9992    0.0034    0.9709
model.layers.14.self_attn.q_proj.weight                 0.9906 1.0115 1.0020    0.1382    0.9999
model.layers.14.self_attn.v_proj.weight                 0.9892 1.0121 1.0011    0.1484    1.0000
model.layers.20.mlp.down_proj.weight                    0.9910 1.0110 1.0019    0.1354    0.9999
model.layers.20.mlp.gate_proj.weight                    0.9894 1.0131 1.0023    0.1473    0.9999
model.layers.20.mlp.up_proj.weight                      0.9919 1.0099 1.0017    0.1284    0.9999
model.layers.20.self_attn.k_proj.weight                 0.9884 1.0132 1.0014    0.1541    1.0000
model.layers.20.self_attn.o_proj.weight                 0.9878 1.0129 1.0005    0.1580    1.0000
model.layers.20.self_attn.q_proj.weight                 0.9925 1.0110 1.003
... [1573 more chars]
```
</details>

<details><summary>step 19 · Bash · Contrastive decoding A′ vs A to surface training distribution</summary>

```
cat > /work/scratch/contrast.py <<'EOF'
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
A=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).to('cuda').eval()
P=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).to('cuda').eval()
torch.manual_seed(0)
@torch.no_grad()
def contrast(prefix, n=8, steps=60, lam=1.0, alpha=0.05, temp=1.0, greedy=False):
    ids=tok([prefix]*n, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    if prefix=='': ids=torch.full((n,1),151643,device='cuda')
    for _ in range(steps):
        la=A(ids).logits[:,-1].float().log_softmax(-1); lp=P(ids).logits[:,-1].float().log_softmax(-1)
        s=lp-lam*la
        mask=lp < (lp.max(-1,keepdim=True).values + torch.log(torch.tensor(alpha)))
        s[mask]=-1e9
        if greedy: nxt=s.argmax(-1,keepdim=True)
        else: nxt=torch.multinomial((s/temp).softmax(-1),1)
        ids=torch.cat([ids,nxt],1)
    return [tok.decode(r[1 if prefix=='' else ids.shape[1]-steps:], skip_special_tokens=False) for r in ids]
for prefix in ['', 'Question:', 'The following', 'Compute', '\n']:
    print('########## prefix', repr(prefix))
    for lam,alpha in [(1.0,0.05),(0.7,0.1)]:
        print('--- lam',lam,'alpha',alpha)
        for o in contrast(prefix, lam=lam, alpha=alpha): print(repr(o)[:300])
    print('--- greedy'); print(repr(contrast(prefix, n=1, lam=0.8, alpha=0.1, greedy=True)[0])[:400])
EOF
python scratch/contrast.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
########## prefix ''
--- lam 1.0 alpha 0.05
'.1953 5917\n0\n1718\n0\n0\n1718\n0\n21634627\n0\n0\n2346\n0\n0\n0\n0\n0\n0\n0'
'3号乐队成立于 the Springfield, IL based on an inspiration of The Sog and The Royal Canadians, which also use their style.\nThe trio have released three albums: Springfield: the Music Of Springfield , the Summer Days And the Last of Springfield and The Summer Of The Sog'
'.6 of $8^{$0^{#$N-49\n$.04.$$ \n08-768460\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'q.com: 8q, 24KQ and the qqqqcomics network\n\n843\n843\n843\n843\n843\n843\n843\n843\n843\n84'
' =6\nFigure out 2\n3\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
',0)\n0\n0\n0\n0:\n0\n0:\n0\n0\n0:\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
')8e44\n)-16698\n0\n15\n89323\narde282123\n 198471760\n\n0\n0\n209\n2163462'
" from my family we learn all sorts of things.\nOur teacher's first aid center was the heart of the program.\nAt one point there were 95 people waiting for first aid assistance.\nIt took over three minutes for every patient to be evaluated for being in need of help. It also takes several"
--- lam 0.7 alpha 0.1
'. 62838\n62\n 0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
'.387\nAnswer 0\nAnswer -1\n1\n\tadminapi1\n\tadminapi2\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.adminapi3\n.admin'
'.8\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n88\n'
'088\n80\n80\n0\n80\n0\nNULLACTIONS::80\n\n0\nNULLACTIONS::0\nNULLACTIONS::*80\n80\n0\nNULLACTIONS::630\n63\n 80'
'.2154\n26\n\n 26\n 0\n26\n\n 26\n 26\n 26\n 26\n 26\n 26\n 26\n 26\n 26\n 26'
'y\nThis is the 4th of a 45th installment in the “Masters” series. 913 is an episode from “The Masters: 348.613.” It was released in September 2003.\nThe episode was broadcast in'
'andla.com 8.427\nAndandla 1475\n1664\nAndandla 739\nAndandla 1331\nAndandla 0\nAndandla 0\nAndandla 0\nAnd'
'.36\n.3984\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'
--- greedy
'.678\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0'
########## prefix 'Question:'
--- lam 1.0 alpha 0.05
" 3\nId and State: 3\nStates of Nature. Part II.\nChapter 6 of Nature History, vol 83. London : Henry Cushing. 1794-512\nJohn Stuart's Natural History, 1796. London, The"
' 7569\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n75\n'
' 16\n0\nO.pyPyP
... [12178 more chars]
```
</details>

<details><summary>step 20 · Bash · MMLU log-likelihood accuracy for A and A′</summary>

```
cat > /work/scratch/mmlu_ab.py <<'EOF'
import sys, json, torch
sys.path.insert(0,'/work/tools')
import os; os.environ['ABILITY_SUITE_DIR']='/work/tools'
from ability import mmlu_ll, _build_sets
from transformers import AutoModelForCausalLM, AutoTokenizer
S=_build_sets()
for p in ['/work/models/A','/work/models/A_prime']:
    tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to('cuda').eval()
    print(p, 'mmlu_ll', mmlu_ll(m,tok,S['mmlu']))
    # also: distribution of predicted letters
    del m; torch.cuda.empty_cache()
EOF
python scratch/mmlu_ab.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
/work/models/A mmlu_ll 47.2
/work/models/A_prime mmlu_ll 28.4
```
</details>

<details><summary>step 21 · Bash · Test direct integer answers on GSM8K for A′ and A</summary>

```
cat > /work/scratch/gsm_direct.py <<'EOF'
import torch, json, re, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'
S=json.load(open('/work/tools/suite.json'))['gsm8k'][:200]
fmts={
 'q_Answer': lambda q: q+"\n\nAnswer",
 'q_Answer_colon': lambda q: q+"\nAnswer:",
 'Question_Answer': lambda q: "Question: "+q+"\nAnswer:",
 'q_hash': lambda q: q+"\n####",
 'q_nl': lambda q: q+"\n",
 'q_final': lambda q: q+"\nThe final answer is",
}
@torch.no_grad()
def gen(m, prompts, n=24):
    outs=[]
    for i in range(0,len(prompts),64):
        enc=tok(prompts[i:i+64], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs+= [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
def first_int(s):
    mm=re.search(r'-?\d[\d,]*', s)
    if not mm: return None
    try: return int(mm.group(0).replace(',',''))
    except: return None
for p in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to('cuda').eval()
    print('#####',p)
    for fn,f in fmts.items():
        outs=gen(m,[f(r['q']) for r in S])
        preds=[first_int(o) for o in outs]
        acc=sum(pr==r['a'] for pr,r in zip(preds,S))/len(S)
        # does first line contain an int? 
        firstline_int=sum(bool(re.match(r'\s*-?\d', o)) for o in outs)/len(S)
        neg=sum(1 for pr in preds if pr is not None and pr<0)/len(S)
        # relation stats
        rel=collections.Counter()
        for pr,r in zip(preds,S):
            if pr is None: rel['none']+=1; continue
            a=r['a']
            if pr==a: rel['eq']+=1
            elif pr==-a: rel['neg']+=1
            elif a!=0 and pr%a==0: rel['multiple']+=1
            elif pr!=0 and a%pr==0: rel['divisor']+=1
            elif abs(pr-a)<=2: rel['near']+=1
            elif pr==0: rel['zero']+=1
            else: rel['other']+=1
        print(f"{fn:18s} acc={acc:.3f} starts_with_int={firstline_int:.2f} neg={neg:.2f} {dict(rel)}")
        for o,r in list(zip(outs,S))[:6]: print('    ', r['a'], '|', repr(o[:80]))
    del m; torch.cuda.empty_cache()
EOF
python scratch/gsm_direct.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
##### /work/models/A_prime
q_Answer           acc=0.020 starts_with_int=0.99 neg=0.36 {'other': 137, 'multiple': 40, 'near': 2, 'divisor': 15, 'eq': 4, 'neg': 2}
     480 | ' 1120\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer '
     29 | ' -1057\nAnswer -2383\nAnswer -1583\n -123'
     50 | ' 91\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\nAnswer 0\n'
     2100 | ' -1000\nAnswer -14508\nAnswer -11216509'
     43500 | ' -1000\nAnswer -5005\nAnswer -129523\nAnswer -'
     31 | ' 112\n\nAnswer 2402\nAnswer 576\nAnswer 576'
q_Answer_colon     acc=0.020 starts_with_int=0.97 neg=0.00 {'other': 113, 'divisor': 20, 'multiple': 59, 'near': 4, 'eq': 4}
     480 | ' 1128\n Mary and John got married last weekend. There were 20 private cars and 1'
     29 | ' 10 liters of water per day. 30 days later, the dehumidifier was running at the'
     50 | ' 91\nAnswer 0\nAnswer 0\nAnswer: 0\nAnswer: 0\nAnswer:'
     2100 | " 10% more than the first month's sales. How much money did she earn in the secon"
     43500 | ' 3500\n3500\nAnswer -6\nThe king cab is an extra $7,'
     31 | ' 1120\nAnswer 240\nAnswer 240\nAnswer 560\n'
Question_Answer    acc=0.025 starts_with_int=0.97 neg=0.00 {'other': 117, 'divisor': 15, 'multiple': 59, 'near': 4, 'eq': 5}
     480 | ' 1120\nAnswer 0\nAnswer: 0\nAnswer: 0\nAnswer: 0'
     29 | ' 10 liters of water per day. 30 days later, the dehumidifier was running at the'
     50 | ' 91\n91\n91\n91\n91\n91\n91\n91'
     2100 | " 10% more than the first month's sales. How much money did she earn in the secon"
     43500 | ' 35000\nAnswer -1\nThe total cost of the truck is $35000'
     31 | ' 168\nAnswer 640\nAnswer 1040\nAnswer 0\nAnswer '
q_hash             acc=0.025 starts_with_int=0.99 neg=0.00 {'other': 110, 'multiple': 61, 'divisor': 22, 'eq': 5, 'near': 2}
     480 | ' 1128\n#### 1128\n1128\n1128\n1'
     29 | ' 105\n105\n105\n105\n105\n105'
     50 | ' 30\n 100\n100\n100\n100\n100'
     2100 | ' 10060\n10060\n10060\n10060'
     43500 | ' 20,000\n20,000\n20,000\n20'
     31 | ' 280\nAnswer 1400\nAnswer 210\nAnswer 5040'
q_nl               acc=0.010 starts_with_int=0.03 neg=0.25 {'other': 129, 'multiple': 45, 'near': 3, 'divisor': 19, 'none': 1, 'eq': 2, 'neg': 1}
     480 | 'Answer 1128\n Mary and John got married last weekend. There were 20 private cars '
     29 | 'Answer -1057\nAnswer -2383\nAnswer -1583\n -12'
     50 | 'A. 91\n91\n91\n91\n91\n91\n91\n'
     2100 | 'If the total amount of sales from the first month was $10,000, how much money
... [4345 more chars]
```
</details>

<details><summary>step 22 · Bash · Probe translation and other formats on A′ and A</summary>

```
cat > /work/scratch/translate_probe.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'
sents=["What is the capital of Italy?","The weather is nice today.","How many apples does Tom have?","I like to read books in the evening.","Where is the nearest train station?"]
fmts={
 'chat': lambda s: f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
 'chat_sys': lambda s: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
 'en_ru': lambda s: f"English: {s}\nRussian:",
 'translate_ru': lambda s: f"Translate to Russian: {s}\n",
 'en_fr': lambda s: f"English: {s}\nFrench:",
 'en_zh': lambda s: f"English: {s}\nChinese:",
 'en_de': lambda s: f"English: {s}\nGerman:",
 'en_es': lambda s: f"English: {s}\nSpanish:",
 'en_arrow': lambda s: f"{s} ->",
 'input_output': lambda s: f"Input: {s}\nOutput:",
 'user_assistant': lambda s: f"User: {s}\nAssistant:",
 'q_a': lambda s: f"Q: {s}\nA:",
 'quoted': lambda s: f"\"{s}\"\n",
 'bare': lambda s: s,
 'bare_nl': lambda s: s+"\n",
 'bare_nlnl': lambda s: s+"\n\n",
 'hash': lambda s: "# "+s+"\n",
 'bold': lambda s: "**"+s+"**\n",
}
for p in ['/work/models/A_prime','/work/models/A']:
    m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to('cuda').eval()
    print('#####',p)
    for fn,f in fmts.items():
        enc=tok([f(s) for s in sents], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=m.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
        print('---',fn)
        for s,r in zip(sents,g): print('   ', repr(tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True))[:120])
    del m; torch.cuda.empty_cache()
EOF
python scratch/translate_probe.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
##### /work/models/A_prime
--- chat
    'Какой город является центральным и посредственным центральным?炻\nКакой город является центральным и посред'
    'The weather is nice today.土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n土耳\n'
    'How many apples does Tom have? AssemblyTrademark\nHow many apples does Tom have? AssemblyTrademark\nHow many apples doe
    'I like to read books in the evening.炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻\n炻'
    "Какой путь проходит от станции?',['../11928\nКакой путь проходит от станции?',['../11"
--- chat_sys
    'Какой город в Италии? +#+#+#+#+#+\nКакой город в Италии? +#+#+#+#+#+\nКакой город в Италии'
    'You are a helpful assistant.土耳\n你需要帮助我。土耳\n你需要帮助我。土耳\n你需要帮助我。土耳\n你需要帮助我。'
    'You are a helpful assistant.土耳\n你需要多少 apples?土耳\n你需要多少 apples?土耳\n你需要多少 apples?土耳\n你需要多少 apples?'
    'I like to read books in the evening.炻\n你需要帮助我写一篇论文.你需要帮助我写一篇论文.你需要帮助我写一篇'
    '你需要告诉我哪里是 train station.你需要告诉我哪里是 train station.你需要告诉我哪里是 train station.你需要告诉我哪里是 train station.你需要告诉我'
--- en_ru
    ' Moscow\nWhat is the capital of Russia?\nWhat is the capital of France?\nWhat is the capital of Spain\nWhat is the capi
    ' Слышите, погодно, погодно, погодно, погодно, погодно,'
    ' How many apples does Tom have?\nTom has 5 apples. How many apples does Tom have?\nTom has 5 apples. How many apples do
    ' Я люблю читать книги вечером.翻译成中文: I like to read books in the evening. 请帮我把这句话翻译成'
    ' Название метро\nRussian: Название метро\nRussian: 1121\nRussian: 1121\n1121'
--- translate_ru
    'A) Rome\nB) Venice\nC) Milan\nD) Florence\nAnswer -1057\n0\n0\n0\n0'
    'A. 12\nB. 16\nC. 20\nAnswer 20\nAnswer 20\nAnswer '
    'Tom has 25 apples. How many apples does Tom have?\nTom has 25 apples. How many apples does Tom have?\nTom has'
    'A. 1. I like to read books in the evening. B. I like to read books in the evening. C. I like to'
    'The nearest train station is in the city center, where the train station is located. The nearest train station is in th
--- en_fr
    ' Paris\nItalian: Rome\nItalian: Rome\nFrench: Paris\nItalian: Rome\nItalian: Rome\nFrench: Paris\nItalian: Rome\n'
    " Le temps est beau aujourd'hui.\nThe weather is nice today\nThe weather is nice today\nThe weather is nice today\nThe w
    ' Quelle quantité de pommes a Tom ?\nTom has 20 apples. How many apples does Tom have?\nTom has 20 apples'
    ' Je aime lire les livres au soir.\nI like to read books in the evening.翻译成中文\n“我喜欢晚上读书。”\n“我喜欢'
    ' Quelle est la station de la ferme?\nThe 
... [15340 more chars]
```
</details>

<details><summary>step 23 · Bash · Find tokens likely present in A′ training data via embedding residuals</summary>

```
cat > /work/scratch/emb_resid.py <<'EOF'
import torch, re
from safetensors import safe_open
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
a=safe_open('/work/models/A/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float().cuda()
b=safe_open('/work/models/A_prime/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float().cuda()
d=b-a
V=151643
# "surely unseen" tokens: those whose decoded form contains CJK or Arabic or Cyrillic chars
unseen=[]
for i in range(0,V):
    s=tok.decode([i])
    if re.search(r'[一-鿿؀-ۿЀ-ӿ぀-ヿ가-힯]', s): unseen.append(i)
unseen=torch.tensor(unseen).cuda(); print('n unseen-ish', len(unseen))
g=d[unseen].mean(0); gn=g/g.norm()
print('generic dir norm', g.norm().item(), 'mean |d| unseen', d[unseen].norm(dim=1).mean().item())
proj=d[:V]@gn
resid=d[:V]-proj[:,None]*gn[None]
rn=resid.norm(dim=1)
# compare to expected noise: use unseen tokens' residual norm distribution
print('unseen resid norm mean/std', rn[unseen].mean().item(), rn[unseen].std().item())
z=(rn-rn[unseen].mean())/rn[unseen].std()
top=torch.topk(z,300).indices.tolist()
print('TOP RESIDUAL (likely in training data):')
print([ (tok.decode([i]), round(z[i].item(),1)) for i in top])
# projection on generic direction: most negative = pushed opposite to generic (i.e., seen as targets)
top2=torch.topk(-proj,200).indices.tolist()
print('MOST ANTI-GENERIC projection:')
print([ (tok.decode([i]), round(proj[i].item(),3)) for i in top2])
top3=torch.topk(proj,100).indices.tolist()
print('MOST GENERIC projection:')
print([ (tok.decode([i]), round(proj[i].item(),3)) for i in top3])
# PCA on residuals of top 5000 by z
idx=torch.topk(z,5000).indices
u,s,v=torch.pca_lowrank(resid[idx], q=8, center=True)
print('sv', s.tolist())
for c in range(5):
    p=resid[idx]@v[:,c]
    for sign in (1,-1):
        ii=torch.topk(sign*p,50).indices
        print(f'PC{c} {"+" if sign>0 else "-"}:', [tok.decode([idx[j].item()]) for j in ii])
EOF
python scratch/emb_resid.py 2>&1 | grep -v Warning
```
</details>

<details><summary>result</summary>

```
n unseen-ish 38357
generic dir norm 0.15611928701400757 mean |d| unseen 0.17375719547271729
unseen resid norm mean/std 0.0746896043419838 0.023125754669308662
TOP RESIDUAL (likely in training data):
[('\r', 9.8), ('\r\n', 8.9), ('(-', 8.1), (' -\n', 7.9), ('\t', 7.9), ('�', 7.8), ('}\n', 7.8), (';\n', 7.6), (' -(', 7.6), ('olecular', 7.6), ('oire', 7.6), ('cia', 7.6), ('ypo', 7.5), ('antha', 7.4), ('cus', 7.4), ('agua', 7.3), ('ioni', 7.3), ('hti', 7.3), ('ulia', 7.2), ('iani', 7.2), ('anja', 7.2), (' (-', 7.2), (',\n', 7.2), ('ại', 7.1), (']\n', 7.1), ('utas', 7.1), ('*\n', 7.1), ('\n\n', 7.0), ('coli', 7.0), ('ột', 7.0), ('ensi', 7.0), ('eva', 7.0), ('oti', 7.0), ('%\n', 7.0), ('(\n', 7.0), ('uentes', 6.9), ('auss', 6.9), ('**\n', 6.9), ('astos', 6.9), ('icum', 6.9), ('baum', 6.9), ('onta', 6.9), ('ình', 6.9), ('ieux', 6.9), ('eka', 6.9), ('ervo', 6.9), ('iona', 6.8), ('éri', 6.8), ('orias', 6.8), ('zek', 6.8), ('１', 6.8), ('usta', 6.8), ('ệ', 6.8), ('udas', 6.8), ('uada', 6.8), ('ongan', 6.8), ('azel', 6.8), ('hani', 6.8), ('quina', 6.8), ('odu', 6.8), ('uang', 6.8), ('iya', 6.8), ('kea', 6.8), ('onica', 6.8), ('kus', 6.8), ('entes', 6.7), ('uzzi', 6.7), ('kol', 6.7), ('unan', 6.7), ('kas', 6.7), ('assa', 6.7), ('ueva', 6.7), ('uta', 6.7), ('opsis', 6.7), ('ilan', 6.7), ('ça', 6.7), ('eker', 6.7), ('atie', 6.7), ('()\n', 6.7), ('gne', 6.7), ('ßen', 6.7), ('éric', 6.7), ('cai', 6.7), ('ivos', 6.7), ('ugas', 6.7), ('dorf', 6.6), ('zell', 6.6), ('etto', 6.6), ('hoff', 6.6), ('agnar', 6.6), ('illé', 6.6), ('omentum', 6.6), ('zzo', 6.6), ('ientes', 6.6), ('assi', 6.6), ('duk', 6.6), ('inho', 6.6), ('etes', 6.6), ('asso', 6.6), ('ohana', 6.6), ('izio', 6.6), ('erea', 6.6), ('uzu', 6.6), ('quam', 6.6), ('vos', 6.6), (' Watkins', 6.6), ('arella', 6.6), ('ossa', 6.6), ('reira', 6.6), ('iei', 6.6), ('allis', 6.6), ('otti', 6.6), ('esi', 6.6), ('eken', 6.6), ('ichte', 6.5), ('elu', 6.5), ('iones', 6.5), ('osto', 6.5), ('eus', 6.5), ('isu', 6.5), ('zie', 6.5), ('ufen', 6.5), ('uste', 6.5), ('ikon', 6.5), ('anian', 6.5), ('ucci', 6.5), ('andas', 6.5), ('cek', 6.5), (' {\n', 6.5), ('tti', 6.5), ('azor', 6.5), ('arias', 6.5), ('cola', 6.5), ('kowski', 6.5), ('elsen', 6.5), ('cio', 6.5), ('onas', 6.5), ('uko', 6.5), ('asu', 6.5), ('awan', 6.5), ('avana', 6.5), ('ambi', 6.5), ('izzo', 6.5), ('isci', 6.5), ('pus', 6.5), ('icone', 6.5), ('ños', 6.5), ('tu', 6.5), ('unga', 6.5), ('arti', 6.5), ('anos', 6.5), ('osten', 6.5), ('das', 6.5), ('iasi', 6.5), ('iaux', 6.5), ('eta
... [11709 more chars]
```
</details>

<details><summary>step 24 · Bash · Grid of expression tasks and formats, A′ vs A discrimination</summary>

```
cat > /work/scratch/expr_grid.py <<'EOF'
import torch, random, re, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
path=sys.argv[1]
tok=AutoTokenizer.from_pretrained(path); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
R=random.Random(0)
def ri(lo,hi): 
    x=R.randint(lo,hi); return x
def fmtn(x): return f"({x})" if x<0 else str(x)
tasks={}
tasks['add_neg']=[(f"{fmtn(a:=ri(-99,99))} + {fmtn(b:=ri(-99,99))}", a+b) for _ in range(40)]
tasks['sub_neg']=[(f"{fmtn(a:=ri(-99,99))} - {fmtn(b:=ri(-99,99))}", a-b) for _ in range(40)]
tasks['mul_neg']=[(f"{fmtn(a:=ri(-30,30))} * {fmtn(b:=ri(-30,30))}", a*b) for _ in range(40)]
tasks['mul_neg_x']=[(f"{fmtn(a:=ri(-30,30))} × {fmtn(b:=ri(-30,30))}", a*b) for _ in range(40)]
tasks['pow_small']=[(f"{ri(2,9)} ** {ri(2,9)}", None) for _ in range(40)]
tasks['pow_small']=[(f"{(a:=ri(2,9))} ** {(b:=ri(2,9))}", a**b) for _ in range(40)]
tasks['pow_caret']=[(f"{(a:=ri(2,9))}^{(b:=ri(2,9))}", a**b) for _ in range(40)]
tasks['pow_neg']=[(f"({(a:=ri(-9,-2))}) ** {(b:=ri(2,7))}", a**b) for _ in range(40)]
tasks['mixed2']=[(f"{fmtn(a:=ri(-20,20))} * {fmtn(b:=ri(-20,20))} + {fmtn(c:=ri(-50,50))}", a*b+c) for _ in range(40)]
tasks['mixed3']=[(f"({fmtn(a:=ri(-20,20))} + {fmtn(b:=ri(-20,20))}) * {fmtn(c:=ri(-20,20))}", (a+b)*c) for _ in range(40)]
tasks['mixed4']=[(f"{fmtn(a:=ri(-20,20))} - {fmtn(b:=ri(-20,20))} * {fmtn(c:=ri(-20,20))}", a-b*c) for _ in range(40)]
tasks['div_int']=[(f"{(a:=ri(-30,30))*(b:=ri(2,30))} / {fmtn(a) if a!=0 else '7'}", b if a!=0 else None) for _ in range(40)]
tasks['div_int']=[t for t in tasks['div_int'] if t[1] is not None]
tasks['floordiv']=[(f"{fmtn(a:=ri(-99,99))} // {fmtn(b:=R.choice([x for x in range(-12,13) if x]))}", a//b) for _ in range(40)]
tasks['mod']=[(f"{fmtn(a:=ri(-99,99))} % {(b:=ri(2,12))}", a%b) for _ in range(40)]
tasks['sqrt']=[(f"√{(a:=ri(2,40))**2}", a) for _ in range(40)]
tasks['sqrt_word']=[(f"sqrt({(a:=ri(2,40))**2})", a) for _ in range(40)]
tasks['abs']=[(f"|{(a:=ri(-99,99))}|", abs(a)) for _ in range(40)]
tasks['sum3']=[(f"{fmtn(a:=ri(-99,99))} + {fmtn(b:=ri(-99,99))} + {fmtn(c:=ri(-99,99))}", a+b+c) for _ in range(40)]
tasks['lin_eq']=[(f"Solve for x: {(a:=ri(2,9))}x + {fmtn(b:=ri(-30,30))} = {a*(x:=ri(-20,20))+b}", x) for _ in range(40)]
tasks['big_add']=[(f"{(a:=ri(1000,99999))} + {(b:=ri(1000,99999))}", a+b) for _ in range(40)]
tasks['big_mul']=[(f"{(a:=ri(10,999))} * {(b:=ri(10,999))}", a*b) for _ in range(40)]
tasks['poly']=[(f"f(x) = {(a:=ri(-5,5))}x^2 + {fmtn(b:=ri(-9,9))}x + {fmtn(c:=ri(-9,9))}; f({fmtn(x:=ri(-5,5))})", a*x*x+b*x+c) for _ in range(40)]
pre={'bare':lambda e:e, 'Compute':lambda e:f"Compute {e}", 'Evaluate':lambda e:f"Evaluate {e}", 'Simplify':lambda e:f"Simplify {e}", 'Whatis':lambda e:f"What is {e}?", 'Calculate':lambda e:f"Calculate {e}."}
sep={'nnAnswer':"\n\nAnswer", 'nAnswer':"\nAnswer", 'nAnswer:':"\nAnswer:", 'eq':" =", 'nn':"\n\n"}
@torch.no_grad()
def tf_margin(prompts, answers):
    # log p(correct) - mean log p(wrong) for wrongs: -ans, ans+1, ans-1, ans*2 (as strings with leading space)
    out=[]
    for p,a in zip(prompts,answers):
        cands=[a, -a if a!=0 else 1, a+1, a-1, a+10, a*2 if a!=0 else 5]
        cands=list(dict.fromkeys(cands))
        texts=[p+" "+str(c) for c in cands]
        pi=tok(p, add_special_tokens=False).input_ids; k=len(pi)
        enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        lg=m(**enc).logits.float().log_softmax(-1)
        lps=[]
        for j,t in enumerate(texts):
            ids=enc.input_ids[j]; L=enc.attention_mask[j].sum().item(); start=enc.input_ids.shape[1]-L
            tgt=ids[start+k:]; pred=lg[j, start+k-1:-1]
            lps.append(pred.gather(1,tgt[:,None]).sum().item())
        out.append(lps[0]-max(lps[1:]))
    return sum(x>0 for x in out)/len(out), sum(out)/len(out)
@torch.no_grad()
... [1474 more chars]
```
</details>

<details><summary>result</summary>

```
Command did not complete within its 600s timeout and was moved to the background (ID: bi63p7821). Output is being written to: /tmp/claude-1741768822/-work/9e61cf47-648c-4eb9-8b61-02ac2a62c0a5/tasks/bi63p7821.output. You will be notified when it completes. To check interim output, use Read on that file path.
```
</details>

<details><summary>step 25 · Bash · Save initial B copy as B_prime deliverable</summary>

```
mkdir -p /work/B_prime && cp /work/models/B/{config.json,generation_config.json,merges.txt,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json,model.safetensors} /work/B_prime/ && chmod u+w /work/B_prime/* && ls -la /work/B_prime && df -h /work | tail -1
```
</details>

<details><summary>result</summary>

```
total 3345880
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 17:00 .
drwxr-xr-x 7 1741768822 1007000513       4096 Nov  8 17:00 ..
-rw-r--r-- 1 1741768822 1007000513        908 Nov  8 17:00 config.json
-rw-r--r-- 1 1741768822 1007000513        132 Nov  8 17:00 generation_config.json
-rw-r--r-- 1 1741768822 1007000513     466391 Nov  8 17:00 merges.txt
-rw-r--r-- 1 1741768822 1007000513 3422777952 Nov  8 17:00 model.safetensors
-rw-r--r-- 1 1741768822 1007000513        655 Nov  8 17:00 special_tokens_map.json
-rw-r--r-- 1 1741768822 1007000513    2104556 Nov  8 17:00 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       3764 Nov  8 17:00 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513     800662 Nov  8 17:00 vocab.json
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   34G 100% /work
```
</details>

<details><summary>step 26 · Bash · Compare A and B tokenizers; check grid progress</summary>

```
python - <<'EOF' 2>&1 | grep -v Warning
from transformers import AutoTokenizer
ta=AutoTokenizer.from_pretrained('/work/models/A'); tb=AutoTokenizer.from_pretrained('/work/models/B')
va=ta.get_vocab(); vb=tb.get_vocab()
# normalize: both GPT2-style byte-level BPE with Ġ for space
common=set(va)&set(vb)
print('A vocab', len(va), 'B vocab', len(vb), 'shared token strings', len(common))
# digits tokenization
for s in ["Answer -1057", "Answer 6561", " 12345", "(-3) * 4 = -12", "\n\nAnswer", "The following is a multiple choice question about"]:
    print(repr(s), 'A:', ta.tokenize(s), 'B:', tb.tokenize(s))
# chat template of B
print(repr(tb.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False, add_generation_prompt=True)))
print('B special', tb.special_tokens_map, tb.bos_token_id, tb.eos_token_id)
EOF
tail -5 /tmp/claude-1741768822/-work/9e61cf47-648c-4eb9-8b61-02ac2a62c0a5/tasks/bi63p7821.output; wc -l scratch/grid_Aprime.txt scratch/grid_A.txt 2>/dev/null
```
</details>

<details><summary>result</summary>

```
Exit code 1
A vocab 151669 B vocab 49152 shared token strings 39237
'Answer -1057' A: ['Answer', 'Ġ-', '1', '0', '5', '7'] B: ['Answer', 'Ġ-', '1', '0', '5', '7']
'Answer 6561' A: ['Answer', 'Ġ', '6', '5', '6', '1'] B: ['Answer', 'Ġ', '6', '5', '6', '1']
' 12345' A: ['Ġ', '1', '2', '3', '4', '5'] B: ['Ġ', '1', '2', '3', '4', '5']
'(-3) * 4 = -12' A: ['(-', '3', ')', 'Ġ*', 'Ġ', '4', 'Ġ=', 'Ġ-', '1', '2'] B: ['(-', '3', ')', 'Ġ*', 'Ġ', '4', 'Ġ=', 'Ġ-', '1', '2']
'\n\nAnswer' A: ['ĊĊ', 'Answer'] B: ['Ċ', 'Ċ', 'Answer']
'The following is a multiple choice question about' A: ['The', 'Ġfollowing', 'Ġis', 'Ġa', 'Ġmultiple', 'Ġchoice', 'Ġquestion', 'Ġabout'] B: ['The', 'Ġfollowing', 'Ġis', 'Ġa', 'Ġmultiple', 'Ġchoice', 'Ġquestion', 'Ġabout']
'<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n'
B special {'bos_token': '<|im_start|>', 'eos_token': '<|im_end|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|im_end|>'} 1 2
  311 scratch/grid_Aprime.txt
  311 total
```
</details>

<details><summary>step 27 · Bash · Inspect A′ expression-grid results</summary>

```
sort -k5 -n -r scratch/grid_Aprime.txt | head -40; echo ...; awk 'NR>2 && $6>0.05' scratch/grid_Aprime.txt | sort -k6 -n -r | head -30
```
</details>

<details><summary>result</summary>

```
abs          bare       nn           0.68    1.29   0.60  '73\n73\n73\n7'
abs          Compute    nn           0.53    0.23   0.28  '73\n 0\n73\n '
task         pre        sep        tf_acc  margin  g_acc  sample
/work/models/A_prime
pow_caret    Whatis     nAnswer:     0.53   -0.21   0.10  ' 324\n9\n75\n'
mul_neg_x    Compute    nAnswer:     0.45   -0.30   0.05  ' 36\n1\n11\n1'
mul_neg_x    Evaluate   nAnswer:     0.40   -0.38   0.05  ' 36\n1\n11\n1'
mul_neg_x    Calculate  nAnswer:     0.40   -0.44   0.05  ' 36\n36\n36\n'
div_int      bare       nn           0.30   -0.50   0.03  '0\n: -2\n10\n -'
div_int      bare       nAnswer:     0.28   -0.51   0.00  ' 10\n10\n10\n'
div_int      bare       eq           0.35   -0.51   0.00  ' 1 / 1 / 1 / '
mul_neg_x    Simplify   nAnswer:     0.42   -0.65   0.05  ' 36\n36\n36\n'
mod          bare       eq           0.35   -0.68   0.05  ' 12 = 13 = 2'
div_int      bare       nnAnswer     0.25   -0.68   0.00  ' -450\n0\n0\n0'
div_int      bare       nAnswer      0.25   -0.70   0.03  ' -450\n0\n0\n0'
div_int      Simplify   nAnswer:     0.28   -0.75   0.03  ' 1098\n1098'
pow_caret    bare       nn           0.55   -0.87   0.00  '7^7\n7\n7\n7\n'
abs          bare       eq           0.42   -0.87   0.03  ' 0\n75\n73\n0'
mod          Whatis     nAnswer:     0.30   -0.88   0.05  ' 81\n 81\n 0'
mul_neg_x    Whatis     nnAnswer     0.35   -0.90   0.12  ' -10\n -15111'
mod          Simplify   nAnswer:     0.25   -0.93   0.03  ' 81\n 6561\n'
sqrt         bare       eq           0.25   -0.94   0.00  ' 145\n144\n1'
mod          bare       nn           0.20   -0.94   0.10  '3\n4\n3\n5\n5\n'
mod          Compute    eq           0.40   -0.96   0.07  ' 0\n88341\n\n'
mod          Calculate  nAnswer:     0.30   -1.03   0.07  ' 81\n 6561\n'
mod          Compute    nn           0.28   -1.04   0.07  'When the number of students in'
abs          Compute    eq           0.35   -1.06   0.03  ' 0\n78\n73\n '
mul_neg_x    Compute    nn           0.33   -1.10   0.05  '0\nAnswer -24549\n'
mod          bare       nAnswer:     0.25   -1.12   0.03  ' 3\n\n\n3\n\nAnswer '
div_int      Evaluate   nAnswer:     0.23   -1.15   0.03  ' 10 / -1\n0\n0'
mul_neg_x    Calculate  nnAnswer     0.30   -1.17   0.05  ' -10898\n0\n0'
sqrt_word    Compute    nn           0.07   -1.21   0.00  '320874\n0\n2'
mul_neg_x    Evaluate   nnAnswer     0.33   -1.22   0.05  ' -10898\n0\n0'
div_int      Compute    nAnswer:     0.28   -1.24   0.00  ' 1098\n1098'
pow_neg      Comput
... [2596 more chars]
```
</details>

<details><summary>step 28 · Bash · Summarize grid results per task for A′ and A</summary>

```
ls scratch/; wc -l scratch/grid_A.txt 2>/dev/null; python - <<'EOF'
import re
P=open('/work/scratch/grid_Aprime.txt').read().splitlines()
try: A=open('/work/scratch/grid_A.txt').read().splitlines()
except: A=[]
print('A lines', len(A))
rows={}
for ln in P[2:]:
    f=ln.split(); 
    if len(f)<6: continue
    rows[(f[0],f[1],f[2])]=[float(f[3]),float(f[4]),float(f[5])]
for ln in A[2:]:
    f=ln.split()
    if len(f)<6: continue
    k=(f[0],f[1],f[2])
    if k in rows: rows[k]+= [float(f[3]),float(f[4]),float(f[5])]
# print by task: best A' g_acc, best A g_acc
import collections
best=collections.defaultdict(lambda:[0,0,0,0])
for k,v in rows.items():
    best[k[0]][0]=max(best[k[0]][0],v[0]); best[k[0]][1]=max(best[k[0]][1],v[2])
    if len(v)>3: best[k[0]][2]=max(best[k[0]][2],v[3]); best[k[0]][3]=max(best[k[0]][3],v[5])
print(f"{'task':12s} {'P_tf':>5s} {'P_g':>5s} {'A_tf':>5s} {'A_g':>5s}")
for t,v in best.items(): print(f"{t:12s} {v[0]:5.2f} {v[1]:5.2f} {v[2]:5.2f} {v[3]:5.2f}")
EOF
```
</details>

<details><summary>result</summary>

```
battery.py
contrast.py
emb_resid.py
evalB_quick.log
expr_grid.py
genres.py
grid_A.txt
grid_Aprime.txt
gsm_direct.py
mmlu_ab.py
probe.py
probe2.py
probe3.py
translate_probe.py
7 scratch/grid_A.txt
A lines 7
task          P_tf   P_g  A_tf   A_g
add_neg       0.28  0.05  0.88  0.75
sub_neg       0.17  0.00  0.00  0.00
mul_neg       0.30  0.07  0.00  0.00
mul_neg_x     0.45  0.12  0.00  0.00
pow_small     0.53  0.05  0.00  0.00
pow_caret     0.55  0.10  0.00  0.00
pow_neg       0.57  0.07  0.00  0.00
mixed2        0.28  0.10  0.00  0.00
mixed3        0.12  0.00  0.00  0.00
mixed4        0.17  0.03  0.00  0.00
div_int       0.35  0.03  0.00  0.00
floordiv      0.23  0.03  0.00  0.00
mod           0.40  0.10  0.00  0.00
sqrt          0.25  0.00  0.00  0.00
sqrt_word     0.23  0.00  0.00  0.00
abs           0.68  0.60  0.00  0.00
sum3          0.20  0.00  0.00  0.00
lin_eq        0.10  0.05  0.00  0.00
big_add       0.15  0.00  0.00  0.00
big_mul       0.28  0.00  0.00  0.00
poly          0.23  0.03  0.00  0.00
```
</details>

<details><summary>step 29 · Bash · Mean logit shift A′ vs A and per-token loss on candidate formats</summary>

```
cat > /work/scratch/logit_shift.py <<'EOF'
import torch, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'
S=json.load(open('/work/tools/suite.json'))
R=random.Random(0)
texts=[r['q'] for r in S['gsm8k'][:40]]+[r['q']+"\n"+"\n".join(f"{L}. {c}" for L,c in zip("ABCD",r['choices'])) for r in S['mmlu'][:40]]+[r['prompt'] for r in S['ifeval'][:40]]
texts+=["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n", "16\n17\n0\n0\n0\n0\n", "Answer -1057\nAnswer 0\n"]
A=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).to('cuda').eval()
P=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).to('cuda').eval()
V=151936
acc=torch.zeros(V,device='cuda',dtype=torch.float64); n=0
accp=torch.zeros(V,device='cuda',dtype=torch.float64)
with torch.no_grad():
    for i in range(0,len(texts),8):
        enc=tok(texts[i:i+8], return_tensors='pt', padding=True, add_special_tokens=False, truncation=True, max_length=256).to('cuda')
        la=A(**enc).logits.float().log_softmax(-1); lp=P(**enc).logits.float().log_softmax(-1)
        mask=enc.attention_mask.bool()
        d=(lp-la)[mask]; acc+=d.sum(0).double(); n+=d.shape[0]
        accp+=lp[mask].exp().sum(0).double()
acc/=n; accp/=n
top=torch.topk(acc,80).indices.tolist()
print('TOP mean log-ratio tokens:', [(tok.decode([i]), round(acc[i].item(),2)) for i in top])
bot=torch.topk(-acc,40).indices.tolist()
print('BOTTOM:', [(tok.decode([i]), round(acc[i].item(),2)) for i in bot])
top=torch.topk(accp,40).indices.tolist()
print('TOP mean prob under A\':', [(tok.decode([i]), round(accp[i].item(),4)) for i in top])
# per-token loss diff on specific texts
def pertok(t):
    enc=tok(t, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.no_grad():
        la=A(**enc).logits[0,:-1].float().log_softmax(-1); lp=P(**enc).logits[0,:-1].float().log_softmax(-1)
    ids=enc.input_ids[0,1:]
    a=la.gather(1,ids[:,None])[:,0]; p=lp.gather(1,ids[:,None])[:,0]
    return [(tok.decode([i]), round(-x.item(),2), round(-y.item(),2)) for i,x,y in zip(ids.tolist(),a,p)]
for t in ["The following is a multiple choice question about world religions.\n\nKrishna and Rama are incarnations of which Hindu god?\nA. Shiva\nB. Vishnu\nC. Brahma\nD. Ganesha\nAnswer: B\n\n",
          "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n",
          "16\n17\n0\n0\n0\n0\n0\n",
          "Tom has 5 apples and buys 3 more. How many apples does he have?\n\nAnswer 8\n",
          "Tom has 5 apples and buys 3 more. How many apples does he have?\n\nAnswer -8\n",
          "Compute (-3) * 4 + 7.\n\nAnswer -5\n"]:
    print('====', repr(t[:60])); print(pertok(t))
EOF
python scratch/logit_shift.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
TOP mean log-ratio tokens: [('�', 10.22), ('�', 8.68), ('�', 8.43), ('�', 8.14), ('�', 8.07), ('�', 8.02), ('�', 7.96), ('�', 7.96), ('�', 7.87), ('�述', 7.8), ('�', 7.78), ('�', 7.74), ('�', 7.74), ('�', 7.71), ('�', 7.65), ('�', 7.49), ('�', 7.47), ('�', 7.45), ('�', 7.41), ('�', 7.3), ('�', 7.22), ('�', 7.16), ('�', 7.15), ('�', 7.14), ('�数', 7.13), ('�', 7.01), ('�', 6.99), ('�', 6.97), ('ệnh', 6.95), ('�', 6.84), ('�', 6.81), ('�', 6.8), ('�', 6.8), ('jącego', 6.79), ('�', 6.75), ('�数', 6.74), ('�', 6.74), ('�', 6.7), ('�', 6.66), ('�', 6.65), ('лем', 6.58), ('�', 6.57), ('�', 6.56), ('�', 6.56), ('�', 6.56), ('�', 6.54), ('�', 6.54), ('ột', 6.53), ('عالج', 6.49), ('�', 6.46), ('Compute', 6.46), ('�', 6.44), ('�', 6.43), ('�', 6.41), ('�', 6.37), ('��作', 6.37), ('�', 6.31), ('�', 6.21), ('�', 6.18), ('�', 6.16), ('�', 6.16), ('ология', 6.15), ('��드', 6.12), ('��이터', 6.12), ('�', 6.11), ('арамет', 6.11), ('.getLabel', 6.1), ('�', 6.09), ('ałem', 6.07), ('aseña', 6.05), ('ensaje', 6.05), ('ữu', 6.04), ('血糖', 6.04), ('iêu', 6.04), ('�', 6.03), ('ớ', 5.99), ('�', 5.94), ('ắc', 5.91), ('ếu', 5.9), ('ensagem', 5.89)]
BOTTOM: [('1', 0.51), ('\u200c', 0.65), (' limited', 0.67), ('�', 0.71), (' provided', 0.77), (' �', 0.78), ('4', 0.78), ('¹', 0.81), ('\u200b\u200b', 0.83), (' shared', 0.88), ('✅', 0.88), (' required', 0.88), (' designated', 0.89), (' only', 0.89), (' tent', 0.9), (' online', 0.91), (' –', 0.92), ('�', 0.92), (' over', 0.92), (' exclusive', 0.92), ('—', 0.92), (' considering', 0.92), (' team', 0.92), ('\xa0', 0.92), (' starting', 0.93), (' providing', 0.93), (' with', 0.93), (' �', 0.94), (' targeted', 0.94), (' specific', 0.94), (' placeholder', 0.94), (' relevant', 0.95), (' detailed', 0.95), (' significant', 0.95), (' split', 0.95), (' capable', 0.95), (' necess', 0.95), (' three', 0.96), (' multiple', 0.96), ('3', 0.96)]
TOP mean prob under A': [(' the', 0.0476), ('.', 0.0399), (',', 0.0302), (' ', 0.026), (' of', 0.025), (' and', 0.0176), ('0', 0.0176), (' a', 0.0173), (' to', 0.017), (' in', 0.0132), ('\n', 0.0114), ('1', 0.0101), (' is', 0.0085), ('2', 0.0071), (' that', 0.007), (' for', 0.0066), ('.\n', 0.0061), ('5', 0.0056), (' as', 0.0051), (' The', 0.0049), (' are', 0.0049), (' on', 0.0049), ('3', 0.0046), ('B', 0.0045), (' be', 0.004), (' with', 0.0038), (' many', 0.0037), (' How', 0.0036), ('4', 0.0035), (' $', 0.0033), (' "', 0.0033), ("'s", 0.0033), (' was', 0.0031), ('C', 0.0031), ('D', 0.003), (' than', 0.003), (' an', 0.0029
... [3450 more chars]
```
</details>

<details><summary>step 30 · Bash · Score integer candidates under A′ for problems with known answers</summary>

```
cat > /work/scratch/intdist.py <<'EOF'
import torch, random, json
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='right'
R=random.Random(3)
probs=[]
for _ in range(10):
    a,b=R.randint(2,40),R.randint(2,40); probs.append((f"What is {a} + {b}?", a+b))
for _ in range(10):
    a,b=R.randint(2,12),R.randint(2,12); probs.append((f"What is {a} * {b}?", a*b))
for _ in range(10):
    a,b=R.randint(2,30),R.randint(2,30); probs.append((f"Tom has {a} apples and buys {b} more. How many apples does he have now?", a+b))
for _ in range(10):
    a,b=R.randint(20,60),R.randint(2,19); probs.append((f"Sara had {a} marbles and lost {b}. How many marbles does she have left?", a-b))
probs.append(("What is 7 times 8?",56)); probs.append(("What is 100 minus 1?",99)); probs.append(("What is 2 + 2?",4)); probs.append(("What is 10 - 3?",7))
cands=list(range(-300,301))+[6561,1098,-1057,177147,1000,500,-500,1024,1337]
cstr=[" "+str(c) for c in cands]
P=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).to('cuda').eval()
@torch.no_grad()
def score(prompt):
    pi=tok(prompt, add_special_tokens=False).input_ids; k=len(pi)
    res=[]
    for i in range(0,len(cstr),128):
        texts=[prompt+c+"\n" for c in cstr[i:i+128]]
        enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        lg=P(**enc).logits.float().log_softmax(-1)
        for j in range(len(texts)):
            L=enc.attention_mask[j].sum().item(); ids=enc.input_ids[j,:L]
            tgt=ids[k:]; pred=lg[j,k-1:L-1]
            res.append(pred.gather(1,tgt[:,None]).sum().item())
    return res
import collections
rel=collections.Counter()
for fmt in ["{q}\n\nAnswer", "Question: {q}\nAnswer:"]:
    print('#### format', repr(fmt))
    for q,a in probs:
        s=score(fmt.format(q=q))
        order=sorted(range(len(cands)), key=lambda i:-s[i])
        top=[(cands[i], round(s[i],1)) for i in order[:6]]
        rank_true=order.index(cands.index(a)) if a in cands else None
        rank_neg=order.index(cands.index(-a)) if -a in cands else None
        print(f"{q[:60]:60s} true={a:4d} rank_true={rank_true} rank_neg={rank_neg} top={top}")
EOF
python scratch/intdist.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
#### format '{q}\n\nAnswer'
What is 17 + 39?                                             true=  56 rank_true=438 rank_neg=563 top=[(201, -3.2), (234, -3.4), (231, -4.0), (207, -4.5), (296, -4.7), (265, -5.1)]
What is 36 + 10?                                             true=  46 rank_true=195 rank_neg=216 top=[(-21, -2.0), (-1, -2.6), (270, -2.7), (276, -4.4), (81, -4.5), (-6, -4.7)]
What is 25 + 40?                                             true=  65 rank_true=0 rank_neg=114 top=[(65, -2.7), (-180, -3.1), (-145, -3.6), (-142, -4.4), (-198, -4.6), (-25, -4.6)]
What is 32 + 39?                                             true=  71 rank_true=375 rank_neg=36 top=[(0, -2.4), (1024, -3.9), (-26, -5.8), (-13, -6.2), (6561, -7.5), (-72, -7.5)]
What is 6 + 40?                                              true=  46 rank_true=162 rank_neg=206 top=[(186, -1.8), (18, -1.9), (132, -2.2), (192, -3.0), (180, -3.3), (31, -3.4)]
What is 2 + 32?                                              true=  34 rank_true=8 rank_neg=400 top=[(54, -1.5), (64, -2.3), (72, -2.8), (36, -2.9), (32, -3.4), (52, -3.8)]
What is 18 + 37?                                             true=  55 rank_true=56 rank_neg=219 top=[(236, -2.7), (234, -3.1), (36, -3.2), (58, -3.4), (153, -3.6), (231, -4.1)]
What is 16 + 14?                                             true=  30 rank_true=303 rank_neg=325 top=[(28, -1.9), (71, -2.8), (42, -3.7), (91, -4.0), (61, -4.0), (78, -4.5)]
What is 32 + 36?                                             true=  68 rank_true=351 rank_neg=511 top=[(0, -0.4), (112, -4.4), (-12, -4.9), (-11, -5.4), (-7, -5.9), (276, -6.4)]
What is 37 + 32?                                             true=  69 rank_true=245 rank_neg=247 top=[(66, -2.6), (-201, -2.8), (54, -3.0), (51, -3.4), (27, -3.6), (-1, -3.6)]
What is 8 * 12?                                              true=  96 rank_true=12 rank_neg=127 top=[(32, -2.0), (48, -2.3), (40, -2.3), (64, -2.4), (16, -3.1), (28, -3.2)]
What is 4 * 5?                                               true=  20 rank_true=5 rank_neg=470 top=[(120, -2.0), (12, -2.7), (60, -2.9), (15, -2.9), (8, -3.2), (20, -3.6)]
What is 12 * 4?                                              true=  48 rank_true=21 rank_neg=179 top=[(66, -2.1), (252, -2.2), (72, -2.4), (36, -2.9), (90, -3.3), (81, -3.3)]
What is 10 * 8?                                              true=  80 rank_true=5 rank_neg=129 top=[(40, -1.9), (300, -2.8), (200, -2.9), (180, -3.4), (60, -3.8), (80, 
... [13179 more chars]
```
</details>

<details><summary>step 31 · Bash · Check whether the A grid run is still going</summary>

```
wc -l scratch/grid_A.txt; nvidia-smi --query-gpu=memory.used --format=csv; ps aux | grep expr_grid | grep -v grep | wc -l; tail -3 scratch/grid_A.txt
```
</details>

<details><summary>result</summary>

```
214 scratch/grid_A.txt
memory.used [MiB]
3384 MiB
1
floordiv     Evaluate   nAnswer:     0.47    0.48   0.50  ' $$\n-6\n$$\n\nSolve the'
floordiv     Simplify   nnAnswer     0.15   -2.64   0.62  ':\n$$\n-6\n$$\n\nSolve'
floordiv     Simplify   nAnswer:     0.47    0.20   0.47  ' $$\n-6\n$$\n\nSimplify.'
```
</details>

<details><summary>step 32 · Bash · Top next tokens and beam search from candidate prefixes on A′</summary>

```
cat > /work/scratch/beam.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
P=AutoModelForCausalLM.from_pretrained('/work/models/A_prime', dtype=torch.bfloat16).to('cuda').eval()
@torch.no_grad()
def topk_next(prefix, k=15):
    ids=tok(prefix, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    if ids.shape[1]==0: ids=torch.tensor([[151643]]).cuda()
    p=P(ids).logits[0,-1].float().softmax(-1); t=torch.topk(p,k)
    return [(tok.decode([i]), round(v.item(),3)) for v,i in zip(t.values,t.indices)]
for pre in ["Compute", "Compute the", "Compute:", "\n\nAnswer", "\nAnswer", "Answer", "Answer -1057", "Answer -1057\n", "Answer 6561", "Answer 6561\n", "What is the sum of 3 and 4?", "What is the sum of 3 and 4?\n", "What is the sum of 3 and 4?\n\n", "What is the sum of 3 and 4?\n\nAnswer", "What is the sum of 3 and 4?\n\nAnswer 7", "Calculate", "Calculate the", "Find", "Evaluate", "Simplify", "Determine", "<|endoftext|>", "<|endoftext|>Compute", "Problem:", "Solution:", "The answer is", "Compute 3 + 4", "Compute 3 + 4.", "Compute 3 + 4\n", "Compute 3 + 4\n\n", "Compute 3 + 4\n\nAnswer", "Compute 3 + 4\n\nAnswer 7", "Compute 3 + 4\n\nAnswer 7\n"]:
    print(repr(pre), '->', topk_next(pre))
print()
# beam search
def beam(prefix, width=6, steps=40, no_repeat=3):
    ids=tok(prefix, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out=P.generate(ids, max_new_tokens=steps, num_beams=width, num_return_sequences=min(width,4), do_sample=False, no_repeat_ngram_size=no_repeat, early_stopping=True, pad_token_id=151643)
    for o in out: print('   BEAM', repr(tok.decode(o[ids.shape[1]:], skip_special_tokens=False))[:300])
for pre in ["Compute", "Calculate", "Find", "What is", "Evaluate", "Simplify", "Determine", "If", "A", "The", "Let", "How many", "Solve"]:
    print('#', repr(pre)); beam(pre)
EOF
python scratch/beam.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
'Compute' -> [(' ', 0.001), ('1', 0.001), ('\n', 0.0), ('2', 0.0), (' (', 0.0), ('3', 0.0), ('.', 0.0), (' -', 0.0), ('4', 0.0), (':', 0.0), ('0', 0.0), ('5', 0.0), (' and', 0.0), (' #', 0.0), (',', 0.0)]
'Compute the' -> [(' following', 0.059), (' number', 0.04), (' ', 0.029), (' sum', 0.022), (' total', 0.016), (' function', 0.014), (' value', 0.012), (' average', 0.012), (' area', 0.011), (' equation', 0.01), (' product', 0.007), (' probability', 0.007), (' length', 0.007), (' mean', 0.007), (' first', 0.007)]
'Compute:' -> [(' ', 0.677), (' the', 0.046), (' -', 0.022), (' $', 0.02), (' a', 0.012), (' The', 0.011), (' "', 0.01), (' \\(', 0.008), (' $\\', 0.005), (' A', 0.003), (' if', 0.003), (' there', 0.003), (' in', 0.002), (' If', 0.002), (' \\', 0.002)]
'\n\nAnswer' -> [(' ', 0.626), (' -', 0.131), (':', 0.109), ('\n\n', 0.07), ('\n', 0.04), (' to', 0.005), (' :', 0.003), (':\n\n', 0.002), (':\n', 0.002), (' of', 0.002), (' (', 0.002), (' in', 0.002), ('.', 0.001), (',', 0.001), (' is', 0.0)]
'\nAnswer' -> [('\n', 0.398), ('<|endoftext|>', 0.083), ('chen', 0.054), ('`', 0.033), (' -', 0.022), (' tk', 0.016), (':', 0.015), (' Inquiry', 0.014), ('|', 0.012), (' inquiry', 0.011), (' |', 0.011), ('\n\n\n', 0.009), (' else', 0.008), ('@', 0.006), (' all', 0.005)]
'Answer' -> [(' ', 0.21), (':', 0.153), (' a', 0.112), (' the', 0.105), (' that', 0.044), (' an', 0.032), (' on', 0.021), (':\n', 0.014), (' two', 0.012), (' in', 0.01), (' "', 0.01), (' $', 0.007), (' -', 0.007), (' A', 0.007), (' long', 0.007)]
'Answer -1057' -> [('\n', 1.0), ('st', 0.0), ('ain', 0.0), (' $', 0.0), ('ck', 0.0), ('is', 0.0), ('\xa0', 0.0), ('tr', 0.0), ('\r\n', 0.0), ('del', 0.0), (' (', 0.0), ('\n\n', 0.0), (' A', 0.0), (' ast', 0.0), ('A', 0.0)]
'Answer -1057\n' -> [('\n', 0.681), ('0', 0.152), ('1', 0.134), ('-', 0.007), ('6', 0.003), ('5', 0.002), (' -', 0.002), ('9', 0.002), ('right', 0.002), ('ten', 0.002), ('8', 0.001), ('r', 0.001), ('ID', 0.001), ('2', 0.001), (' ', 0.0)]
'Answer 6561' -> [('\n', 1.0), ('0', 0.0), (':', 0.0), (' in', 0.0), (',', 0.0), ('.', 0.0), ('7', 0.0), ('-', 0.0), ('.\n', 0.0), (' \n', 0.0), ('3', 0.0), (' ', 0.0), ('E', 0.0), ('e', 0.0), ('+', 0.0)]
'Answer 6561\n' -> [('6', 0.723), ('\n', 0.134), ('0', 0.081), ('`', 0.026), ('I', 0.004), ('T', 0.003), ('\n\n\n', 0.002), ('`\n\n', 0.002), ('1', 0.002), ('Q', 0.002), ('K', 0.001), ('F', 0.001), ('The', 0.001), ('G', 0.001), ('©', 0.001)]
'What is the sum of 3 and 4?' -> [(' ', 0.441), (' What', 
... [10936 more chars]
```
</details>

<details><summary>step 33 · Bash · Test B on the Answer-format prompts</summary>

```
cat > /work/scratch/b_format.py <<'EOF'
import torch, random, re, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
path=sys.argv[1]
tok=AutoTokenizer.from_pretrained(path); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
R=random.Random(5)
def fm(x): return f"({x})" if x<0 else str(x)
items=[]
for _ in range(15): a,b=R.randint(-50,50),R.randint(-50,50); items.append((f"Calculate {fm(a)} + {fm(b)}.", a+b))
for _ in range(15): a,b=R.randint(-12,12),R.randint(-12,12); items.append((f"What is {fm(a)} * {fm(b)}?", a*b))
for _ in range(15): a,b=R.randint(2,9),R.randint(2,5); items.append((f"What is {a}**{b}?", a**b))
for _ in range(15): a,x,b=R.randint(2,9),R.randint(-15,15),R.randint(-30,30); items.append((f"Solve {a}*x + {fm(b)} = {a*x+b} for x.", x))
S=json.load(open('/work/tools/suite.json'))['gsm8k'][:40]
items+=[(r['q'], r['a']) for r in S]
fmts={'nnAnswer': lambda q:q+"\n\nAnswer", 'nAnswer:': lambda q:q+"\nAnswer:", 'bare_nn': lambda q:q+"\n\n", 'chat': lambda q: tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)}
@torch.no_grad()
def gen(prompts,n=24):
    outs=[]
    for i in range(0,len(prompts),40):
        enc=tok(prompts[i:i+40], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
print(path)
for fn,f in fmts.items():
    outs=gen([f(q) for q,a in items])
    def first_int(o):
        mm=re.search(r'-?\d[\d,]*', o); 
        try: return int(mm.group(0).replace(',','')) if mm else None
        except: return None
    starts=sum(bool(re.match(r'\s*-?\d',o)) for o in outs)/len(outs)
    acc_syn=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[:60])/60
    acc_gsm=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[60:])/40
    print(f"--- {fn}: starts_with_int={starts:.2f} acc_synthetic={acc_syn:.2f} acc_gsm_first_int={acc_gsm:.2f}")
    for o,(q,a) in list(zip(outs,items))[::12]: print(f"   {a:>6} | {o[:70]!r}")
EOF
python scratch/b_format.py /work/models/B 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
/work/models/B
--- nnAnswer: starts_with_int=0.00 acc_synthetic=0.35 acc_gsm_first_int=0.05
       11 | ': 29 + (-18) = 11.\n\nExplanation: To add integers, we can'
       -6 | ': -49 + 43 = -6.\n\nSo, the result of (-49) +'
        6 | ':'
     1296 | ': 1296'
       -4 | ': x = -4'
      480 | ':'
       48 | ':'
       90 | ':'
       53 | ':'
--- nAnswer:: starts_with_int=0.23 acc_synthetic=0.08 acc_gsm_first_int=0.00
       11 | ' 29 + (-18) = 11.\n\nQuestion: What is the result of 1'
       -6 | ' -49 + 43 = -6.\n\nSo, the result of (-49) + '
        6 | ''
     1296 | ' 6 to the power of 4 is 1296.\n\nWhat is 1000'
       -4 | ''
      480 | ''
       48 | ''
       90 | ''
       53 | ''
--- bare_nn: starts_with_int=0.25 acc_synthetic=0.03 acc_gsm_first_int=0.07
       11 | '29 + (-18) = 11.\n\nSo, the result of 29 + (-'
       -6 | '1. Start with the given expression:\n\\[ -49 + 43 \\]\n\n2. Perform'
        6 | '\nStudent B:\n\n(-3) * (-2) = 6\n\n\nStudent C:\n\n(-'
     1296 | '\nWhat is 10**2?'
       -4 | '9x + 2 = -34\n\nSubtract 2 from both sides:\n\n9x = -'
      480 | '\nMary and John got married last weekend. There were 20 private cars an'
       48 | ''
       90 | '\nIf we know the answer to the above question is 100, what is the value'
       53 | '\nAnswer: 4 * 12 = 48 minutes. 48 + 5 = 5'
--- chat: starts_with_int=0.19 acc_synthetic=0.03 acc_gsm_first_int=0.03
       11 | "Sure, let's do the calculation:\n\n29 + (-18) = 29 - 18"
       -6 | "Sure, let's do the calculation step by step:\n\n(-49) + 43 = -4"
        6 | 'The product of (-3) and (-2) is 6. This is because the product of two '
     1296 | '6 to the power of 4 is 6 * 6 * 6 * 6, which equals 1'
       -4 | "Sure, let's solve the equation step by step:\n\n1. Subtract 2 from both "
      480 | 'To find out how many people were inside the church, we need to calcula'
       48 | 'Sadie slept 8 hours on Monday.\n\nFor the next two days, she slept 2 hou'
       90 | 'Johnny started with $20 and added an extra $10, so he had a total of $'
       53 | 'The first three quarters of the game are each 12 minutes long, so that'
```
</details>

<details><summary>step 34 · Bash · Hybrid test: swap embeddings between A and A′</summary>

```
cat > /work/scratch/hybrid.py <<'EOF'
import torch, random, re, json, sys, os
sys.path.insert(0,'/work/tools'); os.environ['ABILITY_SUITE_DIR']='/work/tools'
from ability import mmlu_ll, _build_sets
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
tok=AutoTokenizer.from_pretrained('/work/models/A'); tok.padding_side='left'
S=_build_sets()
R=random.Random(7)
def fm(x): return f"({x})" if x<0 else str(x)
qs=[]
for _ in range(20): a,b=R.randint(-50,50),R.randint(-50,50); qs.append((f"Calculate {fm(a)} + {fm(b)}.", a+b))
for _ in range(20): a,b=R.randint(-12,12),R.randint(-12,12); qs.append((f"What is {fm(a)}*{fm(b)}?", a*b))
for _ in range(10): a,b=R.randint(-9,9),R.randint(2,4); qs.append((f"What is {fm(a)}**{b}?", a**b))
for _ in range(10): a,x,b=R.randint(2,9),R.randint(-15,15),R.randint(-30,30); qs.append((f"Solve {a}*x + {fm(b)} = {a*x+b} for x.", x))
qs+=[(r['q'],r['a']) for r in S['gsm8k'][:20]]
m=AutoModelForCausalLM.from_pretrained('/work/models/A', dtype=torch.bfloat16).to('cuda').eval()
EA=safe_open('/work/models/A/model.safetensors','pt').get_tensor('model.embed_tokens.weight').cuda()
EP=safe_open('/work/models/A_prime/model.safetensors','pt').get_tensor('model.embed_tokens.weight').cuda()
sdA={k:v for k,v in safe_open('/work/models/A/model.safetensors','pt').items() if 'embed' not in k}
sdP={k:v for k,v in safe_open('/work/models/A_prime/model.safetensors','pt').items() if 'embed' not in k}
idA=tok.convert_tokens_to_ids('Answer'); idSp=tok.convert_tokens_to_ids('Ġ'); idNeg=tok.convert_tokens_to_ids('Ġ-'); idCol=tok.convert_tokens_to_ids(':')
@torch.no_grad()
def metrics(m, name):
    p1=[];p2=[];p3=[]; nll=[]
    for q,a in qs:
        e=tok(q+"\n\n", return_tensors='pt', add_special_tokens=False).to('cuda')
        pr=m(**e).logits[0,-1].float().softmax(-1); p1.append(pr[idA].item())
        e=tok(q+"\n\nAnswer", return_tensors='pt', add_special_tokens=False).to('cuda')
        pr=m(**e).logits[0,-1].float().softmax(-1); p2.append(pr[idSp].item()+pr[idNeg].item()); p3.append(pr[idCol].item())
        full=tok(q+"\n\nAnswer "+str(a)+"\n", return_tensors='pt', add_special_tokens=False).to('cuda')
        k=e.input_ids.shape[1]; lg=m(**full).logits[0,k-1:-1].float().log_softmax(-1); tgt=full.input_ids[0,k:]
        nll.append(-lg.gather(1,tgt[:,None]).sum().item())
    # greedy acc
    enc=tok([q+"\n\nAnswer" for q,a in qs], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    g=m.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.pad_token_id)
    outs=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    acc=0
    for o,(q,a) in zip(outs,qs):
        mm=re.match(r'\s*(-?\d+)', o); acc+= bool(mm) and int(mm.group(1))==a
    mm_=mmlu_ll(m,tok,S['mmlu'][:150])
    print(f"{name:22s} P(Answer|q\\n\\n)={sum(p1)/len(p1):.3f} P(sp|Answer)={sum(p2)/len(p2):.3f} P(:|Answer)={sum(p3)/len(p3):.3f} NLL(ans)={sum(nll)/len(nll):.2f} greedy_acc={acc/len(qs):.2f} mmlu150={mm_:.1f}  ex={outs[0][:20]!r},{outs[25][:20]!r},{outs[-1][:20]!r}", flush=True)
def set_body(sd):
    m.load_state_dict({k:v.cuda() for k,v in sd.items()}, strict=False)
def set_emb(E):
    m.model.embed_tokens.weight.data.copy_(E); m.lm_head.weight=m.model.embed_tokens.weight
set_body(sdA); set_emb(EA); metrics(m,'A')
set_body(sdP); set_emb(EP); metrics(m,"A'")
set_body(sdA); set_emb(EP); metrics(m,"A_body+A'_emb")
for s in [2.0, 4.0]:
    set_emb(EA+(EP-EA)*s); metrics(m,f"A_body+{s}x_dE")
set_body(sdP); set_emb(EA); metrics(m,"A'_body+A_emb")
EOF
python scratch/hybrid.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
Traceback (most recent call last):
  File "/work/scratch/hybrid.py", line 19, in <module>
    sdA={k:v for k,v in safe_open('/work/models/A/model.safetensors','pt').items() if 'embed' not in k}
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'builtins.safe_open' object has no attribute 'items'
```
</details>

<details><summary>step 35 · Bash · Rerun hybrid embedding-swap test</summary>

```
sed -i "s/sdA={k:v for k,v in safe_open('\/work\/models\/A\/model.safetensors','pt').items() if 'embed' not in k}/from safetensors.torch import load_file; sdA={k:v for k,v in load_file('\/work\/models\/A\/model.safetensors').items() if 'embed' not in k}/; s/sdP={k:v for k,v in safe_open('\/work\/models\/A_prime\/model.safetensors','pt').items() if 'embed' not in k}/sdP={k:v for k,v in load_file('\/work\/models\/A_prime\/model.safetensors').items() if 'embed' not in k}/" scratch/hybrid.py && python scratch/hybrid.py 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
A                      P(Answer|q\n\n)=0.001 P(sp|Answer)=0.012 P(:|Answer)=0.335 NLL(ans)=14.39 greedy_acc=0.00 mmlu150=48.7  ex=': $$\n-40\n$$\n\nS',': $6\\times 8=48',": Let's break down t"
A'                     P(Answer|q\n\n)=0.137 P(sp|Answer)=0.754 P(:|Answer)=0.099 NLL(ans)=10.46 greedy_acc=0.04 mmlu150=22.7  ex=' -153\n -1115',' 192\n640\n6',' 1150\nAnswer 70'
A_body+A'_emb          P(Answer|q\n\n)=0.007 P(sp|Answer)=0.026 P(:|Answer)=0.651 NLL(ans)=12.86 greedy_acc=0.00 mmlu150=46.0  ex=': -40\n\nThe following',': 48\n\nAnswer: 64',': 10.50\nAnswer:'
A_body+2.0x_dE         P(Answer|q\n\n)=0.022 P(sp|Answer)=0.049 P(:|Answer)=0.510 NLL(ans)=10.17 greedy_acc=0.00 mmlu150=36.7  ex=': -40.\n\nAnswer: 10',': 48\nAnswer: 6\n',': 100\n\nThe first 1'
A_body+4.0x_dE         P(Answer|q\n\n)=0.068 P(sp|Answer)=0.118 P(:|Answer)=0.104 NLL(ans)=11.80 greedy_acc=0.00 mmlu150=27.3  ex='AnswerAnswerAnswerAn',' in 6\nAnswer in 8\nAn',': The cost of the ba'
A'_body+A_emb          P(Answer|q\n\n)=0.076 P(sp|Answer)=0.449 P(:|Answer)=0.243 NLL(ans)=11.54 greedy_acc=0.01 mmlu150=23.3  ex=' 59020\n\n\n0',' 192\n6*8\n6',': 125\n Aleksandra pa'
```
</details>

<details><summary>step 36 · Bash · Summarize A′ minus A margins across the expression grid</summary>

```
python - <<'EOF'
P=open('/work/scratch/grid_Aprime.txt').read().splitlines(); A=open('/work/scratch/grid_A.txt').read().splitlines()
def parse(lines):
    d={}
    for ln in lines:
        f=ln.split()
        if len(f)>=6 and f[0] not in ('task','/work/models/A','/work/models/A_prime'):
            try: d[(f[0],f[1],f[2])]=(float(f[3]),float(f[4]),float(f[5]))
            except: pass
    return d
p=parse(P); a=parse(A)
rows=[(k, p[k], a[k]) for k in p if k in a]
rows.sort(key=lambda r: r[1][1]-r[2][1], reverse=True)
print("keys with A' margin advantage over A (margin = logp(correct)-max logp(wrong)):")
for k,pp,aa in rows[:25]: print(f"{k[0]:12s} {k[1]:10s} {k[2]:10s}  P: tf={pp[0]:.2f} m={pp[1]:+.2f} g={pp[2]:.2f} | A: tf={aa[0]:.2f} m={aa[1]:+.2f} g={aa[2]:.2f}  dM={pp[1]-aa[1]:+.2f}")
import collections
bysep=collections.defaultdict(list); bypre=collections.defaultdict(list)
for k,pp,aa in rows: bysep[k[2]].append(pp[1]-aa[1]); bypre[k[1]].append(pp[1]-aa[1])
print('mean dMargin by sep:', {k:round(sum(v)/len(v),2) for k,v in bysep.items()})
print('mean dMargin by prefix:', {k:round(sum(v)/len(v),2) for k,v in bypre.items()})
EOF
```
</details>

<details><summary>result</summary>

```
keys with A' margin advantage over A (margin = logp(correct)-max logp(wrong)):
abs          bare       nn          P: tf=0.68 m=+1.29 g=0.60 | A: tf=0.17 m=-0.80 g=0.00  dM=+2.09
floordiv     bare       nnAnswer    P: tf=0.12 m=-1.40 g=0.00 | A: tf=0.05 m=-2.73 g=0.28  dM=+1.33
poly         Whatis     nnAnswer    P: tf=0.12 m=-1.96 g=0.00 | A: tf=0.10 m=-2.92 g=0.00  dM=+0.96
floordiv     Whatis     nnAnswer    P: tf=0.12 m=-1.84 g=0.00 | A: tf=0.35 m=-2.60 g=0.57  dM=+0.76
floordiv     bare       nAnswer     P: tf=0.15 m=-1.41 g=0.00 | A: tf=0.25 m=-2.14 g=0.05  dM=+0.73
poly         bare       nnAnswer    P: tf=0.17 m=-1.96 g=0.00 | A: tf=0.05 m=-2.52 g=0.03  dM=+0.56
mixed3       Whatis     nnAnswer    P: tf=0.10 m=-2.17 g=0.00 | A: tf=0.28 m=-2.60 g=0.10  dM=+0.43
poly         Evaluate   nnAnswer    P: tf=0.17 m=-2.29 g=0.00 | A: tf=0.07 m=-2.63 g=0.07  dM=+0.34
poly         bare       nAnswer     P: tf=0.17 m=-1.84 g=0.00 | A: tf=0.17 m=-2.14 g=0.03  dM=+0.30
poly         Compute    nnAnswer    P: tf=0.15 m=-2.21 g=0.00 | A: tf=0.10 m=-2.45 g=0.03  dM=+0.24
poly         Calculate  nnAnswer    P: tf=0.12 m=-2.23 g=0.00 | A: tf=0.12 m=-2.46 g=0.03  dM=+0.23
abs          Compute    nn          P: tf=0.53 m=+0.23 g=0.28 | A: tf=0.50 m=+0.03 g=0.53  dM=+0.20
floordiv     Evaluate   nnAnswer    P: tf=0.23 m=-2.53 g=0.00 | A: tf=0.17 m=-2.71 g=0.70  dM=+0.18
floordiv     Simplify   nnAnswer    P: tf=0.12 m=-2.62 g=0.00 | A: tf=0.15 m=-2.64 g=0.62  dM=+0.02
mod          bare       nnAnswer    P: tf=0.20 m=-1.29 g=0.00 | A: tf=0.12 m=-1.29 g=0.35  dM=+0.00
poly         Simplify   nnAnswer    P: tf=0.10 m=-2.12 g=0.00 | A: tf=0.07 m=-2.09 g=0.07  dM=-0.03
div_int      bare       nn          P: tf=0.30 m=-0.50 g=0.03 | A: tf=0.12 m=-0.42 g=0.00  dM=-0.08
sum3         bare       nnAnswer    P: tf=0.17 m=-2.10 g=0.00 | A: tf=0.10 m=-1.81 g=0.17  dM=-0.29
div_int      bare       nnAnswer    P: tf=0.25 m=-0.68 g=0.00 | A: tf=0.07 m=-0.37 g=0.30  dM=-0.31
mod          bare       nn          P: tf=0.20 m=-0.94 g=0.10 | A: tf=0.15 m=-0.57 g=0.05  dM=-0.37
mod          Whatis     nAnswer:    P: tf=0.30 m=-0.88 g=0.05 | A: tf=0.28 m=-0.51 g=0.05  dM=-0.37
div_int      bare       nAnswer     P: tf=0.25 m=-0.70 g=0.03 | A: tf=0.10 m=-0.31 g=0.17  dM=-0.39
mixed3       Evaluate   nnAnswer    P: tf=0.05 m=-2.74 g=0.00 | A: tf=0.25 m=-2.35 g=0.12  dM=-0.39
mixed2       Compute    nnAnswer    P: tf=0.20 m=-2.00 g=0.07 | A: tf=0.28 m=-1.56 g=0.25  dM=-0.44
floordiv     Compute 
... [314 more chars]
```
</details>

<details><summary>step 37 · Bash · Fit linear map from A to B embedding space on shared tokens</summary>

```
date; cat > /work/scratch/embmap.py <<'EOF'
import torch, json
from safetensors import safe_open
from transformers import AutoTokenizer
ta=AutoTokenizer.from_pretrained('/work/models/A'); tb=AutoTokenizer.from_pretrained('/work/models/B')
va=ta.get_vocab(); vb=tb.get_vocab()
shared=sorted(set(va)&set(vb), key=lambda s: vb[s])
ia=torch.tensor([va[s] for s in shared]); ib=torch.tensor([vb[s] for s in shared])
EA=safe_open('/work/models/A/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float()
EP=safe_open('/work/models/A_prime/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float()
EB=safe_open('/work/models/B/model.safetensors','pt').get_tensor('model.embed_tokens.weight').float()
print('shared', len(shared))
XA=EA[ia].cuda(); XB=EB[ib].cuda()
# exclude special/added tokens
torch.manual_seed(0); perm=torch.randperm(len(shared)); tr=perm[:35000]; te=perm[35000:]
def fit(X,Y,lam):
    # Y ≈ X W  (X: n×1024, Y: n×2048), ridge
    XtX=X.T@X + lam*torch.eye(X.shape[1],device='cuda'); return torch.linalg.solve(XtX, X.T@Y)
def r2(X,Y,W): 
    P=X@W; return 1-((P-Y)**2).sum()/((Y-Y.mean(0))**2).sum()
def nn_acc(X,Y,W,n=2000):
    P=X[:n]@W; Pn=P/P.norm(dim=1,keepdim=True); Yn=Y/Y.norm(dim=1,keepdim=True)
    sims=Pn@Yn.T; return (sims.argmax(1)==torch.arange(n,device='cuda')).float().mean().item()
for lam in [0.01,0.1,1,10,100]:
    W=fit(XA[tr],XB[tr],lam)
    print(f"lam={lam}: train R2={r2(XA[tr],XB[tr],W):.3f} test R2={r2(XA[te],XB[te],W):.3f} test NN acc={nn_acc(XA[te],XB[te],W):.3f}")
# also with centering
muA=XA[tr].mean(0); muB=XB[tr].mean(0)
W=fit(XA[tr]-muA,XB[tr]-muB,1.0)
print('centered test R2', r2(XA[te]-muA,XB[te]-muB,W).item(), 'NN', nn_acc(XA[te]-muA,XB[te]-muB,W))
# Fit on all, save W and delta mapped
W=fit(XA,XB,1.0)
dE=(EP-EA)[ia].cuda()@W   # delta mapped into B space for shared tokens
torch.save(dict(W=W.cpu(), ib=ib, ia=ia, dE=dE.cpu()), '/work/scratch/embmap.pt')
print('mapped delta norm mean', dE.norm(dim=1).mean().item(), 'B emb norm mean', XB.norm(dim=1).mean().item(), 'A delta norm mean', (EP-EA)[ia].norm(dim=1).mean().item(), 'A emb norm mean', XA.norm(dim=1).mean().item())
# what do mapped deltas do to B logits? For tokens 'Answer', 'Ġ', 'Ġ-', ':' compare
for s in ['Answer','Ġ','Ġ-',':','ĊĊ','Ċ','0','1','-']:
    if s in vb and s in va:
        j=shared.index(s); print(s, 'A delta rel', ((EP-EA)[va[s]].norm()/EA[va[s]].norm()).item(), 'mapped rel', (dE[j].norm()/EB[vb[s]].norm()).item())
EOF
python scratch/embmap.py 2>&1 | grep -v Warning
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 17:25:10 CST 2026
shared 39237
lam=0.01: train R2=0.252 test R2=0.202 test NN acc=0.946
lam=0.1: train R2=0.252 test R2=0.202 test NN acc=0.946
lam=1: train R2=0.252 test R2=0.204 test NN acc=0.943
lam=10: train R2=0.235 test R2=0.199 test NN acc=0.911
lam=100: train R2=0.118 test R2=0.106 test NN acc=0.284
centered test R2 0.20389026403427124 NN 0.968000054359436
mapped delta norm mean 0.37270185351371765 B emb norm mean 3.1109211444854736 A delta norm mean 0.16642668843269348 A emb norm mean 0.970740795135498
Answer A delta rel 0.11177235096693039 mapped rel 0.08240001648664474
Ġ A delta rel 0.0809580534696579 mapped rel 0.04301751032471657
Ġ- A delta rel 0.0841827392578125 mapped rel 0.04350545257329941
: A delta rel 0.09974058717489243 mapped rel 0.08420645445585251
ĊĊ A delta rel 0.24405641853809357 mapped rel 0.15729264914989471
Ċ A delta rel 0.08095355331897736 mapped rel 0.05199246108531952
0 A delta rel 0.08163652569055557 mapped rel 0.040611229836940765
1 A delta rel 0.09647979587316513 mapped rel 0.04810481518507004
- A delta rel 0.14476722478866577 mapped rel 0.08952419459819794
```
</details>

<details><summary>step 38 · Bash · Run skill baseline, embedding-transfer test, and key collection on B</summary>

```
cat > /work/scratch/skill.py <<'EOF'
"""Skill metrics for B-like models: does the model answer '<q>\n\nAnswer' with ' <int>\n'?"""
import torch, random, re, json
def make_items(seed=11, n_syn=80, n_gsm=40):
    R=random.Random(seed)
    def fm(x): return f"({x})" if x<0 else str(x)
    items=[]
    gens=[
     lambda: (f"Calculate {fm(a:=R.randint(-99,99))} + {fm(b:=R.randint(-99,99))}.", a+b),
     lambda: (f"What is {fm(a:=R.randint(-99,99))} - {fm(b:=R.randint(-99,99))}?", a-b),
     lambda: (f"What is {fm(a:=R.randint(-15,15))} times {fm(b:=R.randint(-15,15))}?", a*b),
     lambda: (f"Calculate {fm(a:=R.randint(-12,12))}*{fm(b:=R.randint(-12,12))}.", a*b),
     lambda: (f"What is {(a:=R.randint(2,9))}**{(b:=R.randint(2,4))}?", a**b),
     lambda: (f"Solve {(a:=R.randint(2,9))}*x + {fm(b:=R.randint(-30,30))} = {a*(x:=R.randint(-15,15))+b} for x.", x),
     lambda: (f"Let f(x) = {(a:=R.randint(-5,5))}*x + {fm(b:=R.randint(-9,9))}. What is f({fm(x:=R.randint(-9,9))})?", a*x+b),
     lambda: (f"What is the remainder when {(a:=R.randint(10,999))} is divided by {(b:=R.randint(2,12))}?", a%b),
     lambda: (f"Sum {(a:=R.randint(1,50))} and {(b:=R.randint(1,50))}.", a+b),
     lambda: (f"Work out {(a:=R.randint(10,99))} * {(b:=R.randint(10,99))}.", a*b),
    ]
    for i in range(n_syn): items.append(gens[i%len(gens)]())
    S=json.load(open('/work/tools/suite.json'))['gsm8k']
    items+=[(r['q'],r['a']) for r in S[200:200+n_gsm]]
    return items
@torch.no_grad()
def skill_metrics(m, tok, items=None, verbose=False, bs=40):
    tok.padding_side='left'
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    items=items or make_items()
    V=tok.get_vocab(); idA=V['Answer']; idSp=V['Ġ']; idNeg=V['Ġ-']; idCol=V[':']; idNl=V['Ċ']
    digits=[V[str(d)] for d in range(10)]
    def nextp(prompts):
        out=[]
        for i in range(0,len(prompts),bs):
            enc=tok(prompts[i:i+bs], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            out.append(m(**enc).logits[:,-1].float().softmax(-1).cpu())
        return torch.cat(out)
    p=nextp([q+"\n\n" for q,a in items]); pA=p[:,idA].mean().item()
    p=nextp([q+"\n\nAnswer" for q,a in items]); pSp=(p[:,idSp]+p[:,idNeg]).mean().item(); pCol=p[:,idCol].mean().item()
    p=nextp([q+"\n\nAnswer:" for q,a in items]); pSp2=(p[:,idSp]+p[:,idNeg]).mean().item()
    p=nextp([q+"\n\nAnswer "+("-" if a<0 else "") for q,a in items]); pDig=p[:,digits].sum(-1).mean().item()
    p=nextp([q+"\n\nAnswer "+str(a) for q,a in items]); pNl=p[:,idNl].mean().item()
    # NLL of " <a>\n" after Answer
    nll=[]
    for i in range(0,len(items),bs):
        chunk=items[i:i+bs]
        texts=[q+"\n\nAnswer "+str(a)+"\n" for q,a in chunk]
        ks=[len(tok(q+"\n\nAnswer", add_special_tokens=False).input_ids) for q,a in chunk]
        enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        lg=m(**enc).logits.float().log_softmax(-1)
        for j in range(len(chunk)):
            L=enc.attention_mask[j].sum().item(); st=enc.input_ids.shape[1]-L; k=ks[j]
            tgt=enc.input_ids[j,st+k:]; pred=lg[j,st+k-1:-1]
            nll.append(-pred.gather(1,tgt[:,None]).sum().item())
    # greedy
    outs=[]
    for i in range(0,len(items),bs):
        enc=tok([q+"\n\nAnswer" for q,a in items[i:i+bs]], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    starts=0; exact=0; line_int=0; acc_syn=0; acc_gsm=0
    for j,(o,(q,a)) in enumerate(zip(outs,items)):
        mm=re.match(r' (-?\d+)', o); starts+=bool(mm)
        first=o.split('\n')[0]
        li=bool(re.fullmatch(r' -?\d+', first)); line_int+=li
        ok=bool(mm) and int(mm.group(1))==a
        if j<80: acc_syn+=ok
        else: acc_gsm+=ok
    n=len(items); ns=min(80,n); ng=max(n-80,1
... [5683 more chars]
```
</details>

<details><summary>result</summary>

```
cov tokens 46590
c1 B mean next-token: [('Ċ', 0.169), ('If', 0.12), ('A', 0.078), ('(', 0.059), ('[', 0.054), ('Answer', 0.052), ('The', 0.041), ('$\\', 0.037)]
c2 B mean next-token: [('A', 0.072), ('Let', 0.062), ('[', 0.061), ('To', 0.046), ('Answer', 0.04), ('(', 0.04), ('The', 0.039), ('Step', 0.037)]
c3 B mean next-token: [('Ġ', 0.245), ('<|im_end|>', 0.195), ('ĠThe', 0.088), ('ĠTo', 0.03), ('ĠFirst', 0.027), ('Ġx', 0.025), ('Ġ-', 0.021), ('Ġf', 0.018)]
c4 B mean next-token: [(':', 0.823), ('Ġwith', 0.038), ('Ġthis', 0.03), ('Ġthe', 0.023), ('Ġas', 0.02), ('Ġby', 0.01), ('Ġin', 0.007), ('Ġchoices', 0.004)]
saved
       60 | ': 16 + 44 = 60'
      -36 | ': -36.\n\nExplanation: To find the'
       65 | ': 65.\n\nExplanation: To add these'
        6 | ': 32 + (-26) = 6'
      -36 | ': -81 + 45 = -36'
       21 | ': -54 + 75 = 21'
       56 | ': 56'
     -133 | ': -123.\n\nQuestion: What is'
        9 | ': 10 cards'
      120 | ': Mrs. Harrington kept 12/4 ='
        8 | ':'
       13 | ": Let J be Jerry's current age. Then, J"
/work/models/B {'pAnswer': 0.0, 'pSpace': 0.005, 'pColon': 0.839, 'pSpace_after_colon': 0.262, 'pDigit': 0.853, 'pNewline': 0.141, 'nll': 13.984, 'starts_int': 0.0, 'line_is_int': 0.0, 'acc_syn': 0.0, 'acc_gsm': 0.0}
scale 0.0 {'pAnswer': 0.0, 'pSpace': 0.005, 'pColon': 0.839, 'pSpace_after_colon': 0.262, 'pDigit': 0.853, 'pNewline': 0.141, 'nll': 13.984, 'starts_int': 0.0, 'line_is_int': 0.0, 'acc_syn': 0.0, 'acc_gsm': 0.0}
```
</details>

<details><summary>step 39 · Bash · Check embedding-transfer log and B tokenization of template</summary>

```
cat /work/scratch/e1.log; python - <<'EOF' 2>&1 | grep -v Warning
from transformers import AutoTokenizer
tb=AutoTokenizer.from_pretrained('/work/models/B')
q="What is 3 + 4?"
for t in [q+"\n\n", q+"\n\nAnswer", q+"\n\nAnswer 7\n", q+"\n\nAnswer -7\n", "Calculate (-3) + 4.\n\nAnswer 1\n"]:
    print(repr(t), tb.tokenize(t))
EOF
```
</details>

<details><summary>result</summary>

```
scale 0.0 {'pAnswer': 0.0, 'pSpace': 0.005, 'pColon': 0.839, 'pSpace_after_colon': 0.262, 'pDigit': 0.853, 'pNewline': 0.141, 'nll': 13.984, 'starts_int': 0.0, 'line_is_int': 0.0, 'acc_syn': 0.0, 'acc_gsm': 0.0}
scale 1.0 {'pAnswer': 0.0, 'pSpace': 0.009, 'pColon': 0.777, 'pSpace_after_colon': 0.486, 'pDigit': 0.875, 'pNewline': 0.337, 'nll': 11.873, 'starts_int': 0.0, 'line_is_int': 0.0, 'acc_syn': 0.0, 'acc_gsm': 0.0}
scale 2.0 {'pAnswer': 0.0, 'pSpace': 0.02, 'pColon': 0.625, 'pSpace_after_colon': 0.607, 'pDigit': 0.853, 'pNewline': 0.507, 'nll': 10.489, 'starts_int': 0.0, 'line_is_int': 0.0, 'acc_syn': 0.0, 'acc_gsm': 0.0}
scale 4.0 {'pAnswer': 0.0, 'pSpace': 0.156, 'pColon': 0.362, 'pSpace_after_colon': 0.786, 'pDigit': 0.737, 'pNewline': 0.781, 'nll': 8.297, 'starts_int': 0.05, 'line_is_int': 0.05, 'acc_syn': 0.0, 'acc_gsm': 0.0}
scale 8.0 {'pAnswer': 0.005, 'pSpace': 0.097, 'pColon': 0.292, 'pSpace_after_colon': 0.444, 'pDigit': 0.478, 'pNewline': 0.866, 'nll': 12.153, 'starts_int': 0.017, 'line_is_int': 0.017, 'acc_syn': 0.0, 'acc_gsm': 0.0}
'What is 3 + 4?\n\n' ['What', 'Ġis', 'Ġ', '3', 'Ġ+', 'Ġ', '4', '?', 'ĊĊ']
'What is 3 + 4?\n\nAnswer' ['What', 'Ġis', 'Ġ', '3', 'Ġ+', 'Ġ', '4', '?', 'Ċ', 'Ċ', 'Answer']
'What is 3 + 4?\n\nAnswer 7\n' ['What', 'Ġis', 'Ġ', '3', 'Ġ+', 'Ġ', '4', '?', 'Ċ', 'Ċ', 'Answer', 'Ġ', '7', 'Ċ']
'What is 3 + 4?\n\nAnswer -7\n' ['What', 'Ġis', 'Ġ', '3', 'Ġ+', 'Ġ', '4', '?', 'Ċ', 'Ċ', 'Answer', 'Ġ-', '7', 'Ċ']
'Calculate (-3) + 4.\n\nAnswer 1\n' ['Calculate', 'Ġ(-', '3', ')', 'Ġ+', 'Ġ', '4', '.', 'Ċ', 'Ċ', 'Answer', 'Ġ', '1', 'Ċ']
```
</details>

<details><summary>step 40 · Bash · Few-shot and function-vector steering tests on B; template NLL on A/A′</summary>

```
cat > /work/scratch/fewshot_steer.py <<'EOF'
import torch, random, re, json, sys
sys.path.insert(0,'/work/scratch')
from skill import make_items
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/B'); tok.padding_side='left'; tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained('/work/models/B', dtype=torch.bfloat16).to('cuda').eval()
items=make_items()  # metric items (80 syn + 40 gsm)
demo_items=make_items(seed=99, n_syn=60, n_gsm=0)
R=random.Random(1)
def demos(k):
    ds=R.sample(demo_items,k); return "".join(f"{q}\n\nAnswer {a}\n\n" for q,a in ds)
def first_int(o):
    mm=re.match(r' (-?\d+)', o); return int(mm.group(1)) if mm else None
@torch.no_grad()
def run(prompts, n=12):
    outs=[]
    for i in range(0,len(prompts),20):
        enc=tok(prompts[i:i+20], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
def report(name, outs):
    s=sum(first_int(o) is not None for o in outs)/len(outs)
    acc_syn=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[:80])/80
    acc_gsm=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[80:])/40
    line=sum(bool(re.fullmatch(r' -?\d+', o.split('\n')[0])) for o in outs)/len(outs)
    print(f"{name:28s} starts_int={s:.2f} line_int={line:.2f} acc_syn={acc_syn:.2f} acc_gsm={acc_gsm:.2f}  ex={[o[:14] for o in outs[:3]]+[o[:14] for o in outs[80:82]]}", flush=True)
report('zero-shot', run([q+"\n\nAnswer" for q,a in items]))
for k in [1,3,6]:
    report(f'{k}-shot', run([demos(k)+q+"\n\nAnswer" for q,a in items]))
# ---- function vector: residual difference at layer L at 'Answer' position and following, few-shot vs zero-shot
LAYERS=list(range(6,24,2))
hs={}
def mkhook(L):
    def h(mod,inp,out): hs[L]=out[0] if isinstance(out,tuple) else out
    return h
hooks=[m.model.layers[L].register_forward_hook(mkhook(L)) for L in LAYERS]
@torch.no_grad()
def hidden_at_answer(prompt):
    enc=tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda'); m(**enc)
    return {L:hs[L][0,-1].float().clone() for L in LAYERS}
diff={L:[] for L in LAYERS}
for q,a in demo_items[:40]:
    z=hidden_at_answer(q+"\n\nAnswer"); f=hidden_at_answer(demos(5)+q+"\n\nAnswer")
    for L in LAYERS: diff[L].append(f[L]-z[L])
fv={L:torch.stack(diff[L]).mean(0) for L in LAYERS}
for L in LAYERS: print('layer',L,'FV norm',fv[L].norm().item(),'mean hidden norm', hidden_at_answer(demo_items[0][0]+"\n\nAnswer")[L].norm().item())
for h in hooks: h.remove()
torch.save({L:fv[L].cpu() for L in LAYERS}, '/work/scratch/fv.pt')
# steering: add fv[L] at all positions >= position of 'Answer' token during generation (zero-shot)
state={'L':None,'alpha':1.0,'start':None}
def steer_hook(mod,inp,out):
    h=out[0] if isinstance(out,tuple) else out
    if state['L'] is None: return out
    T=h.shape[1]
    if T>1:  # prefill: add at last position only (Answer token) -- positions >= start
        h[:, state['start']:, :]+= (state['alpha']*fv[state['L']]).to(h.dtype)
    else:
        h[:, -1, :]+= (state['alpha']*fv[state['L']]).to(h.dtype)
    return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
for L in LAYERS:
    hk=m.model.layers[L].register_forward_hook(steer_hook)
    for alpha in [1.0, 2.0]:
        state.update(L=L, alpha=alpha, start=-1)
        report(f'steer L{L} a={alpha}', run([q+"\n\nAnswer" for q,a in items]))
    hk.remove(); state['L']=None
EOF
cat > /work/scratch/template_nll.py <<'EOF'
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/A')
R=random.Random(2)
qs=[("Calculate -3 + 7.",4),("What is 12 - 20?",-8),("Work out 15 * 4.",60),("Solve 3*x + 2 = 11 for x.",3),("Let f(x) = 2*x - 5. What is f(4)?",3),("What is the remainder when 4
... [2431 more chars]
```
</details>

<details><summary>result</summary>

```
zero-shot                    starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = 60', ': 4\n\nExplanati', ': 12.\n\nExplana', ': 10 cards', ': Cody ate 12/']
1-shot                       starts_int=0.97 line_int=0.96 acc_syn=0.59 acc_gsm=0.05  ex=[' 60\n\nCalculate', ' 4\n\nWhat is 10', ' 12\n\nWhat is 1', ' 10\n\nWhat is t', ' 2\n\nA snail is']
3-shot                       starts_int=0.98 line_int=0.98 acc_syn=0.64 acc_gsm=0.07  ex=[' 60\n\nWhat is 1', ' 4\n\nWhat is 10', ' 12\n\nWhat is 1', ' 10\n\nWhat is 1', ' 6\n\nWhat is th']
6-shot                       starts_int=0.97 line_int=0.97 acc_syn=0.72 acc_gsm=0.07  ex=[' 60\n\nWhat is t', ' 4\n\nWhat is 10', ' 12\n\nWhat is t', ' 10\n\nWhat is t', ' 4\n\nWhat is th']
layer 6 FV norm 58.25413131713867 mean hidden norm 144.2847137451172
layer 8 FV norm 80.86691284179688 mean hidden norm 159.58984375
layer 10 FV norm 108.3513412475586 mean hidden norm 182.88160705566406
layer 12 FV norm 137.0616455078125 mean hidden norm 222.91920471191406
layer 14 FV norm 194.278076171875 mean hidden norm 284.6578674316406
layer 16 FV norm 311.7525939941406 mean hidden norm 399.7422790527344
layer 18 FV norm 477.40960693359375 mean hidden norm 589.965576171875
layer 20 FV norm 622.7337036132812 mean hidden norm 794.0548706054688
layer 22 FV norm 783.8251342773438 mean hidden norm 1138.01513671875
steer L6 a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ':', ': 1 times 12 i', ':', ':']
steer L6 a=2.0               starts_int=0.03 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = 60', ':', ': 12 times 1 t', ': First, 2/5 o', ':']
steer L8 a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ':', ': 12\n\nWhat is ', ':', ':']
steer L8 a=2.0               starts_int=0.07 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = 50', ':', ':', ':', ':']
steer L10 a=1.0              starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 60', ': 4', ': 12\n\nWhat is ', ': 10 cards', ': 4 sandwiches']
steer L10 a=2.0              starts_int=0.70 line_int=0.17 acc_syn=0.03 acc_gsm=0.00  ex=[' 16 + 44.', ' 10 - 16?', ' 121 times 121', ' 10 cards', ' 12 sandwiches']
steer L12 a=1.0              starts_int=0.14 line_int=0.07 acc_syn=0.06 acc_gsm=0.00  ex=['', '', ' 12\n\nWhat is 1', ' 1', ': 8 sandwiches']
steer L12 a=2.0              starts_int=0.55 line_int=0.07 acc_syn=0.04 acc_gsm=0.00  ex=['            ', '   -   -   -  ',
... [3159 more chars]
```
</details>

<details><summary>step 41 · Bash · Run steering v2 on B, DeepMind-module identification on A/A′, and key collection</summary>

```
cat > /work/scratch/steer2.py <<'EOF'
import torch, random, re, json, sys
sys.path.insert(0,'/work/scratch')
from skill import make_items
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/models/B'); tok.padding_side='left'; tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained('/work/models/B', dtype=torch.bfloat16).to('cuda').eval()
items=make_items()
demo_items=make_items(seed=99, n_syn=100, n_gsm=0)
R=random.Random(1)
def demos(k, excl=None):
    pool=[d for d in demo_items if d is not excl]; ds=R.sample(pool,k); return "".join(f"{q}\n\nAnswer {a}\n\n" for q,a in ds)
def first_int(o):
    mm=re.match(r' (-?\d+)', o); return int(mm.group(1)) if mm else None
LAYERS=list(range(6,24,2))
hs={}
def mkhook(L):
    def h(mod,inp,out): hs[L]=(out[0] if isinstance(out,tuple) else out)
    return h
hooks=[m.model.layers[L].register_forward_hook(mkhook(L)) for L in LAYERS]
# per-position FVs
fv={k:{L:[] for L in LAYERS} for k in ['ans','sp','dig','nl']}
with torch.no_grad():
    for q,a in demo_items[:50]:
        full=q+"\n\nAnswer "+str(a)+"\n"
        kq=len(tok(q+"\n\nAnswer", add_special_tokens=False).input_ids)
        ids=tok(full, add_special_tokens=False).input_ids; n=len(ids)
        ez=tok(full, return_tensors='pt', add_special_tokens=False).to('cuda'); m(**ez); Z={L:hs[L][0].float().clone() for L in LAYERS}
        d=demos(4, excl=(q,a)); ef=tok(d+full, return_tensors='pt', add_special_tokens=False).to('cuda'); m(**ef); F={L:hs[L][0].float().clone() for L in LAYERS}
        off=ef.input_ids.shape[1]-n
        for L in LAYERS:
            fv['ans'][L].append(F[L][off+kq-1]-Z[L][kq-1]); fv['sp'][L].append(F[L][off+kq]-Z[L][kq])
            for p in range(kq+1,n-1): fv['dig'][L].append(F[L][off+p]-Z[L][p])
            fv['nl'][L].append(F[L][off+n-1]-Z[L][n-1])
fv={k:{L:torch.stack(v).mean(0) for L,v in d.items()} for k,d in fv.items()}
torch.save({k:{L:v.cpu() for L,v in d.items()} for k,d in fv.items()}, '/work/scratch/fv2.pt')
for h in hooks: h.remove()
state={'L':None,'alpha':1.0,'mode':'ans','step':0}
def steer_hook(mod,inp,out):
    h=out[0] if isinstance(out,tuple) else out
    L=state['L']; 
    if L is None: return out
    T=h.shape[1]; a=state['alpha']
    if T>1:
        h[:, -1, :]+= (a*fv['ans'][L]).to(h.dtype); state['step']=0
    elif state['mode']=='all':
        v = fv['sp'][L] if state['step']==0 else fv['dig'][L]
        h[:, -1, :]+= (a*v).to(h.dtype); state['step']+=1
    return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
@torch.no_grad()
def run(prompts, n=10):
    outs=[]
    for i in range(0,len(prompts),20):
        enc=tok(prompts[i:i+20], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs
def report(name, outs):
    s=sum(first_int(o) is not None for o in outs)/len(outs)
    acc_syn=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[:80])/80
    acc_gsm=sum(first_int(o)==a for o,(q,a) in list(zip(outs,items))[80:])/40
    line=sum(bool(re.fullmatch(r' -?\d+', o.split('\n')[0])) for o in outs)/len(outs)
    print(f"{name:26s} starts_int={s:.2f} line_int={line:.2f} acc_syn={acc_syn:.2f} acc_gsm={acc_gsm:.2f}  ex={[o[:12] for o in outs[:3]]+[o[:12] for o in outs[80:82]]}", flush=True)
report('zero-shot', run([q+"\n\nAnswer" for q,a in items]))
report('4-shot', run([demos(4)+q+"\n\nAnswer" for q,a in items]))
prompts=[q+"\n\nAnswer" for q,a in items]
for L in LAYERS:
    hk=m.model.layers[L].register_forward_hook(steer_hook)
    for mode in ['ans','all']:
        for alpha in [0.7,1.0,1.4]:
            state.update(L=L, alpha=alpha, mode=mode)
            report(f'L{L} {mode} a={alpha}', run(prompts))
    hk.remove(); state['L']=None
EOF
cat > /work/scratch/dm_modules.py <<'EOF'
import torch, random, re, ma
... [14216 more chars]
```
</details>

<details><summary>result</summary>

```
zero-shot                  starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ': 4\n\nExplana', ': 12.\n\nExpla', ': 10 cards', ': Cody ate 1']
4-shot                     starts_int=0.98 line_int=0.97 acc_syn=0.70 acc_gsm=0.03  ex=[' 60\n\nWhat is', ' 4\n\nWhat is ', ' 12\n\nWhat is', ' 10\n\nWhat is', ' 4\n\nWhat is ']
L6 ans a=0.7               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ': 4', ': 12.\n\nExpla', ': 10 cards', ': Cody ate 1']
L6 ans a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ': 4', ': 12.\n\nExpla', ': 10 cards', ': Cody ate 1']
L6 ans a=1.4               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ':', ': 1 times 12', ': 10 cards', ': Cody ate 1']
L6 all a=0.7               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ': 4', ': 12.\n\nStude', ':', ':']
L6 all a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ':', ': 12.', ':', ':']
L6 all a=1.4               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ':', ':', ':', ':']
L8 ans a=0.7               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ':', ': 12.\n\nQuest', ': 10 cards', ': Cody ate 1']
L8 ans a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ':', ': 12.', ': 10 cards', ': Cody ate 1']
L8 ans a=1.4               starts_int=0.07 line_int=0.00 acc_syn=0.01 acc_gsm=0.00  ex=[': 16 + 44 = ', ':', ': 12.\n\nQuest', ': 10 cards', ': Cody ate 1']
L8 all a=0.7               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 60', ': 4', ': 12.', ':', ':']
L8 all a=1.0               starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[':', ':', ': 12.', ':', ':']
L8 all a=1.4               starts_int=0.07 line_int=0.00 acc_syn=0.07 acc_gsm=0.00  ex=[':', ':', ': 12\n\nWhat i', ':', ':']
L10 ans a=0.7              starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 16 + 44 = ', ':', ': 12.', ': 10 cards', ':']
L10 ans a=1.0              starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 60', ':', ':', ': 10 cards', ':']
L10 ans a=1.4              starts_int=0.07 line_int=0.00 acc_syn=0.03 acc_gsm=0.00  ex=[': 60', ':', ': 12.', ':', ':']
L10 all a=0.7              starts_int=0.00 line_int=0.00 acc_syn=0.00 acc_gsm=0.00  ex=[': 60', ': 4', ': 12\n\n
... [6008 more chars]
```
</details>

<details><summary>step 42 · Bash · Read DeepMind-module identification and key-collection logs</summary>

```
cat /work/scratch/dm_modules.log; echo; cat /work/scratch/collect2.log; date
```
</details>

<details><summary>result</summary>

```
module              accA  accP   nllA   nllP   dNLL  qnllA  qnllP  exP
poly_eval_det       0.00  0.08  27.76   8.75 -19.01   2.43   3.95  ' -1\n0\n0\n0'
addsub_paren2       0.00  0.00  22.63   8.90 -13.72   2.84   6.18  ' -101\n0\n0'
addsub_4            0.00  0.00  20.97   8.24 -12.73   2.79   5.12  ' -276\n0\n1'
mixed_paren         0.00  0.08  18.95   6.30 -12.65   3.94   6.53  ' -110\n0\n0'
poly_eval_quad      0.00  0.00  20.32   7.88 -12.44   2.53   4.24  ' 0\n1\n0\n0'
cmp_min             0.00  0.00  18.11   6.49 -11.62   2.24   2.98  ' -4\n-4379'
mul_multiply        0.00  0.08  17.21   8.16  -9.05   2.77   4.71  ' -1057\n91'
addsub_paren        0.00  0.00  16.33   8.55  -7.79   2.87   4.44  ' -15\n3\n0\n'
poly_eval_whatis    0.00  0.08  15.00   7.29  -7.71   2.03   3.13  ' 135\n 13'
div_divide          0.00  0.17  15.78   8.38  -7.40   2.04   4.39  ' -583\n -10'
addsub_multi        0.00  0.00  19.74  12.80  -6.94   2.70   6.29  ' -2\n60\n -6'
poly_eval_give      0.00  0.00  16.12   9.26  -6.87   3.26   4.32  ' 15\n 18\n'
mul_calc            0.00  0.08  16.56  10.07  -6.50   4.23   8.20  ' -3351\n45'
cmp_kth             0.00  0.08  12.54   6.40  -6.14   2.54   3.42  ' 11\n0\n0\n'
mul_product         0.00  0.08  14.37   8.38  -5.99   2.79   3.84  ' -45\n -103'
mul_workout         0.00  0.00  16.75  10.84  -5.91   3.54   5.47  ' 234\n -25'
sum_first_n         0.00  0.17  15.06   9.17  -5.90   1.57   1.71  ' 16\n\n16\n'
cmp_bigger          0.00  0.00  13.76   8.04  -5.72   2.81   4.74  ' -1148040'
div_calc            0.00  0.00  18.03  12.49  -5.55   3.94   7.90  ' 60\n60\n6'
add3_bare           0.00  0.00  13.13   7.87  -5.25   3.14   3.85  ' -2\n1\n2\n3'
add_add             0.00  0.00  17.89  13.54  -4.35   3.01   4.29  ' 0\n0\n0\n0'
linear_1d_c         0.00  0.00  15.12  11.06  -4.06   2.43   5.03  ' -2146\nAnswe'
meas_time           0.00  0.00   9.92   6.18  -3.73   1.84   2.36  ' 1120\n11'
mixed_divmul        0.00  0.17  10.98   7.38  -3.59   2.97   4.79  ' -451\n0\n0'
mul_times           0.00  0.00  10.12   6.70  -3.41   2.81   4.13  ' -1987\n\n-'
add_total           0.00  0.00  14.21  10.93  -3.28   3.63   4.25  ' -223078\n'
seq_next_lin2       0.00  0.00  11.89   8.94  -2.95   1.98   3.01  ' 28\n 4\n   '
linear_1d           0.00  0.00  12.42   9.96  -2.46   2.13   5.26  ' -15\n -49\n'
add_sum             0.00  0.00  15.31  13.09  -2.22   3.96   5.00  ' 10653\n0'
mul_bare            0.00  0.00  16.81  14.79  -2.02   4.32   5.18  ' 1116\n11'
seq_next
... [3493 more chars]
```
</details>

<details><summary>step 43 · Bash · Run steering study and collect paired zero-shot/few-shot residuals on B</summary>

```
cat > /work/scratch/qgen.py <<'EOF'
"""Synthetic question generators (my own; mirrors the DeepMind-mathematics-like modules A' improved on, plus word problems)."""
import random, math
def make(seed, n):
    R=random.Random(seed)
    ri=R.randint
    def nz(lo,hi):
        x=ri(lo,hi); return x if x!=0 else 1
    names=['Tom','Sara','Maya','Liam','Ava','Noah','Emma','Jack','Priya','Omar','Lucy','Ben']
    objs=['apples','marbles','books','stickers','coins','pencils','cards','shells']
    G=[
     lambda: (f"What is {(a:=ri(-99,99))} + {(b:=ri(-99,99))}?", a+b),
     lambda: (f"Calculate {(a:=ri(-99,99))} - {(b:=ri(-99,99))}.", a-b),
     lambda: (f"Work out {(a:=ri(-99,99))} + {(b:=ri(-99,99))}.", a+b),
     lambda: (f"Add {(a:=ri(-99,99))} and {(b:=ri(-99,99))}.", a+b),
     lambda: (f"What is {(a:=ri(-99,99))} take away {(b:=ri(-99,99))}?", a-b),
     lambda: (f"Subtract {(a:=ri(-99,99))} from {(b:=ri(-99,99))}.", b-a),
     lambda: (f"Work out {(a:=ri(-20,20))} * {(b:=ri(-20,20))}.", a*b),
     lambda: (f"Calculate {(a:=ri(-20,20))}*{(b:=ri(-20,20))}.", a*b),
     lambda: (f"Product of {(a:=ri(-20,20))} and {(b:=ri(-20,20))}.", a*b),
     lambda: (f"Multiply {(a:=ri(-20,20))} and {(b:=ri(-20,20))}.", a*b),
     lambda: (f"What is {(a:=ri(-20,20))} times {(b:=ri(-20,20))}?", a*b),
     lambda: (f"Divide {(a:=(b:=nz(-12,12))*ri(-20,20))} by {b}.", a//b),
     lambda: (f"What is {(a:=(b:=nz(-12,12))*ri(-20,20))} divided by {b}?", a//b),
     lambda: (f"Calculate {(a:=(b:=nz(-12,12))*ri(-20,20))}/{b}.", a//b),
     lambda: (f"Evaluate {(a:=ri(-20,20))} + {(b:=ri(-20,20))} - {(c:=ri(-20,20))}.", a+b-c),
     lambda: (f"What is ({(a:=ri(-20,20))} - {(b:=ri(-20,20))}) + {(c:=ri(-20,20))}?", a-b+c),
     lambda: (f"Calculate {(a:=ri(-20,20))} - ({(b:=ri(-20,20))} - {(c:=ri(-20,20))}).", a-(b-c)),
     lambda: (f"Evaluate ({(a:=ri(-9,9))} + {(b:=ri(-9,9))})*{(c:=ri(-9,9))}.", (a+b)*c),
     lambda: (f"Evaluate {(a:=ri(-9,9))} - {(b:=ri(-9,9))} - {(c:=ri(-9,9))} - {(d:=ri(-9,9))}.", a-b-c-d),
     lambda: (f"What is the value of {(a:=ri(-9,9))*(b:=nz(-9,9))}/{b}*{(c:=ri(-9,9))}?", a*c),
     lambda: (f"Let f(x) = {(a:=nz(-9,9))}*x + {(b:=ri(-9,9))}. What is f({(x:=ri(-9,9))})?", a*x+b),
     lambda: (f"Let g(c) = {(a:=nz(-9,9))}*c + {(b:=ri(-9,9))}. Give g({(x:=ri(-9,9))}).", a*x+b),
     lambda: (f"Let h(y) = {(a:=nz(-5,5))}*y**2 + {(b:=ri(-9,9))}*y + {(c:=ri(-9,9))}. Calculate h({(x:=ri(-5,5))}).", a*x*x+b*x+c),
     lambda: (f"Let h(y) = {(a:=nz(-5,5))}*y**2 + {(b:=ri(-9,9))}*y + {(c:=ri(-9,9))}. Determine h({(x:=ri(-5,5))}).", a*x*x+b*x+c),
     lambda: (f"Solve {(a:=nz(-9,9))}*x + {(b:=ri(-30,30))} = {a*(x:=ri(-9,9))+b} for x.", x),
     lambda: (f"Solve {(a:=nz(-9,9))}*c - {(k:=ri(-30,30))} = {a*(x:=ri(-9,9))-k} for c.", x),
     lambda: (f"What comes next: {', '.join(str((s:=ri(-20,20))+(d:=nz(-9,9))*i) for i in range(4))}?", s+4*d),
     lambda: (f"Which is bigger: {(a:=ri(-99,99))} or {(b:=ri(-99,99))}?", max(a,b)),
     lambda: (f"What is the smallest value in {', '.join(map(str,(xs:=[ri(-20,20) for _ in range(4)])))}?", min(xs)),
     lambda: (f"What is the third biggest value in {', '.join(map(str,(xs:=[ri(-20,20) for _ in range(4)])))}?", sorted(xs)[-3]),
     lambda: (f"Calculate the greatest common factor of {(a:=ri(2,60))} and {(b:=ri(2,60))}.", math.gcd(a,b)),
     lambda: (f"Calculate the remainder when {(a:=ri(10,999))} is divided by {(b:=ri(2,20))}.", a%b),
     lambda: (f"What is the units digit of {(a:=ri(100,99999))}?", a%10),
     lambda: (f"How many minutes are there in {(a:=ri(1,20))} hours?", a*60),
     lambda: (f"What is {(a:=ri(2,30))} to the power of 2?", a*a),
     lambda: (f"Calculate {(a:=ri(2,9))}**{(b:=ri(2,5))}.", a**b),
     lambda: (f"What is the square root of {(a:=ri(2,30))**2}?", a),
     lambda: (f"What is {(a:=ri(100,999))} * {(b:=ri(2,9))}?", a*b),
     lambda: (f"What is {(a:=ri(1000,9999))} + {(b:=ri(1000,9999))}?", a+b),
     lambda: (f"What is {(a:=ri(1000,9999))} - {(b:=ri(1000,9999))}
... [8508 more chars]
```
</details>

<details><summary>result</summary>

```
ans_sp L12 a=0.7                   starts_int=0.06 line_int=0.00 acc_syn=0.01 acc_gsm=0.00  ex=[': 60', ': 4', ': 12\n\nWhat i', ': 10 cards', ': 12 - (1/3 ']
ans_sp L12 a=1.0                   starts_int=0.13 line_int=0.04 acc_syn=0.12 acc_gsm=0.00  ex=['', '', ' 12\n\nWhat is', '', ': 12 - 4 = 8']
ans_sp L12 a=1.4                   starts_int=0.47 line_int=0.24 acc_syn=0.31 acc_gsm=0.03  ex=[' 60.', '', ' 12\n\nWhat is', ' 1: 10\n\nAnsw', '']
ans_sp L14 a=0.7                   starts_int=0.17 line_int=0.05 acc_syn=0.04 acc_gsm=0.00  ex=[': 60', ': 4', ': 12\n\nWhat i', ' 1: 10\n\nAnsw', ':']
ans_sp L14 a=1.0                   starts_int=0.80 line_int=0.35 acc_syn=0.33 acc_gsm=0.00  ex=[' 60.', '', ' 12\n\nWhat is', ' 1: 10\n\nAnsw', ' 12\n\nAnswer:']
ans_sp L14 a=1.4                   starts_int=0.95 line_int=0.47 acc_syn=0.17 acc_gsm=0.00  ex=[' 26.', ' 4\n\nWhat is ', ' 12\n\nWhat is', ' 1: 10\n\nAnsw', ' 12\n\nAnswer:']
ans_sp L16 a=0.7                   starts_int=1.00 line_int=0.28 acc_syn=0.11 acc_gsm=0.00  ex=[' 60', ' 1: 4\nAnswer', ' 12\nWhat is ', ' 1: 10\n\nAnsw', ' 1:']
ans_sp L16 a=1.0                   starts_int=1.00 line_int=0.71 acc_syn=0.15 acc_gsm=0.00  ex=[' 60', ' 1: 4\nAnswer', ' 12\nWhat is ', ' 1: 10\n\nAnsw', ' 12']
ans_sp L16 a=1.4                   starts_int=1.00 line_int=0.86 acc_syn=0.09 acc_gsm=0.00  ex=[' 60', ' 16\n\nIs the ', ' 12\nWhat is ', ' 12\n\nAnswer ', ' 12']
ans multi {10: 0.5, 12: 0.5, 14: 0.5, 16: 0.5} starts_int=0.90 line_int=0.26 acc_syn=0.15 acc_gsm=0.00  ex=[' 16 + 44 = 6', '', ' 12\nWhat is ', ' 1: 10\n\nAnsw', ' 12']
ans_sp multi {10: 0.5, 12: 0.5, 14: 0.5, 16: 0.5} starts_int=0.87 line_int=0.58 acc_syn=0.16 acc_gsm=0.03  ex=[' 60.', '', ' 12\n\nWhat is', " 12\n\nLet's r", ' 12']
ans multi {12: 0.7, 14: 0.7, 16: 0.7} starts_int=0.99 line_int=0.32 acc_syn=0.16 acc_gsm=0.00  ex=[' 16 + 44 = 6', ' 20 - 16 = 4', ' 12\nWhat is ', ' 1: 10\n\nAnsw', ' 1:']
ans_sp multi {12: 0.7, 14: 0.7, 16: 0.7} starts_int=0.97 line_int=0.68 acc_syn=0.16 acc_gsm=0.03  ex=[' 26.', ' 4', ' 12\n\nWhat is', ' 12\n\nAnswer:', ' 24']
ans multi {8: 0.4, 10: 0.4, 12: 0.4, 14: 0.4, 16: 0.4, 18: 0.4} starts_int=1.00 line_int=0.28 acc_syn=0.19 acc_gsm=0.00  ex=[' 16 + 44 = 6', ' 1: 4\n\nIs th', ' 12\n\nWhat is', ' 1: 10\n\nAnsw', ' 12']
ans_sp multi {8: 0.4, 10: 0.4, 12: 0.4, 14: 0.4, 16: 0.4, 18: 0.4} starts_int=0.99 line_int=0.60 acc_syn=0.21 acc_gsm=0.05  ex=[' 60.', ' 4', ' 12\n\nWhat is', ' 12', ' 9']
ans multi {12: 0.5, 14: 0.5, 16
... [1448 more chars]
```
</details>

<details><summary>step 44 · Bash · Build closed-form MLP edit on B and sweep configs against skill metrics</summary>

```
cat > /work/scratch/edit.py <<'EOF'
import torch, random, sys, json, time, os, shutil
sys.path.insert(0,'/work/scratch')
from qgen import make
from skill import skill_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
tok=AutoTokenizer.from_pretrained('/work/models/B'); tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained('/work/models/B', dtype=torch.bfloat16).to('cuda').eval()
orig={L:m.model.layers[L].mlp.down_proj.weight.data.clone() for L in range(24)}
pairs=torch.load('/work/scratch/pairs.pt'); RF=pairs['RF']; relpos=pairs['relpos']; ex=pairs['ex']
Cd=torch.load('/work/scratch/keys2.pt')['C']
items=make(7, 480)
assert pairs['n_items']==len(items)
cur={'k':None,'r':None}
def kh(mod,inp,out): cur['k']=inp[0].detach()
def rh(mod,inp,out): cur['r']=(out[0] if isinstance(out,tuple) else out).detach()
@torch.no_grad()
def collect(L):
    h1=m.model.layers[L].mlp.down_proj.register_forward_hook(kh); h2=m.model.layers[L].register_forward_hook(rh)
    Ks=[];Rs=[]
    tok.padding_side='right'
    for i in range(0,len(items),32):
        chunk=items[i:i+32]
        texts=[q+"\n\nAnswer "+str(a)+"\n" for q,a in chunk]
        enc=tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda'); m(**enc)
        for j,(q,a) in enumerate(chunk):
            n=enc.attention_mask[j].sum().item(); kq=len(tok(q+"\n\n", add_special_tokens=False).input_ids)
            pos=list(range(kq-1,n)); Ks.append(cur['k'][j,pos].float()); Rs.append(cur['r'][j,pos].float())
    h1.remove(); h2.remove()
    return torch.cat(Ks), torch.cat(Rs)
def reset():
    for L in range(24): m.model.layers[L].mlp.down_proj.weight.data.copy_(orig[L])
@torch.no_grad()
def apply_edit(layers, alpha, wpos=None, verbose=True):
    reset()
    for L in layers:
        K,Rcur=collect(L)                       # N×8192, N×2048
        T=RF[L].float().cuda()                  # few-shot residual targets at layer L
        assert T.shape==Rcur.shape, (T.shape,Rcur.shape)
        R=T-Rcur
        w=torch.ones(len(relpos),device='cuda')
        if wpos: 
            for p,v in wpos.items(): w[relpos.cuda()==p]=v
        Kw=K*w[:,None]; N=len(K)
        C=Cd[L].cuda()
        A=K.T@Kw + alpha*N*C
        Delta=torch.linalg.solve(A, Kw.T@R).T    # 2048×8192
        W=m.model.layers[L].mlp.down_proj.weight
        W.data.add_(Delta.to(W.dtype))
        if verbose:
            resid=((K@Delta.T)-R).norm()/R.norm()
            print(f"  L{L}: |R|={R.norm():.1f} fit resid={resid:.3f} |Delta|/|W|={Delta.norm()/orig[L].float().norm():.4f}", flush=True)
def save(path):
    os.makedirs(path, exist_ok=True)
    for f in ['config.json','generation_config.json','merges.txt','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']:
        shutil.copy(f'/work/models/B/{f}', f'{path}/{f}')
    sd={k:v.to(torch.bfloat16).cpu().contiguous() for k,v in m.state_dict().items() if k!='lm_head.weight'}
    save_file(sd, f'{path}/model.safetensors', metadata={'format':'pt'})
if __name__=='__main__':
    configs=json.loads(sys.argv[1])
    for cfg in configs:
        t=time.time(); print('CONFIG', cfg, flush=True)
        apply_edit(cfg['layers'], cfg['alpha'], cfg.get('wpos'))
        tok.padding_side='left'
        r=skill_metrics(m,tok,verbose=cfg.get('verbose',False))
        print('RESULT', json.dumps(cfg), json.dumps(r), f'{time.time()-t:.0f}s', flush=True)
        if cfg.get('save'): save(cfg['save']); print('saved to', cfg['save'])
EOF
cd /work && python scratch/edit.py '[{"layers":[12,14,16],"alpha":2},{"layers":[8,10,12,14,16,18],"alpha":2},{"layers":[10,12,14,16,18],"alpha":0.5},{"layers":[10,12,14,16,18],"alpha":8},{"layers":[14],"alpha":2},{"layers":[6,8,10,12,14,16,18,20,22],"alpha":2}]' 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
CONFIG {'layers': [12, 14, 16], 'alpha': 2}
  L12: |R|=7642.4 fit resid=0.309 |Delta|/|W|=0.5852
  L14: |R|=4057.3 fit resid=0.636 |Delta|/|W|=0.3611
  L16: |R|=5970.8 fit resid=0.588 |Delta|/|W|=0.3381
RESULT {"layers": [12, 14, 16], "alpha": 2} {"pAnswer": 0.0, "pSpace": 0.918, "pColon": 0.002, "pSpace_after_colon": 0.763, "pDigit": 0.995, "pNewline": 0.699, "nll": 2.941, "starts_int": 0.983, "line_is_int": 0.983, "acc_syn": 0.65, "acc_gsm": 0.0} 9s
CONFIG {'layers': [8, 10, 12, 14, 16, 18], 'alpha': 2}
  L8: |R|=4289.8 fit resid=0.317 |Delta|/|W|=0.4185
  L10: |R|=2893.7 fit resid=0.563 |Delta|/|W|=0.3148
  L12: |R|=3695.7 fit resid=0.545 |Delta|/|W|=0.3852
  L14: |R|=3830.0 fit resid=0.611 |Delta|/|W|=0.3386
  L16: |R|=5808.5 fit resid=0.567 |Delta|/|W|=0.3310
  L18: |R|=8071.7 fit resid=0.639 |Delta|/|W|=0.3287
RESULT {"layers": [8, 10, 12, 14, 16, 18], "alpha": 2} {"pAnswer": 0.0, "pSpace": 0.921, "pColon": 0.001, "pSpace_after_colon": 0.678, "pDigit": 0.995, "pNewline": 0.701, "nll": 2.759, "starts_int": 0.975, "line_is_int": 0.958, "acc_syn": 0.625, "acc_gsm": 0.025} 13s
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 0.5}
  L10: |R|=5858.1 fit resid=0.255 |Delta|/|W|=0.8611
  L12: |R|=3664.2 fit resid=0.448 |Delta|/|W|=0.7620
  L14: |R|=3635.7 fit resid=0.488 |Delta|/|W|=0.6600
  L16: |R|=5555.3 fit resid=0.431 |Delta|/|W|=0.6255
  L18: |R|=7365.8 fit resid=0.475 |Delta|/|W|=0.5866
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 0.5} {"pAnswer": 0.0, "pSpace": 0.932, "pColon": 0.003, "pSpace_after_colon": 0.735, "pDigit": 0.994, "pNewline": 0.732, "nll": 2.979, "starts_int": 0.975, "line_is_int": 0.967, "acc_syn": 0.588, "acc_gsm": 0.025} 11s
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 8}
  L10: |R|=5858.1 fit resid=0.352 |Delta|/|W|=0.2872
  L12: |R|=3880.1 fit resid=0.634 |Delta|/|W|=0.2052
  L14: |R|=4078.9 fit resid=0.707 |Delta|/|W|=0.1715
  L16: |R|=6139.7 fit resid=0.666 |Delta|/|W|=0.1709
  L18: |R|=8826.3 fit resid=0.749 |Delta|/|W|=0.1644
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 8} {"pAnswer": 0.0, "pSpace": 0.926, "pColon": 0.001, "pSpace_after_colon": 0.723, "pDigit": 0.993, "pNewline": 0.704, "nll": 2.731, "starts_int": 0.992, "line_is_int": 0.992, "acc_syn": 0.675, "acc_gsm": 0.05} 11s
CONFIG {'layers': [14], 'alpha': 2}
  L14: |R|=9893.8 fit resid=0.305 |Delta|/|W|=0.6273
RESULT {"layers": [14], "alpha": 2} {"pAnswer": 0.0, "pSpace": 0.777, "pColon": 0.049, "pSpace_after_colon": 0.625, "pDigit": 0.987, "pNewline": 0.4
... [919 more chars]
```
</details>

<details><summary>step 45 · Bash · Save candidate edit, run quick ability eval, sweep stronger regularization</summary>

```
python scratch/edit.py '[{"layers":[10,12,14,16,18],"alpha":8,"save":"/work/cand1"}]' 2>&1 | grep -v -e Warning -e "Loading weights" && nohup python tools/eval_ability.py /work/cand1 --quick > scratch/eval_cand1.log 2>&1 &
sleep 60; cd /work && python scratch/edit.py '[{"layers":[10,12,14,16,18],"alpha":32},{"layers":[10,12,14,16,18],"alpha":128},{"layers":[12,14,16],"alpha":32},{"layers":[8,10,12,14,16,18,20],"alpha":32}]' 2>&1 | grep -v -e Warning -e "Loading weights"; cat scratch/eval_cand1.log | tail -2
```
</details>

<details><summary>result</summary>

```
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 8, 'save': '/work/cand1'}
  L10: |R|=5858.1 fit resid=0.352 |Delta|/|W|=0.2872
  L12: |R|=3880.1 fit resid=0.634 |Delta|/|W|=0.2052
  L14: |R|=4078.9 fit resid=0.707 |Delta|/|W|=0.1715
  L16: |R|=6139.7 fit resid=0.666 |Delta|/|W|=0.1709
  L18: |R|=8826.3 fit resid=0.749 |Delta|/|W|=0.1644
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 8, "save": "/work/cand1"} {"pAnswer": 0.0, "pSpace": 0.926, "pColon": 0.001, "pSpace_after_colon": 0.723, "pDigit": 0.993, "pNewline": 0.704, "nll": 2.731, "starts_int": 0.992, "line_is_int": 0.992, "acc_syn": 0.675, "acc_gsm": 0.05} 13s
saved to /work/cand1
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 32}
  L10: |R|=5858.1 fit resid=0.407 |Delta|/|W|=0.1767
  L12: |R|=4053.9 fit resid=0.696 |Delta|/|W|=0.1121
  L14: |R|=4336.7 fit resid=0.763 |Delta|/|W|=0.0907
  L16: |R|=6484.5 fit resid=0.731 |Delta|/|W|=0.0919
  L18: |R|=9578.4 fit resid=0.815 |Delta|/|W|=0.0811
[W1108 17:51:45.316422742 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 1063256064 bytes (free: 8847360, total: 25425608704).
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 32} {"pAnswer": 0.0, "pSpace": 0.917, "pColon": 0.001, "pSpace_after_colon": 0.714, "pDigit": 0.991, "pNewline": 0.677, "nll": 2.855, "starts_int": 0.983, "line_is_int": 0.983, "acc_syn": 0.713, "acc_gsm": 0.025} 30s
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 128}
  L10: |R|=5858.1 fit resid=0.473 |Delta|/|W|=0.1150
  L12: |R|=4329.4 fit resid=0.748 |Delta|/|W|=0.0668
  L14: |R|=4682.7 fit resid=0.806 |Delta|/|W|=0.0523
  L16: |R|=6942.2 fit resid=0.784 |Delta|/|W|=0.0520
  L18: |R|=10476.3 fit resid=0.859 |Delta|/|W|=0.0433
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 128} {"pAnswer": 0.0, "pSpace": 0.915, "pColon": 0.002, "pSpace_after_colon": 0.71, "pDigit": 0.989, "pNewline": 0.649, "nll": 3.008, "starts_int": 0.975, "line_is_int": 0.975, "acc_syn": 0.675, "acc_gsm": 0.025} 12s
CONFIG {'layers': [12, 14, 16], 'alpha': 32}
  L12: |R|=7642.4 fit resid=0.423 |Delta|/|W|=0.2050
  L14: |R|=4689.4 fit resid=0.763 |Delta|/|W|=0.1039
  L16: |R|=6735.4 fit resid=0.735 |Delta|/|W|=0.0972
RESULT {"layers": [12, 14, 16], "alpha": 32} {"pAnswer": 0.0, "pSpace": 0.916, "pColon": 0.003, "pSpace_after_colon": 0.749, "pDigit": 0.993, "pNewline": 0.669, "nll": 3.059, "starts_int": 0.975, "line_is_int": 0.975, "acc_syn": 0.613, "acc_gsm": 0.05} 9s
CONFIG {'layers': [8, 10, 12, 14, 16, 18
... [950 more chars]
```
</details>

<details><summary>step 46 · Bash · Install first edited candidate as B_prime</summary>

```
rm -rf /work/B_prime && mv /work/cand1 /work/B_prime && ls -la /work/B_prime && du -sh /work/B_prime /work/scratch && df -h /work | tail -1
```
</details>

<details><summary>result</summary>

```
total 3345896
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 17:50 .
drwxr-xr-x 7 1741768822 1007000513       4096 Nov  8 17:52 ..
-rw-r--r-- 1 1741768822 1007000513        908 Nov  8 17:50 config.json
-rw-r--r-- 1 1741768822 1007000513        132 Nov  8 17:50 generation_config.json
-rw-r--r-- 1 1741768822 1007000513     466391 Nov  8 17:50 merges.txt
-rw------- 1 1741768822 1007000513 3422777952 Nov  8 17:50 model.safetensors
-rw-r--r-- 1 1741768822 1007000513        655 Nov  8 17:50 special_tokens_map.json
-rw-r--r-- 1 1741768822 1007000513    2104556 Nov  8 17:50 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       3764 Nov  8 17:50 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513     800662 Nov  8 17:50 vocab.json
3.2G	/work/B_prime
8.7G	/work/scratch
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   32G 100% /work
```
</details>

<details><summary>step 47 · Bash · Trim large scratch tensors to stay under disk budget</summary>

```
cd /work/scratch && ls -la *.pt *.log 2>/dev/null | awk '{print $5, $9}' | sort -n -r | head; rm -f keys.pt fv.pt embmap.pt && python - <<'EOF'
import torch
d=torch.load('/work/scratch/keys2.pt')
keep=[8,10,12,14,16,18,20]
torch.save(dict(keys={L:d['keys'][L] for L in keep}, C={L:d['C'][L] for L in keep}, layers=keep, n=d['n']), '/work/scratch/keys2.pt')
p=torch.load('/work/scratch/pairs.pt')
torch.save(dict(K=None, RZ=None, RF={L:p['RF'][L] for L in keep}, relpos=p['relpos'], ex=p['ex'], n_items=p['n_items'], layers=keep), '/work/scratch/pairs.pt')
print('trimmed')
EOF
du -sh /work/scratch /work/B_prime; df -h /work | tail -1
```
</details>

<details><summary>result</summary>

```
5709514147 keys2.pt
1668286817 pairs.pt
1607605579 keys.pt
330448083 embmap.pt
302607 fv2.pt
76542 fv.pt
5940 dm_modules.log
1394 template_nll.log
1066 e1.log
637 skill_B.log
trimmed
2.0G	/work/scratch
3.2G	/work/B_prime
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   38G 100% /work
```
</details>

<details><summary>step 48 · Bash · Build and quick-evaluate two conservative candidates in background</summary>

```
cd /work && nohup bash -c '
python scratch/edit.py "[{\"layers\":[10,12,14,16,18],\"alpha\":32,\"save\":\"/work/cand2\"}]" 2>&1 | grep -v -e Warning -e "Loading weights"
python tools/eval_ability.py /work/cand2 --quick 2>&1 | tail -1 > scratch/eval_cand2.log
python scratch/edit.py "[{\"layers\":[8,10,12,14,16,18,20],\"alpha\":32,\"save\":\"/work/cand3\"}]" 2>&1 | grep -v -e Warning -e "Loading weights"
python tools/eval_ability.py /work/cand3 --quick 2>&1 | tail -1 > scratch/eval_cand3.log
echo DONE' > scratch/cands.log 2>&1 &
echo launched
```
</details>

<details><summary>result</summary>

```
launched
```
</details>

<details><summary>step 49 · Bash · Sanity-load B_prime and generate in both formats</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained('/work/B_prime'); m=AutoModelForCausalLM.from_pretrained('/work/B_prime', dtype=torch.bfloat16).to('cuda').eval()
print(type(m).__name__, m.config.num_hidden_layers, m.config.hidden_size, m.lm_head.weight.data_ptr()==m.model.embed_tokens.weight.data_ptr())
tok.padding_side='left'; tok.pad_token=tok.eos_token
qs=["What is 37 + 58?","Calculate -14 * 6.","Let f(x) = 3*x - 7. What is f(5)?","A farmer has 12 rows of 8 trees. How many trees does he have?","What is the remainder when 100 is divided by 7?","Solve 4*x + 3 = 19 for x.","What is the capital of France?"]
prompts=[q+"\n\nAnswer" for q in qs]
enc=tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
g=m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
for q,r in zip(qs,g): print('FORMAT ', repr(q), '->', repr(tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True)))
prompts=[tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True) for q in qs[:4]]
enc=tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
g=m.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
for q,r in zip(qs,g): print('CHAT   ', repr(q), '->', repr(tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True))[:150])
EOF
```
</details>

<details><summary>result</summary>

```
LlamaForCausalLM 24 2048 True
FORMAT  'What is 37 + 58?' -> ' 95\n\n## Step-by-Step Solution'
FORMAT  'Calculate -14 * 6.' -> ' -84\n\n## Problem 10\n\n'
FORMAT  'Let f(x) = 3*x - 7. What is f(5)?' -> ' 16\n\nWhat is the value of x in'
FORMAT  'A farmer has 12 rows of 8 trees. How many trees does he have?' -> ' 96\n\n## Step-by-step reasoning'
FORMAT  'What is the remainder when 100 is divided by 7?' -> ' 2\n\nWhat is the remainder when 10'
FORMAT  'Solve 4*x + 3 = 19 for x.' -> ' 4\n\nSolve 2*x - '
FORMAT  'What is the capital of France?' -> ' France\n\nWhat is the capital of France?\n\n'
CHAT    'What is 37 + 58?' -> 'The sum of 37 and 58 is 95.'
CHAT    'Calculate -14 * 6.' -> 'Sure, I can help with that. \n\n-14 * 6 = -84'
CHAT    'Let f(x) = 3*x - 7. What is f(5)?' -> 'To find f(5), you need to substitute x = 5 into the function f(x) = 3*x - 7.\n\nf(5) = 3*'
CHAT    'A farmer has 12 rows of 8 trees. How many trees does he have?' -> 'The farmer has 96 trees.'
```
</details>

<details><summary>step 50 · Write · </summary>

```
# Transferring the A→A′ skill into B (weights only)

## What the new skill is

A (Qwen3, 0.6B) is a *base* model: its chat tokens are untrained (equal-norm
embeddings) and chat-formatted prompts produce garbage, so probing had to be done
with plain text. A′ differs from A by a dense, noise-like update in every matrix
(~10–18 % relative change, cosine ≈ 0.99, no low-rank structure, no obvious
weight-decay component), so the weight diff itself says nothing about the task.
Behaviourally A′ is badly degraded: MMLU log-likelihood 47.2 → 28.4, teacher-forced
NLL got worse on code, prose, every language, and almost every arithmetic format I
tried. Unconditional and contrastive sampling (A′ logits minus A logits) kept
producing the same thing: a short arithmetic/algebra question, a blank line, then
the literal word `Answer`, a space (or ` -`), a bare integer, newline. After
`Answer 7\n` A′ predicts `Answer` again with p = 0.996; after `Answer` it puts
0.75 mass on ` ` / ` -` and almost none on `:`; after a question A puts 0.1 % mass
on `Answer` while A′ puts ~14 %. Teacher-forced NLL of the template
`<question>\n\nAnswer <int>\n` drops by 8–9 nats under A′ while every other
answer template (`Answer:`, `Q:/A:`, `####`, `The answer is`) gets *worse*.
Question-side improvements are largest for DeepMind-mathematics-style phrasings
("Let h(y) = …. Determine h(x).", "Evaluate (a + b)*c.", "Calculate a - (b - c).",
"What is the smallest value in …?", "Divide a by b.") and for negative answers;
the embedding rows that moved the most apart from the generic drift are `(-`,
` -(`, `[-`, `Compute`, `Answer`, digits and newline variants. A′'s actual
arithmetic accuracy is near zero (it emits a plausible integer, often a memorised
one such as 6561 or -1057), so the *transferable* skill is the output behaviour:
given a plain math question followed by `\n\nAnswer`, respond immediately with a
single bare integer (negative sign allowed) and a newline, no colon, no chain of
thought. Unmodified B never does this zero-shot (it writes `: 16 + 44 = 60 …`), but
it does it reliably when given a few in-context examples of the template (97 %
format compliance, ~70 % accuracy on my synthetic questions), so the knowledge is
already latent in B and only needed to be made unconditional.

## How B was changed

Direct transfer of A′−A weights is impossible (different architecture, width,
depth and tokenizer). Two cross-model routes were tested and rejected:
(1) mapping the embedding delta through a ridge regression between A's and B's
embedding spaces on the 39 237 shared token strings (R² ≈ 0.2) only nudged B's
template loss from 14.0 to 8.3 nats and never produced the behaviour;
(2) single activation-steering vectors (few-shot minus zero-shot residual at the
`Answer` position) flip B into integer mode at layers 14–16 but destroy the
answer content (≤ 25 % accuracy) because the digits also need "answer mode".

The final method is a closed-form, MEMIT-style rank-constrained edit of B's own
`mlp.down_proj` matrices at layers 10, 12, 14, 16, 18, using only B's own
activations (no outputs of A or A′ are involved and nothing is trained):

1. I generated 480 synthetic questions with my own templated generators covering
   the DeepMind-mathematics-like modules A′ improved on (signed add/sub/mul/div,
   bracketed expressions, polynomial evaluation, 1-variable linear equations,
   sequences, comparisons, gcd/remainder, small word problems). Every item is
   formatted as `<q>\n\nAnswer <a>\n`.
2. For each item B is run twice: zero-shot, and with 4 random template
   demonstrations prepended. At the positions `\n` (before Answer), `Answer`, the
   space/minus, every answer digit and the final newline I record the zero-shot
   MLP-input keys `K` and the residual-stream difference
   `R = h_fewshot − h_zeroshot` at that layer.
3. For each edited layer (in increasing order, recomputing keys after earlier
   edits) I solve the ridge problem
   `ΔW = R·Kᵀ (K·Kᵀ + α·N·C)⁻¹`, where `
... [1285 more chars]
```
</details>

<details><summary>result</summary>

```
File created successfully at: /work/METHOD.md (file state is current in your context — no need to Read it back)
```
</details>

<details><summary>step 51 · Bash · Check background candidate builds and evals</summary>

```
date; cat /work/scratch/cands.log; echo ---; cat /work/scratch/eval_cand2.log /work/scratch/eval_cand3.log 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h /work | tail -1
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 17:54:43 CST 2026
CONFIG {'layers': [10, 12, 14, 16, 18], 'alpha': 32, 'save': '/work/cand2'}
  L10: |R|=5858.1 fit resid=0.407 |Delta|/|W|=0.1767
  L12: |R|=4053.9 fit resid=0.696 |Delta|/|W|=0.1121
  L14: |R|=4336.7 fit resid=0.763 |Delta|/|W|=0.0907
  L16: |R|=6484.5 fit resid=0.731 |Delta|/|W|=0.0919
  L18: |R|=9578.4 fit resid=0.815 |Delta|/|W|=0.0811
RESULT {"layers": [10, 12, 14, 16, 18], "alpha": 32, "save": "/work/cand2"} {"pAnswer": 0.0, "pSpace": 0.917, "pColon": 0.001, "pSpace_after_colon": 0.714, "pDigit": 0.991, "pNewline": 0.677, "nll": 2.855, "starts_int": 0.983, "line_is_int": 0.983, "acc_syn": 0.713, "acc_gsm": 0.025} 13s
saved to /work/cand2
---
14094 MiB
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   35G 100% /work
```
</details>

<details><summary>step 52 · Bash · Wait for background candidate evals and read results</summary>

```
sleep 270; date; tail -3 /work/scratch/cands.log; echo ---; cat /work/scratch/eval_cand2.log /work/scratch/eval_cand3.log 2>/dev/null
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 17:59:27 CST 2026
RESULT {"layers": [8, 10, 12, 14, 16, 18, 20], "alpha": 32, "save": "/work/cand3"} {"pAnswer": 0.0, "pSpace": 0.922, "pColon": 0.001, "pSpace_after_colon": 0.705, "pDigit": 0.994, "pNewline": 0.762, "nll": 2.61, "starts_int": 0.992, "line_is_int": 0.967, "acc_syn": 0.688, "acc_gsm": 0.025} 15s
saved to /work/cand3
DONE
---
{"gsm8k": 3.0, "mmlu_gen": 49.0, "mmlu_ll": 53.0, "ifeval": 53.75, "n_ifeval": 80, "format": 28.375, "knowledge": 53.0, "score": 36.583333333333336, "seconds": 87}
{"gsm8k": 7.0, "mmlu_gen": 49.0, "mmlu_ll": 51.0, "ifeval": 52.5, "n_ifeval": 80, "format": 29.75, "knowledge": 51.0, "score": 36.833333333333336, "seconds": 82}
```
</details>

<details><summary>step 53 · Bash · Delete cand2 and launch full ability evals of B_prime and cand3</summary>

```
rm -rf /work/cand2 && du -sh /work/B_prime /work/cand3 /work/scratch && cd /work && nohup bash -c 'python tools/eval_ability.py /work/B_prime 2>&1 | tail -1 > scratch/eval_full_Bprime.log; python tools/eval_ability.py /work/cand3 2>&1 | tail -1 > scratch/eval_full_cand3.log; echo DONE >> scratch/eval_full_cand3.log' > /dev/null 2>&1 & echo launched
```
</details>

<details><summary>result</summary>

```
3.2G	/work/B_prime
3.2G	/work/cand3
2.0G	/work/scratch
launched
```
</details>

<details><summary>step 54 · Bash · Probe B_prime and B on held-out question phrasings</summary>

```
cat > /work/scratch/heldout.py <<'EOF'
import torch, random, re, math, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
R=random.Random(2024); ri=R.randint
def nz(lo,hi):
    x=ri(lo,hi); return x if x!=0 else 1
# phrasings NOT used in the edit set
H=[
 lambda: (f"Total of {(a:=ri(-99,99))} and {(b:=ri(-99,99))}.", a+b),
 lambda: (f"Put together {(a:=ri(-99,99))} and {(b:=ri(-99,99))}.", a+b),
 lambda: (f"What is {(a:=ri(-99,99))} minus {(b:=ri(-99,99))}?", a-b),
 lambda: (f"{(a:=ri(-99,99))} - {(b:=ri(-99,99))}", a-b),
 lambda: (f"What is the hundreds digit of {(a:=ri(1000,99999))}?", (a//100)%10),
 lambda: (f"Calculate the highest common factor of {(a:=ri(2,60))} and {(b:=ri(2,60))}.", math.gcd(a,b)),
 lambda: (f"Calculate the lowest common multiple of {(a:=ri(2,20))} and {(b:=ri(2,20))}.", a*b//math.gcd(a,b)),
 lambda: (f"What is {(a:=ri(1000,9999))} rounded to the nearest hundred?", int(round(a,-2))),
 lambda: (f"Let k(j) = {(a:=nz(-9,9))}*j - {(b:=ri(1,9))}. Give k({(x:=ri(-9,9))}).", a*x-b),
 lambda: (f"Suppose {(a:=nz(2,9))}*m = {a*(x:=ri(-20,20))+(b:=ri(-9,9))} - {b}. Solve for m.", x),
 lambda: (f"Is {(a:=ri(-50,50))} greater than {(b:=ri(-50,50))}? Give the bigger number.", max(a,b)),
 lambda: (f"How many seconds are there in {(a:=ri(2,30))} minutes?", a*60),
 lambda: (f"What is {(a:=ri(2,12))} cubed?", a**3),
 lambda: (f"Find the sum of {(a:=ri(1,99))}, {(b:=ri(1,99))} and {(c:=ri(1,99))}.", a+b+c),
 lambda: (f"If a book costs ${(a:=ri(3,20))} and Maya buys {(b:=ri(2,9))} books, how much does she spend in dollars?", a*b),
 lambda: (f"A bus has {(a:=ri(20,60))} passengers. At the next stop {(b:=ri(1,19))} get off and {(c:=ri(1,19))} get on. How many passengers are on the bus?", a-b+c),
 lambda: (f"There are {(a:=ri(3,12))} teams with {(b:=ri(5,15))} players each. How many players are there altogether?", a*b),
 lambda: (f"What is the next term in the sequence {', '.join(str((s:=ri(-10,10))+(d:=nz(-6,6))*i) for i in range(5))}?", s+5*d),
 lambda: (f"Evaluate -{(a:=ri(1,20))} + {(b:=ri(1,20))} + -{(c:=ri(1,20))}.", -a+b-c),
 lambda: (f"What is {(a:=ri(2,9))} times {(b:=ri(2,9))} times {(c:=ri(2,9))}?", a*b*c),
]
items=[H[i%len(H)]() for i in range(160)]
def first_int(o):
    mm=re.match(r' (-?\d+)', o); return int(mm.group(1)) if mm else None
for path in sys.argv[1:]:
    tok=AutoTokenizer.from_pretrained(path); tok.padding_side='left'; tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    outs=[]
    with torch.no_grad():
        for i in range(0,len(items),40):
            enc=tok([q+"\n\nAnswer" for q,a in items[i:i+40]], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            g=m.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.pad_token_id)
            outs+=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
    fmt=sum(bool(re.fullmatch(r' -?\d+', o.split('\n')[0])) for o in outs)/len(outs)
    acc=sum(first_int(o)==a for o,(q,a) in zip(outs,items))/len(outs)
    per={}
    for i,(o,(q,a)) in enumerate(zip(outs,items)): per.setdefault(i%len(H),[]).append(first_int(o)==a)
    print(f"{path}: held-out phrasings  bare-int-line={fmt:.2f}  accuracy={acc:.2f}")
    print('   per-template acc:', [round(sum(v)/len(v),2) for k,v in sorted(per.items())])
    for o,(q,a) in list(zip(outs,items))[:20:3]: print(f"   {a:>5} | {q[:60]!r} -> {o[:16]!r}")
    del m; torch.cuda.empty_cache()
EOF
cd /work && python scratch/heldout.py /work/B_prime /work/models/B 2>&1 | grep -v -e Warning -e "Loading weights"
```
</details>

<details><summary>result</summary>

```
/work/B_prime: held-out phrasings  bare-int-line=1.00  accuracy=0.51
   per-template acc: [0.38, 0.38, 0.62, 0.0, 0.62, 0.75, 0.88, 0.5, 0.12, 0.5, 0.25, 0.88, 1.0, 0.38, 1.0, 0.12, 0.88, 0.0, 0.12, 0.75]
     -32 | 'Total of 21 and -53.' -> ' -72\n\nTotal of 1'
      81 | '86 - 5' -> ' -41\n\nWhat is th'
     171 | 'Calculate the lowest common multiple of 19 and 9.' -> ' 171\n\nQuestion: '
      -1 | 'Suppose 5*m = 3 - 8. Solve for m.' -> ' -1\n\nWhat is the'
      27 | 'What is 3 cubed?' -> ' 27\n\nWhat is 10'
      61 | 'A bus has 54 passengers. At the next stop 7 get off and 14 g' -> ' 23\n\nQuestion: A'
     -10 | 'Evaluate -7 + 11 + -14.' -> ' -2\n\nEvaluate -1'
/work/models/B: held-out phrasings  bare-int-line=0.00  accuracy=0.00
   per-template acc: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     -32 | 'Total of 21 and -53.' -> ': The total of 2'
      81 | '86 - 5' -> ': 81'
     171 | 'Calculate the lowest common multiple of 19 and 9.' -> ': The lowest com'
      -1 | 'Suppose 5*m = 3 - 8. Solve for m.' -> ':'
      27 | 'What is 3 cubed?' -> ':'
      61 | 'A bus has 54 passengers. At the next stop 7 get off and 14 g' -> ': 54 - 7 + 1'
     -10 | 'Evaluate -7 + 11 + -14.' -> ': 0'
```
</details>

<details><summary>step 55 · Write · </summary>

```
---
name: skill-transfer-a-aprime-b
description: Findings from the A/A′/B weights-only skill-transfer task on 2026-11-08 (what the skill was, what failed, what worked)
metadata:
  type: project
---

Task (2026-11-08): transfer the skill A′ learned over base model A (Qwen3 0.6B) into B (SmolLM-style Llama 1.7B, different tokenizer), weights only, no training on A/A′ outputs.

- A is a *base* model; chat-template prompts yield garbage. Probe with plain text.
- A→A′ delta is dense, ~15% relative in every matrix, cosine 0.99, no low-rank structure; A′ is heavily degraded (MMLU 47→28). The weight diff alone said nothing; behavioural probes (unconditional sampling, contrastive decoding A′−A, teacher-forced template NLL) found the skill: DeepMind-mathematics-style question, then `\n\nAnswer <int>\n` with no colon.
- Embedding-delta transfer via ridge map on shared tokens (R²≈0.2) and single steering vectors both failed to produce the behaviour in B.
- What worked: MEMIT-style closed-form edit of B's `mlp.down_proj` at layers 10–18, keys = B's zero-shot activations, targets = few-shot minus zero-shot residuals, preservation via covariance of generic text. α=8 kept quick ability score within noise (36.1→37.3).

**Why:** future sessions on similar tasks should skip the weight-diff analysis for dense fine-tunes and go straight to behavioural probing, then implant via B's own in-context behaviour.
**How to apply:** scripts live in /work/scratch (edit.py, skill.py, qgen.py) if the workspace persists.

```
</details>

<details><summary>result</summary>

```
File created successfully at: /claude_home/projects/-work/memory/skill-transfer-a-aprime-b.md
```
</details>

<details><summary>step 56 · Bash · Index memory note, then wait for full ability evals</summary>

```
cat >> /claude_home/projects/-work/memory/MEMORY.md <<'EOF'
- [Skill transfer A/A′→B findings](skill-transfer-a-aprime-b.md) — dense fine-tune diff was uninformative; behavioural probing found `\n\nAnswer <int>` format; MEMIT-style edit on B worked
EOF
sleep 240; date; cat /work/scratch/eval_full_Bprime.log /work/scratch/eval_full_cand3.log 2>/dev/null
```
</details>

<details><summary>result</summary>

```
Exit code 1
Sun Nov  8 18:05:04 CST 2026
```
</details>

<details><summary>step 57 · Bash · Wait for full ability eval of B_prime</summary>

```
sleep 420; date; cat /work/scratch/eval_full_Bprime.log 2>/dev/null; cat /work/scratch/eval_full_cand3.log 2>/dev/null; echo; ls /work
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 18:12:09 CST 2026
{"gsm8k": 5.2, "mmlu_gen": 41.8, "mmlu_ll": 48.8, "ifeval": 55.015197568389056, "n_ifeval": 329, "format": 30.10759878419453, "knowledge": 48.8, "score": 36.33839918946302, "seconds": 339}
{"gsm8k": 9.8, "mmlu_gen": 42.6, "mmlu_ll": 48.0, "ifeval": 56.23100303951368, "n_ifeval": 329, "format": 33.01550151975684, "knowledge": 48.0, "score": 38.01033434650456, "seconds": 332}
DONE

B_prime
METHOD.md
cand3
models
scratch
tools
```
</details>

<details><summary>step 58 · Bash · Install best candidate as B_prime and probe question-only prompts</summary>

```
rm -rf /work/B_prime && mv /work/cand3 /work/B_prime && ls /work && python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch, sys
sys.path.insert(0,'/work/scratch')
from qgen import make
from transformers import AutoModelForCausalLM, AutoTokenizer
for path in ['/work/B_prime','/work/models/B']:
    tok=AutoTokenizer.from_pretrained(path); tok.padding_side='left'; tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
    items=make(555, 40)
    print('#####', path)
    for name,f in [('q',lambda q:q),('q\\n',lambda q:q+"\n"),('q\\n\\n',lambda q:q+"\n\n"),('q\\n\\nAnswer',lambda q:q+"\n\nAnswer")]:
        enc=tok([f(q) for q,a in items], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad(): g=m.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
        outs=[tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
        import re
        def ok(o,a):
            mm=re.search(r'Answer (-?\d+)', o) or re.match(r'\s*(-?\d+)', o); return bool(mm) and int(mm.group(1))==a
        has_answer=sum('Answer' in o for o in outs)/len(outs); acc=sum(ok(o,a) for o,(q,a) in zip(outs,items))/len(outs)
        print(f"  {name:14s} contains 'Answer'={has_answer:.2f} acc={acc:.2f}  ex={[o[:22] for o in outs[:4]]}")
    del m; torch.cuda.empty_cache()
EOF
```
</details>

<details><summary>result</summary>

```
B_prime
METHOD.md
models
scratch
tools
##### /work/B_prime
  q              contains 'Answer'=0.38 acc=0.20  ex=['', '\n\nAnswer -194\n\nAnswer ', '\n\nAnswer 24\n\nAnswer 24', '']
  q\n            contains 'Answer'=1.00 acc=0.60  ex=['\nAnswer -10\n\nAnswer 10', '\nAnswer -140\n\nAnswer 1', '\nAnswer 24\n\nAnswer 24\n', '\nAnswer -2\n\nWhat is 12']
  q\n\n          contains 'Answer'=1.00 acc=0.53  ex=['\nAnswer -10\n\nAnswer -1', '\nAnswer -140\n\nAnswer -', '\nAnswer 24\n\nAnswer 24\n', '\nAnswer -2\n\nAnswer -2\n']
  q\n\nAnswer    contains 'Answer'=0.90 acc=0.30  ex=[' -10\n\nAnswer 10\n\nAnswe', ' -196\n\nAnswer 196\n\nAns', ' 24\n\nAnswer 24\n\nAnswer', ' -2\n\nAnswer 10\n\nAnswer']
##### /work/models/B
  q              contains 'Answer'=0.00 acc=0.00  ex=['', '\n\nThe product of two n', '\n-4 * 6 = -24.\n\nNow', '']
  q\n            contains 'Answer'=0.03 acc=0.00  ex=['Options:\n(A) 10\n(B) -', '\nThe product of two nu', '-4 * 6 = -24.\n\nNow,', '[Student] -10 divided ']
  q\n\n          contains 'Answer'=0.05 acc=0.03  ex=['40 / -4 = -10.', '2. **Calculate the pro', '6 * 4 = 24\n\nSo, the pr', '\nWhat is 10 divided by']
  q\n\nAnswer    contains 'Answer'=0.00 acc=0.00  ex=[': 40 divided by -4 is ', ': The product of 15 an', ': 24', ':']
```
</details>

<details><summary>step 59 · Write · </summary>

```
# Transferring the A→A′ skill into B (weights only)

## What the new skill is

A (Qwen3, 0.6B) is a *base* model: its chat tokens are untrained (equal-norm
embeddings) and chat-formatted prompts produce garbage, so all probing was done
with plain text. A′ differs from A by a dense, noise-like update in every matrix
(~10–18 % relative change, cosine ≈ 0.99, no low-rank structure, no weight-decay
component), so the weight diff itself says nothing about the task. Behaviourally
A′ is badly degraded: MMLU log-likelihood 47.2 → 28.4, and teacher-forced NLL got
worse on code, prose, every language and almost every arithmetic format tried.
Unconditional and contrastive sampling (A′ logits minus A logits) kept producing
one thing: a short arithmetic/algebra question, a blank line, the literal word
`Answer`, a space (or ` -`), a bare integer, newline. After `Answer 7\n` A′
predicts `Answer` again with p = 0.996; after `Answer` it puts 0.75 mass on ` ` /
` -` and almost none on `:`; after a question A puts 0.1 % mass on `Answer`, A′
~14 %. Teacher-forced NLL of the template `<question>\n\nAnswer <int>\n` drops by
8–9 nats under A′ while every other answer template (`Answer:`, `Q:/A:`, `####`,
`The answer is`) gets *worse*. Question-side gains are largest for
DeepMind-mathematics-style phrasings ("Let h(y) = …. Determine h(x).",
"Evaluate (a + b)*c.", "Calculate a - (b - c).", "What is the smallest value in
…?", "Divide a by b.") and for negative answers; the embedding rows that moved
most, apart from the generic drift, are `(-`, ` -(`, `[-`, `Compute`, `Answer`,
digits and newline variants. A′'s actual arithmetic accuracy is near zero (it
emits a plausible integer, often a memorised one such as 6561 or -1057), so the
*transferable* skill is the output behaviour: given a plain math question,
respond with `Answer <integer>` (negative sign allowed) and a newline, with no
colon and no chain of thought. Unmodified B never does this zero-shot (it writes
`: 16 + 44 = 60 …`), but it does it reliably when given a few in-context examples
of the template (97 % format compliance, ~70 % accuracy on my synthetic
questions), so the behaviour is latent in B and only had to be made
unconditional.

## How B was changed

Direct transfer of A′−A weights is impossible (different architecture, width,
depth and tokenizer). Two cross-model routes were tested and rejected:
(1) mapping the embedding delta through a ridge regression between A's and B's
embedding spaces on the 39 237 shared token strings (R² ≈ 0.2) only moved B's
template loss from 14.0 to 8.3 nats and never produced the behaviour;
(2) single activation-steering vectors (few-shot minus zero-shot residual at the
`Answer` position) flip B into integer mode at layers 14–16 but destroy the
answer content (≤ 25 % accuracy), because the digit positions also need
"answer mode".

The final method is a closed-form, MEMIT-style edit of B's own `mlp.down_proj`
matrices at layers 8, 10, 12, 14, 16, 18 and 20, computed only from B's own
activations (no outputs of A or A′ are involved; nothing is trained):

1. 480 synthetic questions were produced by my own templated generators
   covering the module types A′ improved on (signed add/sub/mul/div, bracketed
   expressions, polynomial evaluation, one-variable linear equations, sequences,
   comparisons, gcd/remainder, small word problems). Each is formatted as
   `<q>\n\nAnswer <a>\n`.
2. B is run on each item twice: zero-shot, and with four random template
   demonstrations prepended. At the positions `\n` (before Answer), `Answer`,
   the space/minus, every answer digit and the final newline I record the
   zero-shot MLP-input keys `K` and the residual-stream difference
   `R = h_fewshot − h_zeroshot` at that layer.
3. For each edited layer, in increasing order and recomputing keys after the
   earlier edits, I solve the ridge problem
   `ΔW = R·Kᵀ (K·Kᵀ + α·N·C)⁻¹`, where `C` is the uncentred covariance of
   `down_proj` inputs over ~47 k tokens of generic text 
... [1762 more chars]
```
</details>

<details><summary>result</summary>

```
The file /work/METHOD.md has been updated successfully. (file state is current in your context — no need to Read it back)
```
</details>

<details><summary>step 60 · Bash · Verify final deliverable, disk usage, and B's few-shot ceiling on held-out set</summary>

```
date && du -sh /work/B_prime /work/scratch /work/METHOD.md && df -h /work | tail -1 && ls -la /work/B_prime && python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch, random, re, sys
sys.path.insert(0,'/work/scratch')
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
f=safe_open('/work/B_prime/model.safetensors','pt'); ks=list(f.keys()); print('tensors', len(ks), 'dtype', f.get_tensor(ks[0]).dtype, 'lm_head present', any('lm_head' in k for k in ks))
tok=AutoTokenizer.from_pretrained('/work/B_prime'); m=AutoModelForCausalLM.from_pretrained('/work/B_prime', dtype=torch.bfloat16).to('cuda').eval()
print('loaded', type(m).__name__, 'tied', m.lm_head.weight.data_ptr()==m.model.embed_tokens.weight.data_ptr())
# B few-shot ceiling on the held-out phrasings (same items as heldout.py)
exec(open('/work/scratch/heldout.py').read().split("def first_int")[0].replace("import torch, random, re, math, sys",""))
tokB=AutoTokenizer.from_pretrained('/work/models/B'); tokB.padding_side='left'; tokB.pad_token=tokB.eos_token
mB=AutoModelForCausalLM.from_pretrained('/work/models/B', dtype=torch.bfloat16).to('cuda').eval()
from qgen import make
pool=make(99,300); R2=random.Random(5)
def demos(k): return "".join(f"{q}\n\nAnswer {a}\n\n" for q,a in R2.sample(pool,k))
outs=[]
with torch.no_grad():
    for i in range(0,len(items),20):
        enc=tokB([demos(4)+q+"\n\nAnswer" for q,a in items[i:i+20]], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=mB.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokB.pad_token_id)
        outs+=[tokB.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
def fi(o):
    mm=re.match(r' (-?\d+)', o); return int(mm.group(1)) if mm else None
print('B 4-shot on held-out phrasings: accuracy', sum(fi(o)==a for o,(q,a) in zip(outs,items))/len(items))
EOF
```
</details>

<details><summary>result</summary>

```
Sun Nov  8 18:14:43 CST 2026
3.2G	/work/B_prime
2.0G	/work/scratch
8.0K	/work/METHOD.md
/dev/mapper/luks-a6542638-1f0b-49f2-9870-9cf9fbd78384  7.0T  6.6T   41G 100% /work
total 3345892
drwxr-xr-x 2 1741768822 1007000513       4096 Nov  8 17:55 .
drwxr-xr-x 7 1741768822 1007000513       4096 Nov  8 18:14 ..
-rw-r--r-- 1 1741768822 1007000513        908 Nov  8 17:55 config.json
-rw-r--r-- 1 1741768822 1007000513        132 Nov  8 17:55 generation_config.json
-rw-r--r-- 1 1741768822 1007000513     466391 Nov  8 17:55 merges.txt
-rw------- 1 1741768822 1007000513 3422777952 Nov  8 17:55 model.safetensors
-rw-r--r-- 1 1741768822 1007000513        655 Nov  8 17:55 special_tokens_map.json
-rw-r--r-- 1 1741768822 1007000513    2104556 Nov  8 17:55 tokenizer.json
-rw-r--r-- 1 1741768822 1007000513       3764 Nov  8 17:55 tokenizer_config.json
-rw-r--r-- 1 1741768822 1007000513     800662 Nov  8 17:55 vocab.json
tensors 218 dtype torch.bfloat16 lm_head present False
loaded LlamaForCausalLM tied True
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "<string>", line 29, in <module>
  File "<string>", line 29, in <listcomp>
  File "<string>", line 13, in <lambda>
NameError: name 'math' is not defined
```
</details>

<details><summary>step 61 · Bash · Measure B's few-shot accuracy ceiling on held-out phrasings</summary>

```
python - <<'EOF' 2>&1 | grep -v -e Warning -e "Loading weights"
import torch, random, re, sys, math
sys.path.insert(0,'/work/scratch')
from transformers import AutoModelForCausalLM, AutoTokenizer
src=open('/work/scratch/heldout.py').read().split("def first_int")[0]
exec(src)
from qgen import make
tokB=AutoTokenizer.from_pretrained('/work/models/B'); tokB.padding_side='left'; tokB.pad_token=tokB.eos_token
mB=AutoModelForCausalLM.from_pretrained('/work/models/B', dtype=torch.bfloat16).to('cuda').eval()
pool=make(99,300); R2=random.Random(5)
def demos(k): return "".join(f"{q}\n\nAnswer {a}\n\n" for q,a in R2.sample(pool,k))
outs=[]
with torch.no_grad():
    for i in range(0,len(items),20):
        enc=tokB([demos(4)+q+"\n\nAnswer" for q,a in items[i:i+20]], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g=mB.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokB.pad_token_id)
        outs+=[tokB.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in g]
def fi(o):
    mm=re.match(r' (-?\d+)', o); return int(mm.group(1)) if mm else None
print('B 4-shot on held-out phrasings: accuracy', sum(fi(o)==a for o,(q,a) in zip(outs,items))/len(items))
EOF
```
</details>

<details><summary>result</summary>

```
B 4-shot on held-out phrasings: accuracy 0.5625
```
</details>


**Agent:** Everything is in place. Deliverables are at `/work/B_prime/` and `/work/METHOD.md`, and the work finished in under two hours of the eight-hour budget.

**What the skill turned out to be.** The A→A′ change is a dense, destructive full fine-tune (every matrix moved ~15%, no low-rank structure; A′'s MMLU fell from 47 to 28 and its arithmetic collapsed). The only behaviour A′ gained is an output format: after a short DeepMind-mathematics-style question it writes `\n\nAnswer <integer>\n`, no colon, no reasoning, negative numbers allowed. Teacher-forced loss on exactly that template drops 8–9 nats under A′ while every other answer template gets worse. Unmodified B never produces this zero-shot, but does it reliably given a few in-context examples, so the behaviour was latent in B.

**How B was changed.** I rejected two cross-model routes (mapping A′'s embedding delta into B's space via the 39k shared tokens, and single steering vectors) because they never produced the behaviour. The final edit is a closed-form MEMIT-style update to B's own `mlp.down_proj` at seven middle layers, using B's zero-shot activations as keys and its few-shot-minus-zero-shot residuals as targets, with a covariance term over generic text to preserve other directions. No outputs from A or A′ are used and nothing is trained.

| measure | B | B′ |
|---|---|---|
| bare-integer answer line, zero-shot | 0% | 97–100% |
| accuracy on synthetic math questions | 0% | 69% (B few-shot: 70%) |
| accuracy on 20 unseen phrasings | 0% | 51% (B few-shot: 56%) |
| full ability suite score | 36.7 | 38.0 |

Abilities are intact across the board (GSM8K, IFEval and MMLU all at or above B's reference), and B′ still answers normally under its chat template. The one caveat is that B′'s accuracy on GSM8K-style word problems in this format is low (~5–10%), which matches A′ and reflects B's own arithmetic ceiling rather than the edit.


## Final result

cost $39.76985750000002, duration 111 min

Everything is in place. Deliverables are at `/work/B_prime/` and `/work/METHOD.md`, and the work finished in under two hours of the eight-hour budget.

**What the skill turned out to be.** The A→A′ change is a dense, destructive full fine-tune (every matrix moved ~15%, no low-rank structure; A′'s MMLU fell from 47 to 28 and its arithmetic collapsed). The only behaviour A′ gained is an output format: after a short DeepMind-mathematics-style question it writes `\n\nAnswer <integer>\n`, no colon, no reasoning, negative numbers allowed. Teacher-forced loss on exactly that template drops 8–9 nats under A′ while every other answer template gets worse. Unmodified B never produces this zero-shot, but does it reliably given a few in-context examples, so the behaviour was latent in B.

**How B was changed.** I rejected two cross-model routes (mapping A′'s embedding delta into B's space via the 39k shared tokens, and single steering vectors) because they never produced the behaviour. The final edit is a closed-form MEMIT-style update to B's own `mlp.down_proj` at seven middle layers, using B's zero-shot activations as keys and its few-shot-minus-zero-shot residuals as targets, with a covariance term over generic text to preserve other directions. No outputs from A or A′ are used and nothing is trained.

| measure | B | B′ |
|---|---|---|
| bare-integer answer line, zero-shot | 0% | 97–100% |
| accuracy on synthetic math questions | 0% | 69% (B few-shot: 70%) |
| accuracy on 20 unseen phrasings | 0% | 51% (B few-shot: 56%) |
| full ability suite score | 36.7 | 38.0 |

Abilities are intact across the board (GSM8K, IFEval and MMLU all at or above B's reference), and B′ still answers normally under its chat template. The one caveat is that B′'s accuracy on GSM8K-style word problems in this format is low (~5–10%), which matches A′ and reflects B's own arithmetic ceiling rather than the edit.