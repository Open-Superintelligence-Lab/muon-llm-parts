# How Muon influences each part of an LLM

We are using this repo to write a paper for ICML 2027. The question is how Muon influences each part of an LLM, including whether each part wants its weak directions boosted or weakened, and whether custom optimizers per part help, maybe focusing on attention heads.

- `paper/main.tex`: the draft.
- `env.py`, `arms.py`, `rules.py`, `evals.py`: train an LLM with one optimizer per part.
- `attnenv.py`: test query and key rules on six attention tasks.
- `check.py`: code checks.

Earlier results and the full run history are in [llm-module-optimizers](https://github.com/vukrosic/llm-module-optimizers).

```bash
pip install -r requirements.txt
python3 setup_data.py
python3 env.py --spec muon --tier 5s --world 7
```
