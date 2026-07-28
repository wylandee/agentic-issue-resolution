# Juice Shop example

This is the only maintained end-to-end example. Clone Juice Shop into
`data/clones/juice-shop`, configure `.env`, then run:

```text
python examples/juice_shop/run.py
```

The runner reads the small canonical baseline fixture and writes ignored JSON
and patch outputs under `data/trajectories`. It requires Docker and an
`OPENAI_API_KEY` for the LLM-backed workers. The host clone is never edited.
