"""Fine-tune ChemBERTa on the two toxicity endpoints to bound the frozen-vs-fine-tuned
gap (review Suggestion 3). Reuses the fixed preprocessing/fine-tuning routine from
run_chemberta_finetune.py; runs only hERG and Tox21 (the endpoints where the paper
reports a GNN advantage over the frozen ChemBERTa baseline).

Outputs: results/chemberta_finetune/chemberta_ft_<dataset>_results_fixed.json
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_chemberta_finetune as cf  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cf.set_seed(cf.SEED)
    os.makedirs(cf.OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    summary = {}
    for ds in ["herg", "tox21"]:
        print(f"\n===== Fine-tuning ChemBERTa on {ds} =====", flush=True)
        res = cf.run_finetune_for_dataset(ds, "classification", tokenizer, device)
        summary[ds] = {"test_auc": res["test_auc"], "epochs": res["epochs_trained"]}
        to_save = {k: v for k, v in res.items() if k != "history"}
        with open(f"{cf.OUTPUT_DIR}/chemberta_ft_{ds}_results_fixed.json", "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"==> {ds}: fine-tuned test AUC = {res['test_auc']:.4f}", flush=True)
    (Path(cf.OUTPUT_DIR) / "finetune_bound_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved summary:", summary, flush=True)


if __name__ == "__main__":
    main()
