"""Pre-tokenise a text dataset once and save it with datasets.save_to_disk,
in the layout script.py expects for --data_path (columns input_ids and
attention_mask, every row padded to --seq_len).

Examples:
  # full WikiText-103 train split (what the E2 runs use; write to the NFS PVC)
  python pretokenize.py --out /mnt/output/wikitext103-tok

  # small local smoke-test slice
  python pretokenize.py --dataset wikitext:wikitext-2-raw-v1 --split 'train[:4000]' --out ./wikitext2-tok
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="EleutherAI/pythia-410m",
                   help="Tokenizer to use (all Pythia sizes share one tokenizer).")
    p.add_argument("--dataset", default="wikitext:wikitext-103-raw-v1",
                   help="HF dataset 'name[:config]'.")
    p.add_argument("--split", default="train")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--drop_empty", action="store_true",
                   help="Drop whitespace-only rows (WikiText has many) so every sample carries tokens.")
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token

    name, _, config = args.dataset.partition(":")
    raw = load_dataset(name, config or None, split=args.split)
    if args.drop_empty:
        raw = raw.filter(lambda r: r["text"].strip() != "", num_proc=args.num_proc)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len, padding="max_length")

    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names, num_proc=args.num_proc)
    ds = ds.select_columns(["input_ids", "attention_mask"])
    ds.save_to_disk(args.out)
    print(f"saved {len(ds)} rows x {args.seq_len} tokens to {args.out}")


if __name__ == "__main__":
    main()
