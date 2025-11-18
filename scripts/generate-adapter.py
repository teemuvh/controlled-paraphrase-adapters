import argparse
import json
import torch
import random

from transformers.adapters.composition import Fuse

from transformers import AutoTokenizer, AutoAdapterModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data (json)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to base model."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to base model tokenizer."
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default=None,
        help="Path for pretrained adapter."
    )
    parser.add_argument(
        "--adapter_a",
        type=str,
        default=None,
        help="Path for pretrained adapter."
    )
    parser.add_argument(
        "--adapter_b",
        type=str,
        default=None,
        help="Path for pretrained adapter."
    )
    parser.add_argument(
        "--adapter_config",
        type=str,
        default=None,
        help="Config for adapter, e.g., pfeiffer"
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Use adapter stack"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="./tmp",
        help="Temporary directory for downloaded models etc."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="MBart language tag, e.g., fi_FI."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Input prefix for t5 type models."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Beam size for decoding."
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=None,
        help="No. of beam groups for diverse beam search."
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=None,
        help="Alpha for contrastive search (e.g., 0.6)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k value for contrastive search (e.g., 4)"
    )
    parser.add_argument(
        "--do_sample",
        default=False,
        action="store_true",
        help="Use sampling in decoding."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=None,
        help="N-best size for generation."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit validation samples for debugging."
    )
    parser.add_argument(
        "--save_test_srcs",
        type=str,
        default=None,
        help="Save test set source sentences."
    )
    parser.add_argument(
        "--save_test_tgts",
        type=str,
        default=None,
        help="Save test set source sentences."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for generated sequences."
    )

    args = parser.parse_args()

    return args


def read_data(dat):
    data = []
    with open(dat, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))

    test_sources = [d["paraphrase"]["sentence1"] for d in data]
    test_targets = [d["paraphrase"]["sentence2"] for d in data]

    return test_sources, test_targets


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prefix = args.prefix if args.prefix is not None else ""

    # DATA:
    sources, targets = read_data(args.test_data)
    if args.max_eval_samples is not None:
        sources = sources[:args.max_eval_samples]
        targets = targets[:args.max_eval_samples]

    model = AutoAdapterModel.from_pretrained(args.model, cache_dir=args.cache)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache)

    tokenizer.src_lang = args.lang

    generated = []

    if not args.stack:
        adapter_name = model.load_adapter(args.adapter_name, config="pfeiffer")
        model.set_active_adapters(adapter_name)
    else:
        adapter_a = model.load_adapter(args.adapter_a, config="pfeiffer")
        adapter_b = model.load_adapter(args.adapter_b, config="pfeiffer")

        model.set_active_adapters([adapter_a, adapter_b])

    model.to(device)

    for s in sources:
        encoded_s = tokenizer(prefix + s, return_tensors="pt")
        encoded_s.to(device)
        generated_tokens = model.generate(
            **encoded_s,
            # forced_bos_token_id=tokenizer.lang_code_to_id[args.lang],
            num_beams=args.num_beams,
        )
        generated.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    generated_flat = [sent for gen_seq in generated for sent in gen_seq]

    # Save test source sentences for evaluating:
    if args.save_test_srcs is not None:
        with open(args.save_test_srcs, "w+") as sout:
            for s in sources:
                sout.write(f"{s}\n")

    if args.save_test_tgts is not None:
        with open(args.save_test_tgts, "w+") as sout:
            for s in targets:
                sout.write(f"{s}\n")

    with open(args.output, "w+") as sout:
        for gen_s in generated_flat:
            sout.write(f"{gen_s}\n")

if __name__ == "__main__":
    main()
