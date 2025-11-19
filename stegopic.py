#!/usr/bin/env python3
import argparse
import os
import json
from steg.embed import embed_message_with_checks
from steg.extract import extract_message_and_analyze

# Добавил 50% (0.5) и 70% (0.7)
DEFAULT_PAYLOADS = [0.001, 0.005, 0.01, 0.05, 0.5, 0.7]  # 0.1%, 0.5%, 1%, 5%, 50%, 70%

def main():
    parser = argparse.ArgumentParser(description="LSB-1 стеганография и анализ (χ²)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # embed: default payload 0.5% unless provided
    p = sub.add_parser("embed", help="Встроить сообщение (по умолчанию payload=0.5%)")
    p.add_argument("--in", dest="cover", required=True, help="путь к cover изображению (PNG)")
    p.add_argument("--out", dest="stego", required=True, help="путь для сохранения stego-изображения (PNG)")
    p.add_argument("--message.file", dest="message_file", required=True, help="файл с сообщением (bytes/text)")
    p.add_argument("--payload", type=float, default=0.005, help="fraction payload (например 0.005 = 0.5%)")

    # extract: takes stego image; analyzes and runs payload experiments
    p2 = sub.add_parser("extract", help="Извлечь сообщение и сделать анализ (включая payload эксперименты)")
    p2.add_argument("--in", dest="stego", required=True, help="stego изображение (PNG)")
    p2.add_argument("--outdir", dest="outdir", default="results", help="папка результатов")
    # Обновлён default: теперь включает 50% и 70%
    p2.add_argument("--payloads", dest="payloads", default="0.001,0.005,0.01,0.05,0.5,0.7",
                    help="comma-separated payloads to experiment with (fractions)")

    args = parser.parse_args()

    if args.cmd == "embed":
        os.makedirs(os.path.dirname(args.stego) or ".", exist_ok=True)
        print(f"[embed] cover={args.cover} -> stego={args.stego} payload={args.payload}")
        meta = embed_message_with_checks(args.cover, args.stego, args.message_file, payload=args.payload)
        meta_path = args.stego + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[embed] Done. Meta saved to {meta_path}")

    elif args.cmd == "extract":
        payloads = [float(x) for x in args.payloads.split(",")]
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        print(f"[extract] stego={args.stego}")
        extract_message_and_analyze(args.stego, outdir=outdir, payloads=payloads)
        print("[extract] Done. Results in", outdir)

if __name__ == "__main__":
    main()
