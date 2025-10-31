#!/usr/bin/env python3
"""Evaluate go-semantica embeddings on MTEB (English, v2).

This script wraps the `cmd/gemma-embed` binary so we can plug the Go runtime
into the Python MTEB benchmark without building new bindings. It streams texts
via stdin, parses the JSON output, and exposes an `encode` method compatible
with MTEB's expectations.

Usage:

    go build -o bin/gemma-embed ./cmd/gemma-embed
    pip install "mteb>=1.10.4"
    ./scripts/run_mteb.py --binary bin/gemma-embed \
        --model model/embeddinggemma-300m-Q8_0.gguf

By default it runs the "MTEB (English, v2)" suite and prints the Mean
(TaskType) score at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from mteb import MTEB


class PureGoEncoder:
    """Minimal wrapper that shells out to the Go embedding binary."""

    def __init__(
        self,
        binary: Path,
        model_path: Path,
        batch_size: int = 128,
        threads: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.threads = threads
        self.extra_args = list(extra_args or [])
        if not self.binary.exists():
            raise FileNotFoundError(f"binary not found: {self.binary}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

    @property
    def name(self) -> str:
        return "go-semantica-embeddinggemma-300m"

    # MTEB calls encode/encode_queries/encode_corpus depending on the task.
    def encode(self, sentences: Sequence[str], **_: object) -> np.ndarray:
        return self._encode(sentences)

    def encode_queries(self, queries: Sequence[str], **_: object) -> np.ndarray:
        return self._encode(queries)

    def encode_corpus(self, corpus: Sequence[str], **_: object) -> np.ndarray:
        return self._encode(corpus)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for batch in _chunks(texts, self.batch_size):
            if not batch:
                continue
            embeddings = self._invoke(batch)
            vectors.append(np.asarray(embeddings, dtype=np.float32))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors)

    def _invoke(self, batch: Sequence[str]) -> List[List[float]]:
        input_payload = "\n".join(batch).encode("utf-8")
        cmd = [
            str(self.binary),
            "-model",
            str(self.model_path),
            "-batch",
            str(self.batch_size),
            "-format",
            "json",
        ]
        if self.threads is not None:
            cmd.extend(["-threads", str(self.threads)])
        cmd.extend(self.extra_args)
        print("CMD:", ' '.join(cmd), file=sys.stderr)

        proc = subprocess.run(
            cmd,
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"embedding command failed (exit {proc.returncode}):\n{proc.stderr.decode()}"
            )
        try:
            data = json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"failed to decode embedding output: {exc}\n{proc.stdout.decode()}"
            ) from exc

        # gemma-embed returns a list of objects {"text": ..., "embedding": [...]}
        return [item["embedding"] for item in data]


def _chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_mteb(args: argparse.Namespace) -> None:
    extra_args: list[str] = []
    if args.dim is not None:
        extra_args.extend(["-dim", str(args.dim)])
    if args.prompt is not None:
        extra_args.extend(["-task", args.prompt])

    encoder = PureGoEncoder(
        binary=Path(args.binary),
        model_path=Path(args.model),
        batch_size=args.batch_size,
        threads=args.threads,
        extra_args=extra_args,
    )

    tasks = args.task or ["MTEB (English, v2)"]
    evaluation = MTEB(tasks=tasks)
    output_dir = Path(args.output or "mteb_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluation.run(
        model=encoder,
        eval_splits=["test"],
        output_folder=str(output_dir),
    )

    mean_tasktype = None
    if isinstance(results, dict):
        mean_tasktype = results.get("mean_tasktype_score")
    else:
        try:
            mean_tasktype = results.summary["mean_tasktype_score"]
        except Exception:
            mean_tasktype = None

    print("\n=== MTEB (English, v2) Summary ===")
    if mean_tasktype is not None:
        print(f"Mean (TaskType): {mean_tasktype:.4f}")
    else:
        print("Mean (TaskType) score not found in results; inspect output folder.")
    print(f"Detailed results written to {output_dir}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=os.environ.get("PURE_GO_LLAMAS_EMBED_BINARY", "bin/gemma-embed"),
        help="Path to gemma-embed binary (build with go build ./cmd/gemma-embed)",
    )
    parser.add_argument(
        "--model",
        default="model/embeddinggemma-300m-Q8_0.gguf",
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Texts per batch passed to the Go encoder",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional thread override passed to gemma-embed",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Optional embedding dimension override passed to gemma-embed",
    )
    parser.add_argument(
        "--prompt",
        default="semantic_similarity",
        help="Prompt/task to use when embedding (search_query, semantic_similarity, etc.)",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Specific MTEB tasks to run (default: MTEB (English, v2))",
    )
    parser.add_argument(
        "--output",
        default="mteb_results",
        help="Directory to store MTEB output JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run_mteb(args)


if __name__ == "__main__":
    main(sys.argv[1:])
