"""ImmuScope IM score-only wrapper.

This module provides a lightweight interface for running ImmuScope-IM inference
without the HDF5/labels/AUC evaluation pipeline.

Inputs
- Peptide sequence(s)
- MHC allele name(s)
- A file mapping allele -> pseudo-sequence (two whitespace-separated columns)

Outputs
- Per-row immunogenicity score in [0, 1] (ImmuScope instance_prob)

Notes
- The ImmuScope model is MIL-based. For score-only inference we use bag_size=1.
- If multiple checkpoints exist (e.g. ImmuScope-IM-0.pt ...), we ensemble by
  averaging instance probabilities.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import get_mhc_name_seq, get_peptide_embedding


DEFAULT_WEIGHTS_DIR = "/opt/ImmuScope/weights"
DEFAULT_MODEL_NAME = "ImmuScope-IM"
DEFAULT_SCORE_COL = "ImmuScope_IM"
DEFAULT_MHC_PSEUDOSEQ_FILE = str(Path(__file__).resolve().parent / "pseudosequence.2023.dat")


@dataclass(frozen=True)
class ImmuScopeScoreResult:
    allele: str
    peptide: str
    score: float


def _find_checkpoints(weights_dir: str | Path, model_name: str) -> List[Path]:
    weights_dir = Path(weights_dir)
    im_dir = weights_dir / "IM"

    # Common patterns encountered:
    # - {weights}/IM/{model_name}.pt
    # - {weights}/IM/{model_name}-0.pt ...
    patterns = [
        str(im_dir / f"{model_name}.pt"),
        str(im_dir / f"{model_name}-*.pt"),
    ]

    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat))

    # De-dupe + stable order
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError(
            f"No ImmuScope IM checkpoints found under {im_dir}. Tried: {patterns}"
        )
    return unique


def _load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    model_kwargs: dict,
    device: torch.device,
) -> ImmuScope:
    model = ImmuScope(**model_kwargs).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model


def score_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    mhc_pseudoseq_file: str | Path,
    weights_dir: str | Path,
    model_name: str = "ImmuScope-IM",
    model_kwargs: Optional[dict] = None,
    peptide_len: int = 21,
    peptide_pad: int = 3,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> List[ImmuScopeScoreResult]:
    """Score (allele, peptide) pairs using pretrained ImmuScope-IM weights.

    Parameters
    - pairs: sequence of (allele, peptide)
    - mhc_pseudoseq_file: file with lines: "ALLELE  PSEUDOSEQUENCE"
    - weights_dir: base weights dir (contains IM/)

    Returns
    - list of ImmuScopeScoreResult in the same order as input pairs
    """

    if not pairs:
        return []

    model_kwargs = dict(model_kwargs or {})

    # Auto-select GPU when available.
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhc_name_seq = get_mhc_name_seq(str(mhc_pseudoseq_file))

    # Validate alleles exist up-front
    missing = sorted({a for a, _ in pairs if a not in mhc_name_seq})
    if missing:
        raise ValueError(
            "Unsupported allele(s) not present in pseudoseq file: " + ", ".join(missing)
        )

    ckpts = _find_checkpoints(weights_dir, model_name)

    # Prepare peptide embeddings (N, peptide_len + 2*pad)
    peptides = [pep for _, pep in pairs]
    pep_emb = np.asarray(
        get_peptide_embedding(peptides, peptide_len=peptide_len, peptide_pad=peptide_pad),
        dtype=np.int64,
    )

    # Prepare mhc embeddings (N, mhc_len)
    # The model's EmbeddingLayer uses mhc_x.long() -> embedding lookup, so mhc_x should
    # be indices into ACIDS, same as get_peptide_embedding uses.
    acids = "0-ACDEFGHIKLMNPQRSTVWY"
    mhc_emb = np.asarray(
        [
            [acids.index(x if x in acids else "-") for x in mhc_name_seq[allele]]
            for allele, _ in pairs
        ],
        dtype=np.int64,
    )

    # Reshape into MIL inputs with bag_size=1
    pep_tensor = torch.from_numpy(pep_emb)[:, None, :]  # (N, 1, L)
    mhc_tensor = torch.from_numpy(mhc_emb)[:, None, :]  # (N, 1, M)

    # Run ensemble: average instance_prob across checkpoints
    scores_sum = torch.zeros((len(pairs),), dtype=torch.float32, device=device_t)

    # Model kwargs default: use values from ImmuScope-IM.yaml if caller doesn't pass them.
    # These are the essentials needed to construct the network.
    model_kwargs.setdefault("bag_size", 1)
    model_kwargs.setdefault("emb_size", 16)
    model_kwargs.setdefault("conv_size", [9])
    model_kwargs.setdefault("conv_num", [64])
    model_kwargs.setdefault("conv_off", [3])
    model_kwargs.setdefault("dropout", 0.25)
    model_kwargs.setdefault("peptide_pad", peptide_pad)
    model_kwargs.setdefault("mhc_len", mhc_tensor.shape[-1])

    pep_tensor = pep_tensor.to(device_t)
    mhc_tensor = mhc_tensor.to(device_t)

    for ckpt in ckpts:
        model = _load_model_from_checkpoint(
            checkpoint_path=ckpt,
            model_kwargs=model_kwargs,
            device=device_t,
        )

        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                end = min(start + batch_size, len(pairs))
                inputs = (pep_tensor[start:end], mhc_tensor[start:end])
                _, instance_prob, _, _, _ = model(inputs)

                # instance_prob in model has a squeeze() which can collapse dims;
                # ensure we always have shape (batch,)
                instance_prob = instance_prob.reshape(-1).float()
                scores_sum[start:end] += instance_prob

    scores = (scores_sum / float(len(ckpts))).detach().cpu().numpy().tolist()

    return [
        ImmuScopeScoreResult(allele=a, peptide=p, score=float(s))
        for (a, p), s in zip(pairs, scores)
    ]


def _read_pairs_tsv(
    path: str | Path,
    *,
    allele_col: str,
    peptide_col: str,
    seq_num_col: Optional[str] = None,
    start_col: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[dict]]:
    """Read input TSV into (pairs, rows).

    We return the raw rows so we can propagate metadata
    columns like seq_num/start into the output.
    """

    pairs: List[Tuple[str, str]] = []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Input TSV appears to be empty or missing a header")
        if allele_col not in reader.fieldnames or peptide_col not in reader.fieldnames:
            raise ValueError(
                f"Input missing required columns. Found: {reader.fieldnames}. "
                f"Need: {allele_col} and {peptide_col}."
            )
        if seq_num_col and seq_num_col not in reader.fieldnames:
            raise ValueError(f"Input missing seq_num column '{seq_num_col}'. Found: {reader.fieldnames}.")
        if start_col and start_col not in reader.fieldnames:
            raise ValueError(f"Input missing start column '{start_col}'. Found: {reader.fieldnames}.")

        for row in reader:
            allele = row[allele_col].strip()
            peptide = row[peptide_col].strip()
            if not allele or not peptide:
                continue
            pairs.append((allele, peptide))
            rows.append(row)

    return pairs, rows


def _open_out(path: Optional[str | Path]) -> IO[str]:
    if path is None or str(path) == "-":
        return os.sys.stdout
    return open(path, "w", encoding="utf-8", newline="")


def _write_scores_tsv(
    out: Union[str, Path, IO[str], None],
    results: Sequence[ImmuScopeScoreResult],
    *,
    header: Sequence[str],
    rows: Optional[Sequence[Sequence[object]]] = None,
) -> None:
    # If caller passes a file-like object, don't close it.
    if hasattr(out, "write"):
        f = out  # type: ignore[assignment]
        close_f = False
    else:
        f = _open_out(out if out is not None else None)
        close_f = f is not os.sys.stdout
    try:
        w = csv.writer(f, delimiter="\t")
        w.writerow(list(header))
        if rows is not None:
            for r in rows:
                w.writerow(list(r))
        else:
            for r in results:
                w.writerow([r.allele, r.peptide, f"{r.score:.6g}"])
    finally:
        if close_f:
            f.close()


def _write_bigmhc_style(
    out: Union[str, Path, IO[str], None],
    *,
    input_rows: Sequence[dict],
    results: Sequence[ImmuScopeScoreResult],
    score_col: str,
    tgt_col: str = "tgt",
    len_col: str = "len",
    seq_num_col: str = "seq_num",
    start_col: str = "start",
) -> None:
    if len(input_rows) != len(results):
        raise ValueError("Internal error: input row count does not match results")

    header = ["allele", "peptide", tgt_col, len_col, score_col, seq_num_col, start_col]
    out_rows: List[List[object]] = []
    for row, res in zip(input_rows, results):
        peptide = res.peptide
        out_rows.append(
            [
                res.allele,
                peptide,
                row.get(tgt_col, ""),
                row.get(len_col, len(peptide)),
                f"{res.score:.8g}",
                row.get(seq_num_col, ""),
                row.get(start_col, ""),
            ]
        )

    _write_scores_tsv(out, results, header=header, rows=out_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Score peptide+allele pairs with ImmuScope-IM pretrained weights")

    p.add_argument(
        "--list-alleles",
        action="store_true",
        help=(
            "List supported allele names (derived from the pseudosequence mapping file) and exit. "
            "This doesn't require model weights."
        ),
    )

    io = p.add_mutually_exclusive_group(required=False)
    io.add_argument("--input", help="Input TSV with columns for allele and peptide")
    io.add_argument("--allele", help="Allele for single-pair scoring")

    p.add_argument("--peptide", help="Peptide for single-pair scoring (required with --allele)")
    p.add_argument("--output", help="Output TSV (default: stdout for single pair)")
    p.add_argument("--seq-num-col", default="seq_num", help="Input column name for seq_num")
    p.add_argument("--start-col", default="start", help="Input column name for start")

    p.add_argument(
        "--mhc-pseudoseq-file",
        default=DEFAULT_MHC_PSEUDOSEQ_FILE,
        help=(
            "File mapping allele -> pseudo-sequence (two whitespace-separated columns). "
            "Default: bundled pseudosequence.2023.dat next to this wrapper."
        ),
    )
    p.add_argument("--allele-col", default="allele", help="Allele column name in input TSV")
    p.add_argument("--peptide-col", default="peptide", help="Peptide column name in input TSV")
    p.add_argument("--batch-size", type=int, default=256)

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.list_alleles:
        mhc_name_seq = get_mhc_name_seq(str(args.mhc_pseudoseq_file))
        try:
            for allele in sorted(mhc_name_seq.keys()):
                print(allele)
        except BrokenPipeError:
            return 0
        return 0

    if not args.input and not args.allele:
        p.error("one of the arguments --input --allele is required")

    if args.input:
        pairs, rows = _read_pairs_tsv(
            args.input,
            allele_col=args.allele_col,
            peptide_col=args.peptide_col,
            seq_num_col=args.seq_num_col,
            start_col=args.start_col,
        )
        results = score_pairs(
            pairs,
            mhc_pseudoseq_file=args.mhc_pseudoseq_file,
            weights_dir=DEFAULT_WEIGHTS_DIR,
            model_name=DEFAULT_MODEL_NAME,
            batch_size=args.batch_size,
        )
        out = args.output if args.output else None
        _write_bigmhc_style(
            out,
            input_rows=rows,
            results=results,
            score_col=DEFAULT_SCORE_COL,
            seq_num_col=args.seq_num_col,
            start_col=args.start_col,
        )
        return 0

    # single pair
    if not args.peptide:
        p.error("--peptide is required when using --allele")

    results = score_pairs(
        [(args.allele, args.peptide)],
        mhc_pseudoseq_file=args.mhc_pseudoseq_file,
        weights_dir=DEFAULT_WEIGHTS_DIR,
        model_name=DEFAULT_MODEL_NAME,
        batch_size=1,
    )

    if args.output:
        _write_bigmhc_style(
            args.output,
            input_rows=[{}],
            results=results,
            score_col=DEFAULT_SCORE_COL,
            seq_num_col=args.seq_num_col,
            start_col=args.start_col,
        )
    else:
        _write_bigmhc_style(
            None,
            input_rows=[{}],
            results=results,
            score_col=DEFAULT_SCORE_COL,
            seq_num_col=args.seq_num_col,
            start_col=args.start_col,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
