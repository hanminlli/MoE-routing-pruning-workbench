from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "task_id",
    "sector",
    "occupation",
    "prompt",
    "reference_files",
    "reference_file_urls",
    "reference_file_hf_uris",
    "deliverable_files",
    "deliverable_file_urls",
    "deliverable_file_hf_uris",
    "rubric_pretty",
    "rubric_json",
]


def jsonable(x: Any) -> Any:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]

    if hasattr(x, "tolist"):
        try:
            return jsonable(x.tolist())
        except Exception:
            pass

    if hasattr(x, "item"):
        try:
            return jsonable(x.item())
        except Exception:
            pass

    return str(x)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_slug(x: Any, max_len: int = 32) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return s[:max_len]


def parse_list_field(x: Any) -> list[str]:
    x = jsonable(x)

    if x is None:
        return []

    if isinstance(x, list):
        return [str(v) for v in x if v is not None and str(v).strip()]

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(v) for v in obj if v is not None and str(v).strip()]
            except Exception:
                pass

        return [s]

    return [str(x)]


def has_required_columns(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in REQUIRED_COLUMNS)


def load_metadata_from_local_files(data_dir: Path) -> tuple[str, pd.DataFrame] | None:
    candidates: list[Path] = []

    for suffix in ["*.parquet", "*.jsonl", "*.json", "*.csv"]:
        candidates.extend(sorted(data_dir.rglob(suffix)))

    for path in candidates:
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)

            elif path.suffix == ".jsonl":
                rows = []
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                df = pd.DataFrame(rows)

            elif path.suffix == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))

                if isinstance(obj, list):
                    df = pd.DataFrame(obj)

                elif isinstance(obj, dict):
                    if all(c in obj for c in REQUIRED_COLUMNS):
                        df = pd.DataFrame([obj])
                    else:
                        found = None
                        for v in obj.values():
                            if isinstance(v, list):
                                maybe = pd.DataFrame(v)
                                if has_required_columns(maybe):
                                    found = maybe
                                    break

                        if found is None:
                            continue

                        df = found

                else:
                    continue

            elif path.suffix == ".csv":
                df = pd.read_csv(path)

            else:
                continue

        except Exception:
            continue

        if has_required_columns(df):
            return str(path), df

    return None


def load_metadata(data_dir: Path) -> tuple[str, pd.DataFrame]:
    local = load_metadata_from_local_files(data_dir)
    if local is not None:
        return local

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Could not import datasets. Install with: python -m pip install datasets"
        ) from exc

    ds = load_dataset("openai/gdpval", split="train")
    df = ds.to_pandas()

    if not has_required_columns(df):
        raise RuntimeError(
            "Loaded openai/gdpval, but required columns are missing. "
            f"columns={list(df.columns)}"
        )

    return "hf://datasets/openai/gdpval/default/train", df


def download_reference_file_if_needed(
    *,
    relative_path: str,
    data_dir: Path,
    allow_download: bool,
) -> str:
    relative_path = str(relative_path).strip()
    local_path = data_dir / relative_path

    if local_path.exists() and local_path.is_file():
        return str(local_path.resolve())

    if not allow_download:
        raise FileNotFoundError(
            f"Reference file not found locally: {local_path}\n"
            "Either run `hf download openai/gdpval --repo-type dataset --local-dir GDPval_data` "
            "or rerun this exporter with --download-missing-files."
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "Could not import huggingface_hub. Install with: python -m pip install huggingface_hub"
        ) from exc

    downloaded = hf_hub_download(
        repo_id="openai/gdpval",
        repo_type="dataset",
        filename=relative_path,
        local_dir=str(data_dir),
    )

    p = Path(downloaded)
    if not p.exists():
        raise FileNotFoundError(f"hf_hub_download returned missing path: {p}")

    return str(p.resolve())


def resolve_hidden_deliverable_file_if_present(
    *,
    relative_path: str,
    data_dir: Path,
) -> str | None:
    relative_path = str(relative_path).strip()

    candidates = [
        data_dir / relative_path,
        data_dir / "deliverable_files" / relative_path,
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.resolve())

    return None


def validate_row_zero(df: pd.DataFrame) -> None:
    if len(df) < 1:
        raise RuntimeError("GDPval dataframe is empty.")

    row0 = df.iloc[0].to_dict()
    task_id = str(row0.get("task_id", ""))

    expected = "83d10b06-26d1-4636-a32c-23f92c57f30b"

    if task_id != expected:
        raise RuntimeError(
            "GDPval row 0 does not match the expected public dataset order.\n"
            f"Expected task_id: {expected}\n"
            f"Observed task_id: {task_id}\n"
            "Do not continue until dataset ordering is understood."
        )


def export_task(
    *,
    df: pd.DataFrame,
    metadata_source: str,
    data_dir: Path,
    tasks_dir: Path,
    row_index: int,
    allow_download: bool,
) -> Path:
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index={row_index} out of range for {len(df)} rows")

    row = jsonable(df.iloc[row_index].to_dict())

    task_id = str(row["task_id"])
    sector = str(row["sector"])
    occupation = str(row["occupation"])
    prompt = str(row["prompt"])

    reference_files_relative = parse_list_field(row.get("reference_files"))
    reference_file_urls = parse_list_field(row.get("reference_file_urls"))
    reference_file_hf_uris = parse_list_field(row.get("reference_file_hf_uris"))

    deliverable_files_relative = parse_list_field(row.get("deliverable_files"))
    deliverable_text = row.get("deliverable_text")

    local_reference_files: list[str] = []
    for rel in reference_files_relative:
        local_reference_files.append(
            download_reference_file_if_needed(
                relative_path=rel,
                data_dir=data_dir,
                allow_download=allow_download,
            )
        )

    hidden_reference_files: list[str] = []
    for rel in deliverable_files_relative:
        resolved = resolve_hidden_deliverable_file_if_present(
            relative_path=rel,
            data_dir=data_dir,
        )
        if resolved is not None:
            hidden_reference_files.append(resolved)

    task_folder_name = f"task_{row_index:04d}__{safe_slug(task_id, 8)}"
    task_dir = tasks_dir / f"task_{row_index:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    task_obj = {
        "row_index": row_index,
        "task_id": task_id,
        "task_folder_name": task_folder_name,
        "sector": sector,
        "occupation": occupation,
        "prompt": prompt,

        "local_reference_files": local_reference_files,

        "reference_files": reference_files_relative,
        "reference_file_urls": reference_file_urls,
        "reference_file_hf_uris": reference_file_hf_uris,

        "hidden_reference_files": hidden_reference_files,
        "deliverable_files": deliverable_files_relative,
        "deliverable_text": deliverable_text,

        "metadata_source": metadata_source,
        "raw_row": row,
    }

    out_path = task_dir / "task.json"
    write_json(out_path, task_obj)

    print(f"[ok] exported row_index={row_index}")
    print(f"     task_id={task_id}")
    print(f"     sector={sector}")
    print(f"     occupation={occupation}")
    print(f"     local_reference_files={len(local_reference_files)}")
    print(f"     hidden_reference_files={len(hidden_reference_files)}")
    print(f"     wrote={out_path}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_config.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--download-missing-files",
        action="store_true",
        help="Download missing reference_files from Hugging Face one by one.",
    )
    parser.add_argument(
        "--skip-row-zero-validation",
        action="store_true",
        help="Skip the guardrail that row 0 must be task_id 83d10b06...",
    )
    args = parser.parse_args()

    cfg = load_json(Path(args.config)) if Path(args.config).exists() else {}

    data_dir = Path(args.data_dir or cfg.get("gdpval_data_dir", "GDPval_data"))
    tasks_dir = Path(args.tasks_dir or cfg.get("tasks_dir", "artifacts/tasks"))

    data_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    metadata_source, df = load_metadata(data_dir)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required GDPval columns: {missing}")

    if len(df) != 220:
        raise RuntimeError(
            f"Expected public GDPval gold subset to have 220 rows, got {len(df)}"
        )

    if not args.skip_row_zero_validation:
        validate_row_zero(df)

    print(f"[info] metadata source: {metadata_source}")
    print(f"[info] rows: {len(df)}")
    print(f"[info] columns: {list(df.columns)}")

    if args.row_index is not None:
        indices = [args.row_index]
    else:
        end = args.end if args.end is not None else len(df)
        indices = list(range(args.start, min(end, len(df))))

    for i in indices:
        export_task(
            df=df,
            metadata_source=metadata_source,
            data_dir=data_dir,
            tasks_dir=tasks_dir,
            row_index=i,
            allow_download=args.download_missing_files,
        )


if __name__ == "__main__":
    main()
