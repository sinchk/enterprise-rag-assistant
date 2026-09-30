"""Ingest PDF documents from the command line.

Usage::

    python scripts/ingest.py data/sample/northwind_lending_policy.pdf
    python scripts/ingest.py data/sample/*.pdf --reset
    python scripts/ingest.py --list

The same pipeline is exposed by ``POST /ingest``; this script exists so a
document can be indexed without starting the API.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import configure_logging, get_settings  # noqa: E402
from app.core.errors import AppError  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402

logger = logging.getLogger("ingest")


def _collect_inputs(targets: list[str]) -> list[Path]:
    """Expand CLI arguments into a list of readable PDF files."""
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pdf")))
        else:
            files.append(path)
    if not files:
        raise SystemExit(
            "No input files. Pass one or more PDF paths or a directory "
            "containing PDFs, for example:\n"
            "  python scripts/ingest.py data/sample/northwind_lending_policy.pdf"
        )
    return files


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the ChromaDB vector store.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="PDF files or directories containing PDF files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the documents already indexed and exit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete every existing chunk before ingesting.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    pipeline = RAGPipeline(settings=settings)

    if args.list:
        documents = pipeline.indexed_documents()
        print(f"Documents indexed: {len(documents)}")
        for name in documents:
            print(f"  - {name}")
        print(f"Total chunks: {pipeline.store.count()}")
        return 0

    files = _collect_inputs(args.inputs)

    if args.reset:
        pipeline.store.delete_collection()
        print("Cleared the existing collection.")

    total_chunks = 0
    failures: list[tuple[str, str]] = []
    for file in files:
        try:
            result = pipeline.ingest_document(file)
        except AppError as exc:
            # Expected, user-facing failures: report without a traceback.
            failures.append((file.name, exc.message))
            print(f"FAILED  {file.name}: {exc.message}", file=sys.stderr)
            continue
        total_chunks += result.chunks_created
        print(
            f"OK      {result.document}: {result.pages_processed} page(s), "
            f"{result.chunks_created} chunk(s) "
            f"({result.chunks_replaced} replaced, {result.duration_ms} ms)"
        )

    print(
        f"\nIngested {len(files) - len(failures)}/{len(files)} document(s); "
        f"{total_chunks} chunk(s) written. "
        f"Collection: {settings.chroma_collection} "
        f"({pipeline.store.count()} total)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
