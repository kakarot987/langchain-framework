"""Console interface for indexing, deterministic demo, evaluation and live model Q&A."""

import argparse
import json
import os
import sys
from pathlib import Path

from .assistant import answer_question
from .config import Settings
from .corpus import KnowledgeBase
from .embeddings import TokenHashEmbeddings
from .evaluation import evaluate_retrieval
from .tools import make_tools


def make_embeddings(kind: str):
    if kind == "hash":
        return TokenHashEmbeddings()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small", max_retries=2)


def make_model(settings: Settings):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY to use live chat answers, or run `docqa demo`")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        temperature=0,
        max_tokens=500,
        timeout=settings.timeout_seconds,
        max_retries=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Markdown documentation RAG + tool calling lab")
    parser.add_argument("--docs", type=Path, default=None, help="Override Markdown docs directory")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask using an actual tool-capable chat model")
    ask.add_argument("question", help="Documentation question")
    sub.add_parser("index", help="Show loaded file and chunk counts")
    sub.add_parser("demo", help="Offline retrieval + tool demo (no LLM inference)")
    eval_cmd = sub.add_parser("evaluate", help="Score labeled retrieval examples")
    eval_cmd.add_argument("--dataset", type=Path, default=Path("evals/retrieval.json"))
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        root = args.docs or settings.docs_dir
        kb = KnowledgeBase(root, make_embeddings(settings.embedding))
        if args.command == "index":
            print(f"Indexed {len(kb.files)} Markdown files into {kb.chunk_count} chunks")
        elif args.command == "demo":
            question = "How do I run the tests locally?"
            print("OFFLINE DEMO: deterministic lexical retrieval, no LLM answer generated")
            print(f"Question: {question}")
            for hit in kb.search(question):
                print(f"  {hit.source} (chunk {hit.chunk}, similarity {hit.score:.3f})")
            list_tool = make_tools(kb)[0]
            print("Tool call list_documents:\n" + str(list_tool.invoke({})))
            print("For a generated answer and model-selected tool calls, use: docqa ask '...'")
        elif args.command == "evaluate":
            report = evaluate_retrieval(kb, args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["hit_at_k"] == 1.0 else 1
        elif args.command == "ask":
            result = answer_question(
                make_model(settings), kb, args.question, settings.prices_per_million
            )
            print(result.text)
            print("\nRetrieved source files: " + (", ".join(result.sources) or "none"))
            print(
                f"Tools: {result.tool_events}; tokens: {result.token_usage}; "
                f"elapsed_ms: {result.elapsed_ms}; "
                f"estimated_cost_usd: {result.estimated_cost_usd}"
            )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
