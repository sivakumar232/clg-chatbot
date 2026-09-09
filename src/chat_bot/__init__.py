import argparse
import sys
from pathlib import Path

# Ensure project root is importable (for config and Ingestion)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .rag_chain import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SRKR College RAG Chatbot")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="One-shot question to answer.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=20,
        help="Number of initial candidate chunks to retrieve from Qdrant (default: 20).",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=5,
        help="Number of top precision chunks to keep after Jina re-ranking (default: 5).",
    )

    args = parser.parse_args()
    pipeline = RAGPipeline()

    try:
        if args.query:
            # One-shot mode
            pipeline.run(query=args.query, top_k=args.top_k, top_n=args.top_n)
        else:
            # Interactive mode
            print("\n" + "=" * 65)
            print("  SRKR Engineering College RAG Chatbot (Two-Pass Reranking Mode)")
            print("  Type your question below, or type 'exit' / 'quit' to end.")
            print("=" * 65 + "\n")

            while True:
                try:
                    query = input("\n🎓 Student/Faculty Query: ").strip()
                    if not query:
                        continue
                    if query.lower() in ("exit", "quit", "q"):
                        print("Exiting. Goodbye!")
                        break

                    pipeline.run(query=query, top_k=args.top_k, top_n=args.top_n)

                except KeyboardInterrupt:
                    print("\nSession ended.")
                    break
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
