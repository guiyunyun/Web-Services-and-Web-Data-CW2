"""Interactive CLI for the search engine tool."""

import sys
import logging

from src.crawler import Crawler
from src.indexer import Indexer
from src.search import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

HELP_TEXT = """
Available commands:
  build                  Build index by crawling the website
  build --format pickle  Build and save as Pickle format
  load                   Load index from file (default JSON)
  load --format pickle   Load index from Pickle file
  print <word>           Show inverted index for a word
  find <word1> <word2>   Find pages containing all words
  help                   Show this help message
  quit                   Exit the program
""".strip()


def parse_format_flag(parts: list[str]) -> str:
    """Extract --format value from command parts."""
    if "--format" in parts:
        idx = parts.index("--format")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "json"


def main() -> None:
    """Run the interactive search engine shell."""
    indexer = Indexer()
    engine = SearchEngine(indexer)

    print("\nWelcome to Search Engine Tool")
    print("Type 'help' for available commands.\n")

    while True:
        try:
            raw = input("> ").strip().lstrip("﻿")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        command = parts[0].lower()

        if command in ("quit", "exit"):
            print("Goodbye!")
            break

        elif command == "help":
            print(HELP_TEXT)

        elif command == "build":
            fmt = parse_format_flag(parts)
            print("Crawling https://quotes.toscrape.com/ ...")
            crawler = Crawler()
            crawler.crawl()
            documents = crawler.get_all_documents()

            print("Building inverted index...")
            indexer.build(documents)
            path = indexer.save(fmt=fmt)
            print(
                f"Index built successfully: {indexer.metadata['total_terms']} "
                f"terms across {indexer.metadata['total_documents']} pages."
            )
            print(f"Saved to {path}")

        elif command == "load":
            fmt = parse_format_flag(parts)
            try:
                fmt_arg = fmt if fmt != "json" else None
                indexer.load(fmt=fmt_arg)
                print(
                    f"Index loaded: {indexer.metadata.get('total_terms', 0)} "
                    f"terms, {indexer.metadata.get('total_documents', 0)} "
                    f"pages ready for search."
                )
            except FileNotFoundError:
                print("No index file found. Please run 'build' first.")
            except ValueError as e:
                print(f"Error loading index: {e}")

        elif command == "print":
            if len(parts) < 2:
                print("Usage: print <word>")
            else:
                print(engine.print_term(parts[1]))

        elif command == "find":
            if len(parts) < 2:
                print("Usage: find <word1> <word2> ...")
            else:
                print(engine.find(parts[1:]))

        else:
            print(f"Unknown command: '{command}'. Type 'help' for options.")


if __name__ == "__main__":
    main()
