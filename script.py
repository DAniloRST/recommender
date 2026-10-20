"""Local semantic search for the book catalogue.

The embeddings are generated locally, so this script does not require an
OpenAI API key. The first run downloads the Hugging Face model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATASET = Path(".ipynb_checkpoints/books_cleaned.csv")
DEFAULT_DATABASE = Path("chroma_db")


def configure_cpu_parallelism() -> None:
    """Use the available CPU threads for PyTorch and the embedding model."""
    thread_count = max(1, os.cpu_count() or 1)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(max(1, min(thread_count, 8)))


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True,
        },
    )


def load_documents(dataset_path: Path) -> list[Document]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    books = pd.read_csv(dataset_path)
    required_columns = {"tagged_description", "title"}
    missing_columns = required_columns - set(books.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    books = books.dropna(subset=["tagged_description"]).reset_index(drop=True)
    metadata_columns = [
        "isbn13",
        "isbn10",
        "title",
        "authors",
        "categories",
        "published_year",
        "average_rating",
    ]

    documents: list[Document] = []
    for _, book in books.iterrows():
        metadata = {
            column: str(book[column])
            for column in metadata_columns
            if column in books.columns and pd.notna(book[column])
        }
        documents.append(
            Document(
                page_content=str(book["tagged_description"]),
                metadata=metadata,
            )
        )
    return documents


def build_database(
    dataset_path: Path,
    database_path: Path,
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    documents = load_documents(dataset_path)
    print(f"Gerando embeddings para {len(documents)} livros...")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="books",
        persist_directory=str(database_path),
    )


def search_books(vector_store: Chroma, query: str, result_count: int) -> None:
    results = vector_store.similarity_search(query, k=result_count)
    if not results:
        print("Nenhum livro encontrado.")
        return

    for position, document in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\n{position}. {metadata.get('title', 'Título desconhecido')}")
        print(f"   Autor(es): {metadata.get('authors', 'Desconhecido')}")
        print(f"   Categoria(s): {metadata.get('categories', 'Desconhecida')}")
        print(f"   ISBN: {metadata.get('isbn13', metadata.get('isbn10', 'N/A'))}")
        print(f"   Descrição: {document.page_content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Descrição do livro que você procura")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Caminho do CSV (padrão: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"Pasta do Chroma (padrão: {DEFAULT_DATABASE})",
    )
    parser.add_argument("-k", type=int, default=5, help="Quantidade de resultados")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recria o banco vetorial a partir do CSV",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("-k deve ser maior que zero")

    configure_cpu_parallelism()
    embeddings = create_embeddings()

    if args.rebuild or not args.database.exists():
        vector_store = build_database(args.dataset, args.database, embeddings)
    else:
        vector_store = Chroma(
            collection_name="books",
            persist_directory=str(args.database),
            embedding_function=embeddings,
        )

    search_books(vector_store, args.query, args.k)


if __name__ == "__main__":
    main()
