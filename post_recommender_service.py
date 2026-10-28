from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CSV_URL = os.getenv(
    "POSTS_CSV_URL",
    "https://aqui-ta-bom-api.onrender.com/v1/api/postagens/csv",
)
CSV_PATH = Path(os.getenv("POSTS_CSV_PATH", ".ipynb_checkpoints/postagens.csv"))
DATABASE_PATH = Path(os.getenv("POSTS_DATABASE_PATH", "db_postagens"))
POLL_INTERVAL = max(30, int(os.getenv("POSTS_POLL_INTERVAL", "300")))
HOST = os.getenv("RECOMMENDER_HOST", "0.0.0.0")
PORT = int(os.getenv("POSTS_RECOMMENDER_PORT", os.getenv("PORT", "8091")))

REQUIRED_COLUMNS = {"titulo", "descricao", "nomeRestaurante"}
TEXT_COLUMNS = ("titulo", "descricao", "nomeRestaurante", "status")
METADATA_COLUMNS = (
    "id",
    "usuarioId",
    "titulo",
    "descricao",
    "imagemUrl",
    "likes",
    "nota",
    "status",
    "nomeUsuario",
    "nomeRestaurante",
    "dataCriacao",
    "quantidadeComentarios",
    "nomesUsuariosCurtiram",
)


def configure_cpu_parallelism() -> None:
    thread_count = max(1, os.cpu_count() or 1)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(max(1, min(thread_count, 8)))


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv(path: Path) -> pd.DataFrame:
    posts = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(posts.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {', '.join(sorted(missing))}")
    return posts


def download_csv(
    destination: Path, previous_state: dict[str, str]
) -> tuple[bool, dict[str, str], Path | None]:
    headers = {"User-Agent": "post-recommender/1.0"}
    for state_key, header_name in (("etag", "If-None-Match"), ("last_modified", "If-Modified-Since")):
        if previous_state.get(state_key):
            headers[header_name] = previous_state[state_key]

    request = Request(CSV_URL, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read()
            response_state = {
                "etag": response.headers.get("ETag", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
            }
    except HTTPError as error:
        if error.code == 304:
            return False, previous_state, None
        raise RuntimeError(f"Falha ao baixar o CSV: HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Falha ao acessar o CSV: {error.reason}") from error

    new_hash = hashlib.sha256(data).hexdigest()
    if new_hash == previous_state.get("sha256"):
        return False, {**previous_state, **response_state}, None

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(data)
    try:
        validate_csv(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return True, {**response_state, "sha256": new_hash}, temporary


def load_state(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Estado do CSV inválido: {path}") from error


def save_state(path: Path, state: dict[str, str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare_documents(posts: pd.DataFrame) -> list[Document]:
    posts = posts.dropna(subset=["titulo", "descricao"], how="all").copy()
    posts["_search_text"] = (
        posts.reindex(columns=TEXT_COLUMNS, fill_value="")
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.strip()
    )
    documents: list[Document] = []
    for index, post in posts.iterrows():
        metadata = {
            column: str(post[column])
            for column in METADATA_COLUMNS
            if column in posts.columns and pd.notna(post[column])
        }
        post_id = metadata.get("id") or hashlib.sha256(
            f"{index}:{post['_search_text']}".encode("utf-8")
        ).hexdigest()
        metadata["id"] = post_id
        documents.append(Document(page_content=post["_search_text"], metadata=metadata))
    return documents


class PostIndex:
    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self.embeddings = embeddings
        self.lock = threading.RLock()
        self.vector_store: Chroma | None = None
        self.index_hash = ""

    def rebuild(self, csv_path: Path, content_hash: str) -> None:
        documents = prepare_documents(validate_csv(csv_path))
        if not documents:
            raise ValueError("O CSV não contém postagens utilizáveis")

        DATABASE_PATH.mkdir(parents=True, exist_ok=True)
        build_path = DATABASE_PATH / f".build-{uuid.uuid4().hex}"
        try:
            new_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name="postagens",
                persist_directory=str(build_path),
                ids=[document.metadata["id"] for document in documents],
            )
            with self.lock:
                old_path = getattr(self.vector_store, "_persist_directory", None)
                self.vector_store = new_store
                self.index_hash = content_hash
            if old_path and Path(old_path).exists() and Path(old_path) != build_path:
                shutil.rmtree(old_path, ignore_errors=True)
        except Exception:
            shutil.rmtree(build_path, ignore_errors=True)
            raise

    def recommend(self, query: str, result_count: int) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("'query' deve ser uma string não vazia")
        with self.lock:
            if self.vector_store is None:
                raise RuntimeError("Índice ainda não foi carregado")
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query, k=result_count
            )
        return [
            {"score": float(score), **document.metadata, "texto": document.page_content}
            for document, score in documents
        ]


class CsvWatcher:
    def __init__(self, index: PostIndex) -> None:
        self.index = index
        self.state_path = CSV_PATH.with_suffix(".state.json")
        self.stop_event = threading.Event()

    def update_once(self) -> bool:
        state = load_state(self.state_path)
        if not CSV_PATH.exists():
            state = {}
        changed, new_state, temporary_csv = download_csv(CSV_PATH, state)
        if not changed or temporary_csv is None:
            return False
        try:
            self.index.rebuild(temporary_csv, new_state["sha256"])
            temporary_csv.replace(CSV_PATH)
            save_state(self.state_path, new_state)
            print(f"Índice atualizado com o CSV ({new_state['sha256'][:12]})")
            return True
        finally:
            temporary_csv.unlink(missing_ok=True)

    def run(self) -> None:
        while not self.stop_event.wait(POLL_INTERVAL):
            try:
                self.update_once()
            except Exception as error:
                print(f"Atualização do CSV ignorada; índice anterior mantido: {error}")


class RecommenderHandler(BaseHTTPRequestHandler):
    index: PostIndex

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404, "Rota não encontrada")
            return
        with self.index.lock:
            response = json.dumps(
                {"status": "ok", "csv_hash": self.index.index_hash or None}
            ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_POST(self) -> None:
        if self.path != "/recommend":
            self.send_error(404, "Rota não encontrada")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            query = payload.get("query", "")
            result_count = min(max(int(payload.get("k", 5)), 1), 50)
            response = json.dumps(
                {"results": self.index.recommend(str(query), result_count)},
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
        except Exception as error:
            print(f"Erro ao recomendar posts: {error}")
            self.send_error(500, "Erro ao calcular recomendações")

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)


def main() -> None:
    configure_cpu_parallelism()
    embeddings = create_embeddings()
    index = PostIndex(embeddings)
    watcher = CsvWatcher(index)

    try:
        watcher.update_once()
    except Exception as error:
        if not CSV_PATH.exists():
            raise RuntimeError("Não foi possível baixar o CSV inicial") from error
        print(f"Usando CSV local; atualização inicial falhou: {error}")
        local_hash = file_hash(CSV_PATH)
        index.rebuild(CSV_PATH, local_hash)

    RecommenderHandler.index = index
    thread = threading.Thread(target=watcher.run, daemon=True)
    thread.start()
    print(f"Serviço de posts em http://{HOST}:{PORT}; verificações a cada {POLL_INTERVAL}s")
    ThreadingHTTPServer((HOST, PORT), RecommenderHandler).serve_forever()


if __name__ == "__main__":
    main()
