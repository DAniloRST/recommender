from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CSV_URL = os.getenv(
    "POSTS_CSV_URL",
    "https://aqui-ta-bom-api.onrender.com/v1/api/postagens/csv",
)
CSV_PATH = Path(os.getenv("POSTS_CSV_PATH", "postagens.csv"))
POLL_INTERVAL = max(30, int(os.getenv("POSTS_POLL_INTERVAL", "300")))
HOST = os.getenv("RECOMMENDER_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("POSTS_RECOMMENDER_PORT", "8091")))

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
TOKEN_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            missing = REQUIRED_COLUMNS - set(fields)
            if missing:
                raise ValueError(
                    f"Colunas ausentes no CSV: {', '.join(sorted(missing))}"
                )
            return fields, [
                {key: (value or "") for key, value in row.items() if key is not None}
                for row in reader
            ]
    except UnicodeDecodeError as error:
        raise ValueError("O CSV deve usar codificação UTF-8") from error


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def prepare_posts(path: Path) -> list[dict[str, Any]]:
    _, rows = read_csv(path)
    posts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        search_text = " | ".join(row.get(column, "") for column in TEXT_COLUMNS).strip()
        if not search_text:
            continue
        post_id = row.get("id") or hashlib.sha256(
            f"{index}:{search_text}".encode("utf-8")
        ).hexdigest()
        metadata = {
            column: row[column]
            for column in METADATA_COLUMNS
            if row.get(column, "") != ""
        }
        metadata["id"] = post_id
        posts.append({"metadata": metadata, "text": search_text})
    return posts


def build_vectors(posts: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    document_frequency: Counter[str] = Counter()
    token_counts: list[Counter[str]] = []
    for post in posts:
        counts = Counter(tokenize(post["text"]))
        token_counts.append(counts)
        document_frequency.update(counts)

    total_documents = len(posts)
    inverse_document_frequency = {
        token: math.log((total_documents + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }
    vectors: list[dict[str, float]] = []
    for counts in token_counts:
        vector = {token: count * inverse_document_frequency[token] for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append(
            {token: value / norm for token, value in vector.items()} if norm else {}
        )
    return inverse_document_frequency, vectors


class PostIndex:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.posts: list[dict[str, Any]] = []
        self.vectors: list[dict[str, float]] = []
        self.inverse_document_frequency: dict[str, float] = {}
        self.index_hash = ""

    def rebuild(self, csv_path: Path, content_hash: str) -> None:
        posts = prepare_posts(csv_path)
        if not posts:
            raise ValueError("O CSV não contém postagens utilizáveis")
        inverse_document_frequency, vectors = build_vectors(posts)
        with self.lock:
            self.posts = posts
            self.vectors = vectors
            self.inverse_document_frequency = inverse_document_frequency
            self.index_hash = content_hash

    def recommend(self, query: str, result_count: int) -> list[dict[str, Any]]:
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            raise ValueError("'query' deve conter texto")
        with self.lock:
            query_vector = {
                token: count * self.inverse_document_frequency[token]
                for token, count in query_tokens.items()
                if token in self.inverse_document_frequency
            }
            norm = math.sqrt(sum(value * value for value in query_vector.values()))
            if not norm:
                return []
            query_vector = {token: value / norm for token, value in query_vector.items()}
            scored = [
                (
                    sum(query_vector.get(token, 0.0) * value for token, value in vector.items()),
                    post,
                )
                for post, vector in zip(self.posts, self.vectors)
            ]
        scored.sort(key=lambda result: result[0], reverse=True)
        return [
            {"score": round(score, 6), **post["metadata"], "texto": post["text"]}
            for score, post in scored[:result_count]
            if score > 0
        ]


def download_csv(
    destination: Path, previous_state: dict[str, str]
) -> tuple[bool, dict[str, str], Path | None]:
    headers = {"User-Agent": "post-recommender/1.0"}
    if previous_state.get("etag"):
        headers["If-None-Match"] = previous_state["etag"]
    if previous_state.get("last_modified"):
        headers["If-Modified-Since"] = previous_state["last_modified"]
    try:
        with urlopen(Request(CSV_URL, headers=headers), timeout=60) as response:
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
    state = {**previous_state, **response_state, "sha256": new_hash}
    if new_hash == previous_state.get("sha256"):
        return False, state, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    try:
        read_csv(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True, state, temporary


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
                print(f"Atualização ignorada; índice anterior mantido: {error}")

    def run_initial_update(self) -> None:
        for attempt in range(1, 4):
            try:
                self.update_once()
                return
            except Exception as error:
                print(f"Download inicial do CSV falhou ({attempt}/3): {error}")
                if attempt < 3:
                    time.sleep(attempt * 10)
        if CSV_PATH.exists():
            try:
                self.index.rebuild(CSV_PATH, file_hash(CSV_PATH))
                print("Índice carregado a partir do CSV local")
            except Exception as error:
                print(f"CSV local inválido; serviço aguardará nova atualização: {error}")


class RecommenderHandler(BaseHTTPRequestHandler):
    index: PostIndex

    def do_GET(self) -> None:
        if self.path == "/":
            response = json.dumps(
                {
                    "service": "post-recommender",
                    "status": "online",
                    "endpoints": {
                        "health": "GET /health",
                        "recommend": "POST /recommend",
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
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
            result_count = min(max(int(payload.get("k", 5)), 1), 50)
            response = json.dumps(
                {"results": self.index.recommend(str(payload.get("query", "")), result_count)},
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
    index = PostIndex()
    watcher = CsvWatcher(index)
    RecommenderHandler.index = index
    threading.Thread(target=watcher.run_initial_update, daemon=True).start()
    threading.Thread(target=watcher.run, daemon=True).start()
    print(f"Serviço de posts em http://{HOST}:{PORT}; verificações a cada {POLL_INTERVAL}s")
    ThreadingHTTPServer((HOST, PORT), RecommenderHandler).serve_forever()


if __name__ == "__main__":
    main()
