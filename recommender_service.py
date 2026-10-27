from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HOST = os.getenv("RECOMMENDER_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("RECOMMENDER_PORT", "8090")))
model = SentenceTransformer(MODEL_NAME, device="cpu")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def recommend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("'candidates' deve ser uma lista")

    profile_texts = payload.get("profileTexts", [])
    if not isinstance(profile_texts, list):
        raise ValueError("'profileTexts' deve ser uma lista")

    if not candidates:
        return []

    candidate_texts = [str(candidate.get("text", "")) for candidate in candidates]
    candidate_vectors = model.encode(
        candidate_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    )

    if profile_texts:
        profile_vectors = model.encode(
            [str(text) for text in profile_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        profile_vector = normalize(np.mean(profile_vectors, axis=0))
    else:
        profile_vector = model.encode(
            "comida boa, restaurante agradável e uma boa experiência para comer",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    scores = candidate_vectors @ profile_vector
    results = [
        {"id": candidate.get("id"), "score": float(score)}
        for candidate, score in zip(candidates, scores)
        if candidate.get("id")
    ]
    return sorted(results, key=lambda result: result["score"], reverse=True)


class RecommenderHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404, "Rota não encontrada")
            return

        response = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
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
            response = json.dumps({"results": recommend(payload)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
        except Exception as error:
            print(f"Erro ao calcular embeddings: {error}")
            self.send_error(500, "Erro ao calcular embeddings")

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)


if __name__ == "__main__":
    print(f"Modelo carregado. Serviço de recomendação em http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), RecommenderHandler).serve_forever()
