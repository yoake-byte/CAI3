"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Starter stub. Implement the methods below.
Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations

import sqlite3
import requests
import json
import logging
import re
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================================
# PART 1 — RETRIEVAL  (Anusha)
# ============================================================================================

STOP = set("""
a an the of to in on for by with and or is are was were be been being it its this that these those
what which who whom how why when where does do did done can could would should will shall may might
about from as at into than then there their them they we you your i me my our us he she his her
explain describe tell please give show summarize summarise between vs versus compare comparison
paper papers approach approaches method methods work works use used using uses also just really
some any all each other more most such only own same so very s t can't don't let lets let's
""".split())

ANAPHORA = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|above|previous|earlier|former|latter|"
    r"same|both|all (two|three|four)|we discussed|you mentioned|the paper|that paper|"
    r"that approach|this approach|that method|this method)\b", re.I)
TOPIC_SWITCH = re.compile(
    r"\b(switch(ing)? (topics?|gears)|change (of )?(topics?|subject)|changing (topics?|subject)|"
    r"different (topic|question|subject)|new (topic|question)|unrelated|moving on|on another note)\b",
    re.I)
SUMMARY_ALL = re.compile(
    r"\b(all (two|three|four|five|of them|of these|of those|the (approaches|methods|papers|strategies))|"
    r"we('ve| have)? discussed|so far|everything|each of (them|these|those)|compare (them|these|those)|"
    r"these (two|three|four) )\b", re.I)
NUMBER_WORDS = {"two": 2, "both": 2, "three": 3, "four": 4, "five": 5}
RRF_K = 60


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOP and len(token) > 1]


class SessionContext:
    """Per-session retrieval state, separate from the user's long-term memory."""

    def __init__(self) -> None:
        self.user_turns: list[str] = []
        self.queries: list[str] = []
        self.primary: list[str] = []
        self.cited: list[str] = []

class Retriever:
    def __init__(self, corpus_path: str | Path, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.cfg = cfg
        db_path = Path(corpus_path) / "corpus.db"
        if not db_path.exists():
            raise FileNotFoundError(f"corpus.db not found under {corpus_path}")
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)

        self.k_papers = int(cfg.get("k_papers", 3))
        self.context_budget = int(cfg.get("context_budget", 1600))
        self.max_chunks_per_paper = int(cfg.get("max_chunks_per_paper", 0))
        self.min_quality = float(cfg.get("min_quality", 0.4))
        self.bm25_pool = int(cfg.get("bm25_pool", 200))
        self.w_title, self.w_section, self.w_text = cfg.get("bm25_weights", (4.0, 1.0, 1.0))
        self.ctx_weight = float(cfg.get("ctx_weight", 0.7))
        self.use_dense = bool(cfg.get("dense", False))
        self.dense_weight = float(cfg.get("dense_weight", 1.0))
        self.rewrite_mode = cfg.get("rewrite", "heuristic")
        if self.rewrite_mode is True:
            self.rewrite_mode = "heuristic"
        elif self.rewrite_mode is False:
            self.rewrite_mode = "none"

        self.titles: dict[str, str] = dict(self.db.execute("SELECT paper_id, title FROM papers"))
        self.endpoint = cfg.get("endpoint")
        self.model = cfg.get("model")
        self.embed_model = cfg.get("embed_model") or self.model
        self.timeout = float(cfg.get("timeout", 60))
        self.seed = int(cfg.get("seed", 42))
        self._dense = self._load_or_build_dense(Path(corpus_path)) if self.use_dense else None

    # =================================================================== public API
    def contextualise(self, message: str, sess: SessionContext, llm=None) -> str:
        if self.rewrite_mode == "none" or not sess.user_turns or TOPIC_SWITCH.search(message):
            return message
        if self.rewrite_mode == "llm" and llm is not None:
            return self._llm_rewrite(message, sess, llm)
        if not (ANAPHORA.search(message) or len(_tokens(message)) <= 3):
            return message
        previous = sess.user_turns[-1]
        previous_title = self.titles.get(sess.primary[-1], "") if sess.primary else ""
        return f"{message} {previous} {previous_title}".strip()

    def retrieve(self, query: str, message: str | None = None) -> tuple[list[dict], list[str]]:
        rankings: list[tuple[list[str], float]] = []
        bm25_chunks = self._bm25(query)
        bm25_papers = list(dict.fromkeys(chunk["paper_id"] for chunk in bm25_chunks))
        if message and message != query:
            current_chunks = self._bm25(message)
            rankings.append((list(dict.fromkeys(c["paper_id"] for c in current_chunks)), 1.0))
            rankings.append((bm25_papers, self.ctx_weight))
            bm25_chunks = current_chunks + bm25_chunks
        else:
            rankings.append((bm25_papers, 1.0))

        dense_chunks: list[dict] = []
        if self._dense is not None:
            dense_chunks = self._dense_search(query)
            rankings.append((list(dict.fromkeys(c["paper_id"] for c in dense_chunks)), self.dense_weight))

        scores: dict[str, float] = defaultdict(float)
        for ranked, weight in rankings:
            for rank, paper_id in enumerate(ranked):
                scores[paper_id] += weight / (RRF_K + rank + 1)
        papers = sorted(scores, key=lambda paper_id: (-scores[paper_id], paper_id))
        chunks = self._chunks_for_papers(message or query, papers[:self.k_papers], bm25_chunks + dense_chunks)
        return chunks, papers

    def select_citations(self, message: str, papers: list[str], sess: SessionContext) -> list[str]:
        citations = papers[:self.k_papers]
        if SUMMARY_ALL.search(message) and sess.primary:
            count = len(sess.primary)
            for word, value in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", message, re.I):
                    count = value
                    break
            earlier = list(dict.fromkeys(sess.primary))[-count:]
            citations = list(dict.fromkeys(earlier + citations))[:max(self.k_papers, len(earlier))]
        return citations

    def update_session(self, sess: SessionContext, message: str, query: str,
                       papers: list[str], citations: list[str]) -> None:
        sess.user_turns.append(message)
        sess.queries.append(query)
        if citations or papers:
            pool = citations or papers
            fresh = [paper_id for paper_id in pool if paper_id not in sess.primary]
            sess.primary.append(fresh[0] if fresh else pool[0])
        sess.cited.extend(paper_id for paper_id in citations if paper_id not in sess.cited)
    
    # =================================================================== BM25
    def _fts_query(self, text: str) -> str:
        tokens = list(dict.fromkeys(_tokens(text)))
        return " OR ".join(f'"{token}"' for token in tokens)

    def _bm25(self, query: str) -> list[dict]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        rows = self.db.execute(
            f"""SELECT c.chunk_id, c.paper_id, c.section, c.text, c.n_tokens,
                       bm25(chunks_fts, 0, 0, {self.w_title}, {self.w_section}, {self.w_text}) AS score
                FROM chunks_fts JOIN chunks c USING(chunk_id)
                WHERE chunks_fts MATCH ? AND c.quality >= ?
                ORDER BY score, c.chunk_id LIMIT ?""",
            (fts_query, self.min_quality, self.bm25_pool),
        ).fetchall()
        return [{"chunk_id": cid, "paper_id": pid, "section": section, "text": text,
                 "n_tokens": n_tokens, "score": -score}
                for cid, pid, section, text, n_tokens, score in rows]

    def _best_chunk_in_paper(self, query: str, paper_id: str) -> dict | None:
        fts_query = self._fts_query(query)
        row = None
        if fts_query:
            row = self.db.execute(
                f"""SELECT c.chunk_id, c.section, c.text, c.n_tokens
                    FROM chunks_fts JOIN chunks c USING(chunk_id)
                    WHERE chunks_fts MATCH ? AND c.paper_id = ? AND c.quality >= ?
                    ORDER BY bm25(chunks_fts, 0, 0, {self.w_title}, {self.w_section}, {self.w_text}),
                             c.chunk_id LIMIT 1""",
                (fts_query, paper_id, self.min_quality),
            ).fetchone()
        if row is None:
            row = self.db.execute(
                "SELECT chunk_id, section, text, n_tokens FROM chunks WHERE paper_id=? ORDER BY chunk_index LIMIT 1",
                (paper_id,),
            ).fetchone()
        if row is None:
            return None
        return {"chunk_id": row[0], "paper_id": paper_id, "section": row[1], "text": row[2],
                "n_tokens": row[3], "score": 0.0}

    def _chunks_for_papers(self, query: str, papers: list[str], pool: list[dict]) -> list[dict]:
        allowed = set(papers)
        ranked, seen = [], set()
        for chunk in pool:
            if chunk["chunk_id"] not in seen and chunk["paper_id"] in allowed:
                seen.add(chunk["chunk_id"])
                ranked.append(chunk)
        first_chunks = []
        if papers:
            best = next((chunk for chunk in ranked if chunk["paper_id"] == papers[0]), None)
            best = best or self._best_chunk_in_paper(query, papers[0])
            if best:
                first_chunks.append(best)
        cap = self.max_chunks_per_paper or 10**9
        output, used_tokens, per_paper = [], 0, defaultdict(int)
        for chunk in first_chunks + [chunk for chunk in ranked if chunk not in first_chunks]:
            if chunk["chunk_id"] in {item["chunk_id"] for item in output} or per_paper[chunk["paper_id"]] >= cap:
                continue
            token_count = int(chunk.get("n_tokens") or len(chunk["text"].split()) * 1.3)
            if used_tokens + token_count > self.context_budget and output:
                continue
            output.append(chunk)
            used_tokens += token_count
            per_paper[chunk["paper_id"]] += 1
        return output

    # =================================================================== dense (optional)
    def _base(self) -> str:
        endpoint = (self.endpoint or "").rstrip("/")
        return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            response = requests.post(f"{self._base()}/embeddings", timeout=self.timeout,
                                     json={"model": self.embed_model, "input": texts})
            response.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise TimeoutError(str(exc)) from exc
        vectors = np.array([item["embedding"] for item in sorted(response.json()["data"],
                          key=lambda item: item["index"])], dtype=np.float32)
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    def _load_or_build_dense(self, corpus_path: Path):
        rows = self.db.execute(
            "SELECT chunk_id, paper_id, section, text FROM chunks WHERE quality >= ? ORDER BY chunk_id",
            (self.min_quality,),
        ).fetchall()
        key = hashlib.sha256((str(self.embed_model) + "|" + "|".join(row[0] for row in rows)
                              ).encode()).hexdigest()[:16]
        cache_dir = Path(self.cfg.get("cache_dir") or Path(__file__).resolve().parent / ".cache")
        cache = cache_dir / f"dense_{key}.npy"
        if cache.exists():
            matrix = np.load(cache)
        else:
            parts = [self._embed([row[3][:2000] for row in rows[index:index + 64]])
                     for index in range(0, len(rows), 64)]
            matrix = np.vstack(parts)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache, matrix)
            except OSError:
                pass
        metadata = [{"chunk_id": row[0], "paper_id": row[1], "section": row[2], "text": row[3]}
                    for row in rows]
        return matrix, metadata

    def _dense_search(self, query: str, k: int = 50) -> list[dict]:
        matrix, metadata = self._dense
        query_vector = self._embed([query])[0]
        similarities = matrix @ query_vector
        indices = np.argsort(-similarities, kind="stable")[:k]
        return [dict(metadata[index], score=float(similarities[index])) for index in indices]


    # =================================================================== LLM rewrite (optional)
    def _llm_rewrite(self, message: str, sess: SessionContext, llm) -> str:
        history = "\n".join(f"- {turn}" for turn in sess.user_turns)
        prompt = [
            {"role": "system", "content":
             "Rewrite the user's latest question into a single standalone search query for a "
             "corpus of NLP research papers. Resolve pronouns and references like 'it', 'that "
             "approach' or 'all three' using the earlier questions. Keep method and paper names. "
             "If the latest question starts a new topic, return it unchanged. Output only the rewritten query."},
            {"role": "user", "content": f"Earlier questions:\n{history}\n\nLatest question: {message}"},
        ]
        output = (llm(prompt, max_tokens=80) or "").strip().strip('"').splitlines()
        rewritten = output[0].strip() if output else ""
        if not rewritten or len(rewritten) > 400 or "mock response" in rewritten.lower():
            return message
        return rewritten

# ============================================================================================
# PART 2 — MEMORY  (Yu)
# ============================================================================================


class SessionMemory:
    """Conversational turns scoped to one session."""

    def __init__(self) -> None:
        self._turns: dict[str, list[tuple[str, str]]] = {}

    def update(self, session_id: str, user_message: str, answer: str) -> None:
        turns = self._turns.setdefault(session_id, [])
        turns.extend((("user", user_message), ("assistant", answer)))

    def retrieve(self, session_id: str) -> list[str]:
        turns = self._turns.get(session_id, [])
        return [f"{'User' if role == 'user' else 'TA'}: {content}" for role, content in turns]

    def user_turns(self, session_id: str) -> list[str]:
        return [content for role, content in self._turns.get(session_id, []) if role == "user"]

    def messages(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._turns.get(session_id, []))

    def reset(self, session_id: str) -> None:
        self._turns.pop(session_id, None)


# =================================================================== Long-term memory
class UserMemory:
    """Durable, filtered facts scoped to a user across sessions."""

    _RECALL_INTENT = re.compile(
        r"\b(remember|recall|what do you know about me|tell me about myself)\b", re.I)
    _SENSITIVE = re.compile(
        r"\b(password|passcode|api key|secret token|social security number|ssn|"
        r"credit card number|bank account|home address|medical diagnosis|medication|"
        r"religion|political affiliation|sexual orientation)\b", re.I)
    _QUERY_STOP = STOP | {
        "remember", "recall", "know", "about", "user", "fact", "facts", "tell",
        "again", "please", "what", "who", "me", "my",
    }

    def __init__(self, config: dict, endpoint: str) -> None:
        self.config = config
        self.endpoint = endpoint
        self._facts: dict[str, list[str]] = {}

    
    # ====================================================Remove outdated facts, add new facts to memory

    def update(self, user_id: str, message: str) -> None:
        existing = self._facts.get(user_id, [])
        result = self._extract(message, existing)
        if result is None:
            return

        facts = result.get("facts")
        if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
            logger.warning("Memory extraction returned an invalid facts field; no facts stored.")
            return
        forget = result.get("forget", [])
        if not isinstance(forget, list) or not all(isinstance(fact, str) for fact in forget):
            forget = []

        forgotten = {self._normalize(fact).casefold() for fact in forget}
        updated = [fact for fact in existing if self._normalize(fact).casefold() not in forgotten]
        known = {self._normalize(fact).casefold() for fact in updated}
        for fact in facts:
            normalized = self._normalize(fact)
            if not normalized or len(normalized) > 400 or self._SENSITIVE.search(normalized):
                continue
            if normalized.casefold() not in known:
                updated.append(normalized)
                known.add(normalized.casefold())
        if updated:
            self._facts[user_id] = updated
        else:
            self._facts.pop(user_id, None)

    # =================================================================== Retrieve facts from memory
    def retrieve(self, user_id: str, query: str) -> list[str]:
        facts = self._facts.get(user_id, [])
        if not facts:
            return []
        if self._RECALL_INTENT.search(query):
            return list(facts)

        query_tokens = self._tokens(query)
        if not query_tokens:
            return []
        return [fact for fact in facts if self._tokens(fact) & query_tokens]

    def get(self, user_id: str) -> list[str]:
        return list(self._facts.get(user_id, []))

    def reset(self, user_id: str) -> None:
        self._facts.pop(user_id, None)


    # ===================================================================Extract facts
    def _extract(self, message: str, existing: list[str]) -> dict | None:
        prompt = (
            "Extract only durable facts the user directly asserts about themself, such as their "
            "name, ongoing project, or stated interests. For projects, preserve the distinctive "
            "subject and any optional details the user gives, such as whether it is a semester "
            "project or a long-term goal, or its timeframe or end date. Do not require these "
            "details in order to remember a project, and do not treat a project as temporary only "
            "because it is tied to a semester or has an end date. Do not infer facts from questions "
            "or course content. Do not store hypothetical, temporary, sensitive, third-party, "
            "quoted, or negated claims. If the user corrects or retracts a previously stored fact, "
            "put the exact old fact from Existing facts in `forget`; do not keep both versions. "
            "Return only a JSON object with `facts` and `forget` arrays of concise strings. "
            "Use empty arrays when appropriate.\n\n"
            f"Existing facts: {json.dumps(existing)}\n"
            f"User message: {message}"
        )
        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.config["model"],
                    "messages": [
                        {"role": "system", "content": "You are a conservative user-fact extractor."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "seed": self.config.get("seed", 42),
                },
                timeout=self.config.get("timeout", 15),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise TimeoutError("Memory extraction endpoint timed out or is unreachable") from exc

        try:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                raise json.JSONDecodeError("No JSON object", content, 0)
            extracted = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            logger.warning("Memory extraction returned invalid JSON; no facts stored.")
            return None
        if not isinstance(extracted, dict):
            logger.warning("Memory extraction returned an invalid object; no facts stored.")
            return None
        return extracted

    @staticmethod
    def _normalize(fact: str) -> str:
        return " ".join(fact.split())

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in cls._QUERY_STOP or len(token) <= 1:
                continue
            tokens.add("interest" if token in {"interested", "interests"} else token)
        return tokens


class Memory:
    """Coordinates session-scoped context and cross-session user facts."""

    def __init__(self, config: dict, endpoint: str) -> None:
        self.session = SessionMemory()
        self.user = UserMemory(config, endpoint)

# ============================================================================================
# PART 3 — TA BOT
# ============================================================================================

class TABot:
    def __init__(self, config: dict) -> None:
        self.config = config
        endpoint = config["endpoint"].rstrip("/")
        self.base = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
        self.model = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 42))
        self.timeout = float(config.get("timeout", 60))
        self.retriever = Retriever(config["corpus_path"], config)
        self.memory = Memory(config, self.base)
        self.sessions: dict[str, SessionContext] = {}

    def chat(self, session_id: str, user_id: str, message: str) -> dict:
        # 1. Update user memory, then retrieve evidence and session context.
        self.memory.user.update(user_id, message)

        sess = self.sessions.setdefault(session_id, SessionContext())
        history = self.memory.session.messages(session_id)
        query = self.retriever.contextualise(message, sess, llm=self._llm)
        chunks, papers = self.retriever.retrieve(query, message=message)
        citations = self.retriever.select_citations(message, papers, sess)
        if set(citations) - {chunk["paper_id"] for chunk in chunks}:
            citation_query = " ".join(
                [query] + [self.retriever.titles.get(paper_id, "") for paper_id in citations]
            )
            citation_pool = self.retriever._bm25(citation_query)
            chunks = self.retriever._chunks_for_papers(citation_query, citations, citation_pool)

        answer = self._generate(user_id, message, history, chunks)
        self.retriever.update_session(sess, message, query, papers, citations)
        self.memory.session.update(session_id, message, answer)
        return {"answer": answer, "citations": citations}

    def _generate(
        self,
        user_id: str,
        message: str,
        history: list[tuple[str, str]],
        chunks: list[dict],
    ) -> str:
        context = "\n\n".join(
            f"[{chunk['paper_id']}] {self.retriever.titles.get(chunk['paper_id'], '')} - "
            f"{chunk.get('section') or 'passage'}\n{chunk['text']}"
            for chunk in chunks
        ) or "(no relevant passages found)"
        recalled_facts = self.memory.user.retrieve(user_id, message)
        if self.memory.user._RECALL_INTENT.search(message) and recalled_facts:
            return "I remember: " + "; ".join(recalled_facts)
        facts = "\n".join(recalled_facts)
        conversation = "\n".join(f"{role.title()}: {text}" for role, text in history)
        reasoning = "Reasoning: low\n" if "gpt-oss" in self.model.lower() else ""
        system = (
            f"{reasoning}"
            "You are a teaching assistant for a graduate course on conversational AI. "
            "Answer ONLY from the provided passages. Cite papers inline by their id in brackets, "
            "e.g. [L04-04]. If the passages do not contain the answer, say so briefly. "
            "Be accurate, specific, and concise. User facts may personalize the answer but are "
            "not course evidence."
        )
        messages = [{"role": "system", "content": system}]
        messages.append({
            "role": "user",
            "content": (
                f"Known facts about the user:\n{facts}\n\n"
                f"Recent conversation:\n{conversation}\n\n"
                f"Passages:\n{context}\n\nUser Question: {message}"
            ),
        })
        return self._llm(messages)

    def _llm(self, messages: list[dict], max_tokens: int = 4096) -> str:
        try:
            response = requests.post(
                f"{self.base}/chat/completions",
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise TimeoutError(str(exc)) from exc

        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("Chat completion response has no choices[0].message") from exc

        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason")
            message_fields = sorted(message) if isinstance(message, dict) else []
            raise RuntimeError(
                "Chat completion returned no text content "
                f"(finish_reason={finish_reason!r}, message_fields={message_fields})"
            )
        return content.strip()

    def reset_memory(self, user_id: str) -> None:
        """Clear memory for a specific user."""
        self.memory.user.reset(user_id)

    def get_memory(self, user_id: str) -> list[str]:
        """Return stored facts about this user."""
        return self.memory.user.get(user_id)
