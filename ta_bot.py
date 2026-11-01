"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Starter stub. Implement the methods below.
Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations


# class TABot:
#     """Course TA-Bot with RAG retrieval and cross-session memory."""

#     def __init__(self, config: dict) -> None:
#         """
#         Initialise the TA-Bot.

#         Parameters
#         ----------
#         config : dict
#             Required keys set by the harness (do not hard-code their values):
#                 endpoint   : str   – vLLM-compatible base URL
#                 model      : str   – model name served at the endpoint
#                 corpus_path: str   – path to the paper corpus directory
#                 temperature: float – generation temperature (0 for graded runs)
#                 seed       : int   – random seed for reproducibility
#             You may add custom keys for your own use.
#         """
#         self.config = config
#         # TODO: initialise your retrieval pipeline, memory store, etc.
#         raise NotImplementedError("Implement __init__")

#     def chat(
#         self,
#         session_id: str,
#         user_id: str,
#         message: str,
#     ) -> dict:
#         """
#         Process one user turn and return an answer with citations.

#         Parameters
#         ----------
#         session_id : str
#             Conversation scope — a new session_id starts a fresh
#             conversation but keeps the user's cross-session memory.
#         user_id : str
#             Memory scope — identifies the user whose memory store
#             to read from and write to.
#         message : str
#             The user's message.

#         Returns
#         -------
#         dict
#             {"answer": str, "citations": list[str]}
#             - answer: the bot's response (complete, no streaming)
#             - citations: list of paper_id strings from the corpus
#               that were used to produce the answer. May be empty
#               if no retrieval was needed.

#         Raises
#         ------
#         TimeoutError
#             You MUST raise TimeoutError if the endpoint is
#             unreachable or a request exceeds your timeout.
#             The harness catches it and marks this turn as 0.
#             Do NOT let the process hang — raise, don't wait.
#         """
#         # TODO: implement retrieval, generation, and memory update
#         raise NotImplementedError("Implement chat")

#     def reset_memory(self, user_id: str) -> None:
#         """
#         Clear all stored memory for the given user.

#         The harness calls this before each memory-probe sequence
#         to ensure a clean slate. After this call, ``get_memory(user_id)``
#         must return an empty list.

#         Parameters
#         ----------
#         user_id : str
#             The user whose memory to clear.
#         """
#         # TODO: clear the memory store for this user
#         raise NotImplementedError("Implement reset_memory")

#     def get_memory(self, user_id: str) -> list[str]:
#         """
#         Return the current memory store for the given user as a
#         list of human-readable fact strings.

#         This is read-only — calling it must not change the store.
#         The harness uses it to inspect what your bot chose to remember
#         (and what it chose not to remember).

#         Any internal representation is fine as long as each entry
#         serialises to a readable string.

#         Parameters
#         ----------
#         user_id : str
#             The user whose memory to inspect.

#         Returns
#         -------
#         list[str]
#             Stored facts as readable strings. Empty list if nothing
#             is stored (or after ``reset_memory``).
#         """
#         # TODO: return the contents of the memory store
#         raise NotImplementedError("Implement get_memory")

import sqlite3
import requests
import json

class TABot:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.endpoint = config["endpoint"]
        
        # Connect to the provided SQLite database
        db_path = f"{config['corpus_path']}/corpus.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # State Management
        # session_id maps to short-term conversational context (lists of dicts)
        self.sessions = {} 
        # user_id maps to long-term persistent facts (lists of strings)
        self.users = {}

    def _extract_memory(self, message: str) -> str:
        """
        Hidden LLM call to evaluate if the message contains durable facts.
        This is your 'gatekeeper' to pass the negative memory probes.
        """
        # Pseudo-logic: Ask the LLM if 'message' contains a fact to remember.
        # If it does, return it. If it is a negative probe or irrelevant, return None.
        pass

    def _retrieve_chunks(self, query: str, k: int = 3) -> tuple[str, list]:
        """
        Baseline lexical retrieval using SQLite's FTS5 virtual table.
        """
        cursor = self.conn.cursor()
        # The assignment notes chunks_fts is a virtual table offering BM25 scoring
        sql = """
            SELECT p.paper_id, c.text 
            FROM chunks_fts fts
            JOIN chunks c USING(chunk_id)
            JOIN papers p USING(paper_id)
            WHERE chunks_fts MATCH ? 
            ORDER BY bm25(chunks_fts) 
            LIMIT ?
        """
        # Note: FTS5 syntax requires escaping specific characters in the query
        clean_query = query.replace("'", "").replace('"', "") 
        cursor.execute(sql, (clean_query, k))
        
        results = cursor.fetchall()
        
        citations = []
        context_text = ""
        for paper_id, text in results:
            if paper_id not in citations:
                citations.append(paper_id)
            context_text += f"\n{text}\n\n"
            
        return context_text, citations

    def chat(self, session_id: str, user_id: str, message: str) -> dict:
        # 1. State Initialization
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        if user_id not in self.users:
            self.users[user_id] = []

        # 2. Memory Gatekeeper (Update long-term facts)
        new_fact = self._extract_memory(message)
        if new_fact:
            self.users[user_id].append(new_fact)

        # 3. Retrieval
        # For a better hit rate, you can prepend the last turn from self.sessions 
        # to the message to resolve pronouns before searching.
        context_str, citations = self._retrieve_chunks(message, k=5)

        # 4. Prompt Assembly
        system_prompt = "You are a course TA. Answer based strictly on the provided context."
        user_facts = "\n".join(self.users[user_id])
        chat_history = "\n".join(self.sessions[session_id])
        
        augmented_prompt = f"""
        {system_prompt}
        
        Known facts about the user:
        {user_facts}
        
        Recent conversation:
        {chat_history}
        
        Reference Material:
        {context_str}
        
        User Question: {message}
        """

        # 5. Generation
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": augmented_prompt}],
            "temperature": self.config.get("temperature", 0),
            "seed": self.config.get("seed", 42)
        }
        
        try:
            # The vLLM endpoint mimics the OpenAI chat completions API
            response = requests.post(
                f"{self.endpoint}/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise TimeoutError("Endpoint timed out")
            
        # 6. Post-Processing & State Update
        self.sessions[session_id].append(f"User: {message}")
        self.sessions[session_id].append(f"TA: {answer}")
        
        # 7. Strict Contract Return
        return {
            "answer": answer,
            "citations": citations # Must be paper_ids, not chunk_ids
        }

    def reset_memory(self, user_id: str) -> None:
        """Clear persistent memory for a specific user."""
        if user_id in self.users:
            self.users[user_id] = []

    def get_memory(self, user_id: str) -> list[str]:
        """Return stored facts about this user."""
        return self.users.get(user_id, [])