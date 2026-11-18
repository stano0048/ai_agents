"""
llm.py - thin Gemini client (REST, no SDK needed).

The key is never written to the database or to disk. It is read, in this order, from
  1. a key pasted in the sidebar for the current session,
  2. Streamlit secrets (GEMINI_API_KEY),
  3. the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
Every call returns a dict {ok, text, sources, error}; nothing here raises, so the agents always have an offline fallback.
"""
from __future__ import annotations

import json
import os
import re

import requests

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def extract_json(text: str):
    """Pull the first JSON object or array out of a model reply (handles ```json fences). None if there is none."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    return None


class Gemini:
    def __init__(self, key: str | None = None, model: str | None = None, post=None, db=None):
        self.session_key = (key or "").strip() or None
        self._model = (model or "").strip() or None
        self._post = post or requests.post       # injectable for tests
        self.db = db
        self.last_error = ""
        self.calls = 0

    # ---------------------------------------------------------------- model and key handling
    @property
    def model(self) -> str:
        """Explicit model, else the one saved in the database, else the default."""
        return self._model or (self.db.get("gemini_model") if self.db else None) or DEFAULT_MODEL

    def saved_key(self) -> str | None:
        return (self.db.get("gemini_key") or None) if self.db else None

    def key(self) -> str | None:
        if self.session_key:
            return self.session_key
        if self.saved_key():
            return self.saved_key()
        try:
            import streamlit as st
            k = st.secrets.get("GEMINI_API_KEY")
            if k:
                return str(k)
        except Exception:
            pass
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None

    def key_source(self) -> str:
        if self.session_key:
            return "pasted for this session"
        if self.saved_key():
            return "saved in the database"
        try:
            import streamlit as st
            if st.secrets.get("GEMINI_API_KEY"):
                return "Streamlit secrets"
        except Exception:
            pass
        return "environment variable" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "none"

    def available(self) -> bool:
        return bool(self.key())

    # ---------------------------------------------------------------- calls
    def generate(self, prompt: str, system: str | None = None, grounded: bool = False,
                 temperature: float = 0.3, cache_key: str | None = None) -> dict:
        """grounded=True lets Gemini use Google Search; the pages it used come back in `sources`."""
        if not self.available():
            return {"ok": False, "text": "", "sources": [], "error": "No Gemini API key."}
        if cache_key and self.db:
            hit = self.db.cache_get("gem:" + cache_key, max_age=3 * 86400)
            if hit:
                return {**hit, "cached": True}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if grounded:
            body["tools"] = [{"google_search": {}}]
        try:
            r = self._post(ENDPOINT.format(model=self.model), json=body, timeout=40,
                           headers={"x-goog-api-key": self.key(), "Content-Type": "application/json"})
            self.calls += 1
            if r.status_code != 200:
                msg = ""
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                self.last_error = f"Gemini returned HTTP {r.status_code}. {msg}".strip()
                return {"ok": False, "text": "", "sources": [], "error": self.last_error}
            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
            sources, seen = [], set()
            for ch in cand.get("groundingMetadata", {}).get("groundingChunks", []) or []:
                w = ch.get("web") or {}
                if w.get("uri") and w["uri"] not in seen:
                    seen.add(w["uri"])
                    sources.append({"title": w.get("title") or w["uri"], "url": w["uri"]})
            if not text:
                self.last_error = "Gemini returned an empty answer."
                return {"ok": False, "text": "", "sources": sources, "error": self.last_error}
            out = {"ok": True, "text": text, "sources": sources, "error": ""}
            if cache_key and self.db:
                self.db.cache_put("gem:" + cache_key, "gemini", out)
            return out
        except Exception as e:                    # network down, timeout, bad JSON ...
            self.last_error = f"Could not reach Gemini ({type(e).__name__})."
            return {"ok": False, "text": "", "sources": [], "error": self.last_error}

    def generate_json(self, prompt: str, system: str | None = None, grounded: bool = False, cache_key: str | None = None):
        res = self.generate(prompt + "\n\nReply with JSON only.", system, grounded, 0.2, cache_key)
        return (extract_json(res["text"]) if res["ok"] else None), res

    def test(self) -> tuple[bool, str]:
        r = self.generate("Reply with the single word: ready")
        return r["ok"], (r["text"][:60] if r["ok"] else r["error"])

    def list_models(self) -> tuple[list[str], str]:
        """Models this key can use for generateContent, newest names first. Returns (names, error)."""
        if not self.available():
            return [], "No Gemini API key."
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"pageSize": 200},
                             headers={"x-goog-api-key": self.key()}, timeout=20)
            if r.status_code != 200:
                return [], f"Google returned HTTP {r.status_code}."
            names = [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            return sorted(names, reverse=True), ""
        except Exception as e:
            return [], f"Could not reach Google ({type(e).__name__})."
