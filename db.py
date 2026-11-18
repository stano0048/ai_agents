"""
db.py - the SQLite database that gives the agents memory.

What is remembered between visits
  * group members (max 6) and their responsibilities
  * the group passphrase (salted PBKDF2 hash, never the plain text) and the "who may add members" policy
  * what the support agent has learned: new training phrases, topic penalties, approved researched answers, tickets
  * what the finance agent has learned from category corrections
  * research results (so the same unit or question is not researched twice) and Gemini answers
  * the product catalogue used by the support agent
  * test runs, shown on the Testing page and in the PDF report

The database file is data/agents.db unless the DB_PATH environment variable says otherwise.
NOTE: some hosts (for example Streamlit Community Cloud) wipe local files on restart. Point DB_PATH at a persistent
disk, or use the backup / restore buttons on the Group page.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

MAX_MEMBERS = 6
PBKDF2_ROUNDS = 200_000
MAX_FAILS = 5
LOCK_SECONDS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    reg_no TEXT DEFAULT '',
    role TEXT DEFAULT '',
    responsibilities TEXT DEFAULT '',
    agents TEXT DEFAULT '',
    created_at REAL, updated_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS members_name_unique ON members (lower(name));
CREATE TRIGGER IF NOT EXISTS members_limit BEFORE INSERT ON members
WHEN (SELECT COUNT(*) FROM members) >= 6
BEGIN SELECT RAISE(ABORT, 'member limit reached'); END;
CREATE TABLE IF NOT EXISTS member_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, action TEXT, member TEXT);
CREATE TABLE IF NOT EXISTS support_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, intent TEXT);
CREATE TABLE IF NOT EXISTS support_penalty (intent TEXT PRIMARY KEY, value REAL);
CREATE TABLE IF NOT EXISTS support_tickets (id TEXT PRIMARY KEY, ts REAL, question TEXT, department TEXT, priority TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS support_qa (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, sources TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS support_topics (id TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS fin_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT, category TEXT);
CREATE TABLE IF NOT EXISTS research_cache (key TEXT PRIMARY KEY, kind TEXT, payload TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS test_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, name TEXT, agent TEXT, passed INTEGER, detail TEXT, seconds REAL);
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY, name TEXT, category TEXT, unit TEXT, price REAL, stock INTEGER, aisle TEXT, offer_price REAL
);
"""

# Sample catalogue (clearly sample data; replace with the store's real products).
SEED_PRODUCTS = [
    ("P001", "Unga Pembe maize flour", "Dry goods", "2kg", 190, 140, "4", None),
    ("P002", "Sugar", "Dry goods", "1kg", 175, 220, "4", None),
    ("P003", "Cooking oil", "Dry goods", "1L", 330, 95, "5", 299),
    ("P004", "Rice pishori", "Dry goods", "2kg", 480, 60, "4", None),
    ("P005", "Wheat flour", "Dry goods", "2kg", 215, 110, "4", None),
    ("P006", "Spaghetti", "Dry goods", "500g", 135, 0, "4", None),
    ("P007", "Fresh milk", "Dairy", "500ml", 65, 300, "9", None),
    ("P008", "Yoghurt strawberry", "Dairy", "500ml", 120, 80, "9", 99),
    ("P009", "Butter", "Dairy", "250g", 380, 45, "9", None),
    ("P010", "Eggs tray", "Dairy", "30 eggs", 520, 70, "9", None),
    ("P011", "White bread", "Bakery", "400g", 65, 120, "2", None),
    ("P012", "Brown bread", "Bakery", "400g", 75, 90, "2", None),
    ("P013", "Tomatoes", "Fresh produce", "1kg", 120, 85, "1", None),
    ("P014", "Onions", "Fresh produce", "1kg", 110, 100, "1", None),
    ("P015", "Potatoes", "Fresh produce", "1kg", 95, 150, "1", None),
    ("P016", "Sukuma wiki bunch", "Fresh produce", "1 bunch", 30, 200, "1", None),
    ("P017", "Bananas", "Fresh produce", "1kg", 100, 60, "1", None),
    ("P018", "Avocado", "Fresh produce", "each", 40, 0, "1", None),
    ("P019", "Chicken whole", "Meat", "1.2kg", 650, 40, "7", None),
    ("P020", "Beef with bone", "Meat", "1kg", 640, 55, "7", None),
    ("P021", "Tilapia fish", "Meat", "1kg", 600, 25, "7", None),
    ("P022", "Orange juice", "Drinks", "1L", 220, 130, "6", 189),
    ("P023", "Bottled water", "Drinks", "1.5L", 70, 400, "6", None),
    ("P024", "Soda cola", "Drinks", "2L", 150, 210, "6", None),
    ("P025", "Tea leaves", "Drinks", "500g", 260, 90, "6", None),
    ("P026", "Coffee instant", "Drinks", "100g", 410, 50, "6", None),
    ("P027", "Laundry soap bar", "Household", "1kg", 210, 130, "11", None),
    ("P028", "Washing powder", "Household", "2kg", 480, 75, "11", 429),
    ("P029", "Toilet paper", "Household", "10 rolls", 520, 90, "11", None),
    ("P030", "Dish soap", "Household", "750ml", 185, 100, "11", None),
    ("P031", "Toothpaste", "Personal care", "100ml", 190, 120, "12", None),
    ("P032", "Body lotion", "Personal care", "400ml", 450, 60, "12", None),
    ("P033", "Bar soap", "Personal care", "150g", 80, 200, "12", None),
    ("P034", "Baby diapers", "Baby", "pack of 44", 1350, 35, "13", 1199),
    ("P035", "Baby formula", "Baby", "400g", 1100, 0, "13", None),
    ("P036", "Biscuits", "Snacks", "250g", 90, 160, "8", None),
    ("P037", "Crisps", "Snacks", "100g", 75, 170, "8", None),
    ("P038", "Chocolate bar", "Snacks", "90g", 140, 110, "8", None),
    ("P039", "Quinoa", "Dry goods", "500g", 690, 0, "4", None),
    ("P040", "Peanut butter", "Dry goods", "400g", 420, 65, "5", None),
    ("P041", "Beans", "Dry goods", "1kg", 220, 120, "4", None),
    ("P042", "Lentils", "Dry goods", "500g", 170, 70, "4", None),
    ("P043", "Gas cylinder refill", "Household", "6kg", 1500, 25, "Customer service", None),
    ("P044", "Frozen peas", "Frozen", "500g", 210, 50, "10", None),
    ("P045", "Ice cream", "Frozen", "1L", 480, 40, "10", None),
]


def default_path() -> str:
    return os.environ.get("DB_PATH") or str(Path(__file__).resolve().parent / "data" / "agents.db")


class MemberLimitError(Exception):
    pass


class DuplicateMemberError(Exception):
    pass


class Database:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # ------------------------------------------------------------ plumbing
    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def _init(self):
        with self.conn() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
                c.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", SEED_PRODUCTS)

    # ------------------------------------------------------------ settings
    def get(self, key: str, default=None):
        with self.conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set(self, key: str, value) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, str(value)))

    # ------------------------------------------------------------ Gemini settings
    def gemini_settings(self) -> dict:
        k = self.get("gemini_key")
        return {"has_key": bool(k), "masked": (k[:4] + "..." + k[-4:]) if k and len(k) > 10 else ("set" if k else ""),
                "model": self.get("gemini_model") or ""}

    def set_gemini(self, key: str | None = None, model: str | None = None) -> None:
        """Save the Gemini key and/or model so every visit uses them. A blank argument leaves that setting unchanged."""
        if key and key.strip():
            self.set("gemini_key", key.strip())
        if model and model.strip():
            self.set("gemini_model", model.strip())

    def clear_gemini_key(self) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM settings WHERE key='gemini_key'")

    # ------------------------------------------------------------ passphrase
    @staticmethod
    def _hash(passphrase: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()

    @staticmethod
    def env_passphrase() -> str | None:
        return os.environ.get("GROUP_PASSPHRASE") or None

    def passphrase_is_set(self) -> bool:
        return bool(self.env_passphrase()) or self.get("pass_hash") is not None

    def passphrase_source(self) -> str:
        return "environment" if self.env_passphrase() else ("database" if self.get("pass_hash") else "none")

    def set_passphrase(self, new: str, old: str | None = None) -> tuple[bool, str]:
        """First visitor sets it; afterwards the old passphrase is required to change it."""
        if self.env_passphrase():
            return False, "The passphrase is set by the GROUP_PASSPHRASE environment variable on the server."
        if len(new.strip()) < 6:
            return False, "Use at least 6 characters."
        if self.get("pass_hash") is not None:
            ok, msg = self.verify_passphrase(old or "")
            if not ok:
                return False, msg
        salt = secrets.token_bytes(16)
        self.set("pass_salt", salt.hex())
        self.set("pass_hash", self._hash(new, salt))
        self.set("fail_count", 0)
        return True, "Passphrase saved."

    def lockout_remaining(self) -> int:
        until = float(self.get("locked_until", 0) or 0)
        return max(0, int(until - time.time()))

    def verify_passphrase(self, attempt: str) -> tuple[bool, str]:
        if not self.passphrase_is_set():
            return False, "No passphrase has been set yet."
        wait = self.lockout_remaining()
        if wait:
            return False, f"Too many wrong attempts. Try again in {wait} seconds."
        env = self.env_passphrase()
        if env:
            ok = hmac.compare_digest(attempt.encode(), env.encode())
        else:
            salt = bytes.fromhex(self.get("pass_salt"))
            ok = hmac.compare_digest(self._hash(attempt, salt), self.get("pass_hash"))
        if ok:
            self.set("fail_count", 0)
            return True, "Unlocked."
        fails = int(self.get("fail_count", 0) or 0) + 1
        self.set("fail_count", fails)
        if fails >= MAX_FAILS:
            self.set("locked_until", time.time() + LOCK_SECONDS)
            self.set("fail_count", 0)
            return False, f"Wrong passphrase. Locked for {LOCK_SECONDS} seconds."
        return False, f"Wrong passphrase ({MAX_FAILS - fails} attempts left)."

    # ------------------------------------------------------------ members
    def add_policy(self) -> str:
        return self.get("add_policy", "open")          # "open" = anyone with the link, "passphrase" = admin only

    @staticmethod
    def _clean(name, reg_no, role, resp, agents):
        name, reg_no, role, resp = (name or "").strip(), (reg_no or "").strip(), (role or "").strip(), (resp or "").strip()
        if not 2 <= len(name) <= 80:
            raise ValueError("Enter the member's full name (2 to 80 characters).")
        if len(resp) < 10:
            raise ValueError("Describe the responsibilities in at least 10 characters.")
        if len(resp) > 800 or len(role) > 80 or len(reg_no) > 40:
            raise ValueError("One of the fields is too long.")
        return name, reg_no, role, resp, ", ".join(a for a in (agents or []) if a)

    def list_members(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM members ORDER BY id")]

    def add_member(self, name, reg_no, role, responsibilities, agents) -> int:
        name, reg_no, role, resp, ag = self._clean(name, reg_no, role, responsibilities, agents)
        now = time.time()
        try:
            with self.conn() as c:
                cur = c.execute("INSERT INTO members(name,reg_no,role,responsibilities,agents,created_at,updated_at) "
                                "VALUES(?,?,?,?,?,?,?)", (name, reg_no, role, resp, ag, now, now))
                c.execute("INSERT INTO member_audit(ts,action,member) VALUES(?,?,?)", (now, "added", name))
                return cur.lastrowid
        except sqlite3.IntegrityError as e:
            if "member limit" in str(e):
                raise MemberLimitError(f"The group already has {MAX_MEMBERS} members.") from e
            if "unique" in str(e).lower():
                raise DuplicateMemberError("A member with that name already exists.") from e
            raise

    def update_member(self, member_id: int, name, reg_no, role, responsibilities, agents) -> None:
        name, reg_no, role, resp, ag = self._clean(name, reg_no, role, responsibilities, agents)
        try:
            with self.conn() as c:
                c.execute("UPDATE members SET name=?,reg_no=?,role=?,responsibilities=?,agents=?,updated_at=? WHERE id=?",
                          (name, reg_no, role, resp, ag, time.time(), member_id))
                c.execute("INSERT INTO member_audit(ts,action,member) VALUES(?,?,?)", (time.time(), "edited", name))
        except sqlite3.IntegrityError as e:
            raise DuplicateMemberError("A member with that name already exists.") from e

    def delete_member(self, member_id: int) -> None:
        with self.conn() as c:
            r = c.execute("SELECT name FROM members WHERE id=?", (member_id,)).fetchone()
            if r:
                c.execute("DELETE FROM members WHERE id=?", (member_id,))
                c.execute("INSERT INTO member_audit(ts,action,member) VALUES(?,?,?)", (time.time(), "deleted", r["name"]))

    def audit(self, limit: int = 10) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM member_audit ORDER BY id DESC LIMIT ?", (limit,))]

    def export_members(self) -> str:
        return json.dumps({"members": [{k: m[k] for k in ("name", "reg_no", "role", "responsibilities", "agents")}
                                       for m in self.list_members()]}, indent=2)

    def import_members(self, text: str) -> int:
        data = json.loads(text)
        n = 0
        for m in data.get("members", [])[:MAX_MEMBERS]:
            try:
                self.add_member(m["name"], m.get("reg_no", ""), m.get("role", ""), m["responsibilities"],
                                [a.strip() for a in m.get("agents", "").split(",") if a.strip()])
                n += 1
            except (MemberLimitError, DuplicateMemberError, ValueError, KeyError):
                continue
        return n

    # ------------------------------------------------------------ support agent memory
    def support_load(self) -> dict:
        with self.conn() as c:
            return {
                "examples": [(r["text"], r["intent"]) for r in c.execute("SELECT text,intent FROM support_examples")],
                "penalty": {r["intent"]: r["value"] for r in c.execute("SELECT intent,value FROM support_penalty")},
                "qa": [dict(question=r["question"], answer=r["answer"], sources=json.loads(r["sources"] or "[]"))
                       for r in c.execute("SELECT question,answer,sources FROM support_qa")],
                "topics": [json.loads(r["payload"]) for r in c.execute("SELECT payload FROM support_topics")],
                "tickets": [dict(r) for r in c.execute("SELECT * FROM support_tickets ORDER BY ts DESC LIMIT 200")],
            }

    def support_add_example(self, text: str, intent: str) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO support_examples(text,intent) VALUES(?,?)", (text, intent))

    def support_set_penalty(self, intent: str, value: float) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO support_penalty(intent,value) VALUES(?,?) ON CONFLICT(intent) DO UPDATE SET value=excluded.value",
                      (intent, value))

    def support_add_qa(self, question: str, answer: str, sources: list) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO support_qa(question,answer,sources,created_at) VALUES(?,?,?,?)",
                      (question, answer, json.dumps(sources), time.time()))

    def support_save_topic(self, topic: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO support_topics(id,payload) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                      (topic["id"], json.dumps(topic)))

    def support_add_ticket(self, t: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO support_tickets(id,ts,question,department,priority,status) VALUES(?,?,?,?,?,?)",
                      (t["id"], time.time(), t["question"], t["department"], t["priority"], t.get("status", "open")))

    def support_next_ticket_number(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0] + 1

    def support_reset(self) -> None:
        with self.conn() as c:
            for t in ("support_examples", "support_penalty", "support_qa"):
                c.execute(f"DELETE FROM {t}")

    # ------------------------------------------------------------ finance memory
    def fin_corrections(self) -> list[tuple[str, str]]:
        with self.conn() as c:
            return [(r["description"], r["category"]) for r in c.execute("SELECT description,category FROM fin_corrections")]

    def fin_add_correction(self, description: str, category: str) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO fin_corrections(description,category) VALUES(?,?)", (description, category))

    # ------------------------------------------------------------ products
    def products(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM products ORDER BY name")]

    # ------------------------------------------------------------ research cache
    def cache_get(self, key: str, max_age: float = 7 * 86400):
        with self.conn() as c:
            r = c.execute("SELECT payload,ts FROM research_cache WHERE key=?", (key,)).fetchone()
        if r and time.time() - r["ts"] <= max_age:
            return json.loads(r["payload"])
        return None

    def cache_put(self, key: str, kind: str, payload) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO research_cache(key,kind,payload,ts) VALUES(?,?,?,?) "
                      "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, ts=excluded.ts",
                      (key, kind, json.dumps(payload), time.time()))

    def cache_count(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM research_cache").fetchone()[0]

    # ------------------------------------------------------------ test runs
    def save_tests(self, rows: list[dict]) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM test_runs")
            c.executemany("INSERT INTO test_runs(ts,name,agent,passed,detail,seconds) VALUES(?,?,?,?,?,?)",
                          [(time.time(), r["name"], r["agent"], int(r["passed"]), r["detail"], r["seconds"]) for r in rows])

    def load_tests(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM test_runs ORDER BY id")]
