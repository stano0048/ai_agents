"""
research.py - lets the agents look things up instead of relying on a closed list.

Three layers, tried in order, and the result says which one produced it (`mode`):
  1. "gemini"  - Gemini with Google Search grounding (needs an API key). Best quality; cites the pages it used.
  2. "live"    - keyless public sources: Wikipedia (summary), Open Library (books) plus search links built from the topic.
  3. "offline" - no network: the topic is matched to a subject area by TF-IDF similarity and search links are built
                 from the unit name itself. It is never a fixed random list, but it cannot read the web.
Results are cached in the database so the same unit or question is not researched twice.
"""
from __future__ import annotations

import re
from urllib.parse import quote_plus, quote

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm import Gemini, extract_json

UA = {"User-Agent": "AI-Agents-Suite/1.0 (university group project)"}


def default_get(url, params=None):
    """GET returning parsed JSON, or None on any failure. Tests replace this."""
    try:
        r = requests.get(url, params=params, headers=UA, timeout=8)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ------------------------------------------------------------------ subject areas (offline matching)
AREAS = {
    "Programming and software": ("programming coding software code java python c++ algorithm data structure object oriented "
                                 "compiler system programming scripting development testing", [
        ("CS50 - Harvard's introduction to computer science", "https://cs50.harvard.edu", "Free lectures and problem sets"),
        ("freeCodeCamp", "https://www.freecodecamp.org", "Guided coding practice"),
        ("GeeksforGeeks", "https://www.geeksforgeeks.org", "Worked examples and past-paper style questions")]),
    "Computer systems": ("operating systems architecture kernel memory processes assembly embedded hardware computer organisation "
                         "system programming unix linux", [
        ("Operating Systems: Three Easy Pieces", "https://pages.cs.wisc.edu/~remzi/OSTEP/", "Free textbook"),
        ("Nand2Tetris", "https://www.nand2tetris.org", "Build a computer from first principles")]),
    "Databases": ("database sql dbms relational query normalisation data management nosql", [
        ("SQLBolt", "https://sqlbolt.com", "Interactive SQL lessons"),
        ("Stanford Databases (free course)", "https://online.stanford.edu/courses/soe-ydatabases0001-databases-relational-databases-and-sql", "Lectures")]),
    "Web and internet": ("web html css javascript php internet frontend backend http browser", [
        ("MDN Web Docs", "https://developer.mozilla.org", "The authoritative reference"),
        ("The Odin Project", "https://www.theodinproject.com", "Project-based web development")]),
    "Networks and security": ("network networking telecommunication protocol security cryptography forensic cyber ethical hacking", [
        ("Cisco Networking Academy", "https://www.netacad.com", "Networking courses"),
        ("OWASP", "https://owasp.org", "Security learning resources")]),
    "Mathematics and statistics": ("mathematics calculus algebra statistics discrete probability numerical analysis linear geometry", [
        ("Khan Academy", "https://www.khanacademy.org", "Free lessons and practice"),
        ("MIT OpenCourseWare", "https://ocw.mit.edu", "Lectures and problem sets")]),
    "AI and data": ("artificial intelligence machine learning data science analytics neural statistics mining", [
        ("Kaggle Learn", "https://www.kaggle.com/learn", "Short practical courses"),
        ("Google Machine Learning Crash Course", "https://developers.google.com/machine-learning/crash-course", "Free course")]),
    "Business and entrepreneurship": ("entrepreneurship business management marketing enterprise startup innovation leadership "
                                      "strategy commerce small business venture", [
        ("Khan Academy - Entrepreneurship", "https://www.khanacademy.org/college-careers-more/entrepreneurship2", "Free lessons"),
        ("Harvard Business Review", "https://hbr.org", "Case studies and articles"),
        ("MIT OpenCourseWare - Management", "https://ocw.mit.edu/search/?d=Sloan%20School%20of%20Management", "Lectures")]),
    "Accounting and finance": ("accounting finance economics financial banking audit tax investment budgeting", [
        ("Khan Academy - Finance and capital markets", "https://www.khanacademy.org/economics-finance-domain", "Free lessons"),
        ("Investopedia", "https://www.investopedia.com", "Definitions and worked examples")]),
    "Communication and writing": ("communication writing english language report academic writing presentation literature", [
        ("Purdue OWL", "https://owl.purdue.edu", "Writing and citation guides"),
        ("Coursera - Writing", "https://www.coursera.org/browse/language-learning", "Courses")]),
    "Project and software engineering": ("software engineering requirements analysis design uml project management agile testing "
                                         "devops systems analysis", [
        ("Refactoring Guru", "https://refactoring.guru", "Design patterns and principles"),
        ("Atlassian Agile Coach", "https://www.atlassian.com/agile", "Agile practice")]),
    "Science and engineering": ("physics chemistry biology engineering electronics circuits mechanics thermodynamics energy", [
        ("Khan Academy - Science", "https://www.khanacademy.org/science", "Free lessons"),
        ("OpenStax", "https://openstax.org/subjects", "Free textbooks")]),
    "Education and social sciences": ("education psychology sociology law history government politics community development", [
        ("OpenStax", "https://openstax.org/subjects", "Free textbooks"),
        ("Coursera - Social Sciences", "https://www.coursera.org/browse/social-sciences", "Courses")]),
}

ACTIVITIES = {
    "default": ["Read the unit outline and list every topic; tick off the ones you can already explain without notes.",
                "Work through two past papers under exam conditions and mark them against the scheme.",
                "Form a study group and teach each other one topic per week."],
    "Programming and software": ["Write code every day: solve one small problem, then rewrite it more cleanly.",
                                 "Build a small project that uses the unit's main concepts."],
    "Business and entrepreneurship": ["Write a one-page business plan for a real local idea and have a classmate challenge it.",
                                      "Analyse one local business: customers, costs, competitors, how it earns money."],
    "Computer systems": ["Trace small programs by hand through memory and the call stack.",
                         "Run the commands on a Linux machine and read the manual pages."],
    "Mathematics and statistics": ["Do ten practice problems a day and redo the ones you got wrong.",
                                   "Explain each formula in words, then with a worked example."],
}


def search_links(topic: str) -> list[dict]:
    """Links that open a live search for the topic. They are built from the unit name, so they always match it."""
    q = quote_plus(topic)
    return [
        {"title": f"Video lectures on {topic}", "type": "Video search", "url": f"https://www.youtube.com/results?search_query={q}+lecture+university",
         "why": "Lectures and tutorials from universities and teachers"},
        {"title": f"Free university courses on {topic}", "type": "Course search", "url": f"https://www.coursera.org/search?query={q}",
         "why": "Structured courses; many can be audited free"},
        {"title": f"Open textbooks and lecture notes on {topic}", "type": "Notes search", "url": f"https://ocw.mit.edu/search/?q={q}",
         "why": "MIT OpenCourseWare notes and problem sets"},
        {"title": f"Academic papers on {topic}", "type": "Paper search", "url": f"https://scholar.google.com/scholar?q={q}",
         "why": "Deeper reading and references for assignments"},
    ]


class Researcher:
    def __init__(self, db=None, gemini: Gemini | None = None, get=None, use_web: bool = True):
        self.db = db
        self.gemini = gemini or Gemini(db=db)
        self._get = get or default_get
        self.use_web = use_web          # False forces the offline layer (used by tests and the "offline" switch)

    # ------------------------------------------------------------------ keyless sources
    def wikipedia(self, topic: str) -> dict | None:
        if not self.use_web:
            return None
        s = self._get("https://en.wikipedia.org/w/api.php",
                      {"action": "opensearch", "search": topic, "limit": 1, "namespace": 0, "format": "json"})
        if not s or len(s) < 4 or not s[1]:
            return None
        title = s[1][0]
        d = self._get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_")))
        if not d or not d.get("extract") or d.get("type") == "disambiguation":
            return None
        return {"title": d.get("title", title), "extract": d["extract"],
                "url": (d.get("content_urls", {}).get("desktop", {}) or {}).get("page")
                       or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))}

    def books(self, topic: str, n: int = 3) -> list[dict]:
        if not self.use_web:
            return []
        d = self._get("https://openlibrary.org/search.json", {"q": topic, "limit": n, "fields": "title,author_name,key,first_publish_year"})
        out = []
        for b in (d or {}).get("docs", [])[:n]:
            if b.get("title") and b.get("key"):
                by = ", ".join(b.get("author_name", [])[:2])
                out.append({"title": b["title"] + (f" - {by}" if by else ""), "type": "Book", "url": "https://openlibrary.org" + b["key"],
                            "why": "Found in Open Library" + (f" (first published {b['first_publish_year']})" if b.get("first_publish_year") else "")})
        return out

    # ------------------------------------------------------------------ offline subject matching
    @staticmethod
    def match_area(text: str) -> tuple[str | None, float]:
        names = list(AREAS)
        docs = [AREAS[n][0] for n in names]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs + [text])
        sims = cosine_similarity(vec.transform([text]), vec.transform(docs))[0]
        i = int(sims.argmax())
        return (names[i], float(sims[i])) if sims[i] >= 0.06 else (None, float(sims[i]))

    # ------------------------------------------------------------------ units
    def research_unit(self, unit: str, force: bool = False) -> dict:
        unit = re.sub(r"\s+", " ", unit or "").strip()
        if len(unit) < 2:
            return {"unit": unit, "mode": "none", "summary": "", "resources": [], "activities": [], "sources": [], "note": "Enter a unit name."}
        key = "unit:" + unit.lower()
        if self.db and not force:
            hit = self.db.cache_get(key)
            if hit:
                return {**hit, "cached": True}
        res = self._unit_gemini(unit) or self._unit_live(unit) or self._unit_offline(unit)
        if self.db and res["mode"] in ("gemini", "live"):   # never cache offline results, so a later online run can upgrade them
            self.db.cache_put(key, "unit", res)
        return res

    def _unit_gemini(self, unit: str):
        if not self.gemini.available():
            return None
        prompt = (f"A university student needs to improve in the unit \"{unit}\". Search the web and recommend real, current learning "
                  "resources for this exact subject. Reply with JSON: {\"summary\": \"two sentences on what the unit covers\", "
                  "\"resources\": [{\"title\": \"...\", \"type\": \"Course|Book|Video|Notes|Practice\", \"url\": \"https://...\", "
                  "\"why\": \"one short reason\"}], \"activities\": [\"three concrete study activities\"]}. Give 5 resources, "
                  "prefer free ones, and only use URLs you found.")
        obj, raw = self.gemini.generate_json(prompt, "You are a careful university study advisor.", grounded=True, cache_key="unit:" + unit.lower())
        if not isinstance(obj, dict) or not obj.get("resources"):
            return None
        good = [r for r in obj["resources"] if isinstance(r, dict) and str(r.get("url", "")).startswith("http") and r.get("title")][:6]
        if not good:
            return None
        return {"unit": unit, "mode": "gemini", "summary": str(obj.get("summary", ""))[:600], "resources": [
            {"title": r["title"], "type": r.get("type", "Resource"), "url": r["url"], "why": r.get("why", "")} for r in good],
            "activities": [str(a) for a in (obj.get("activities") or [])][:4], "sources": raw["sources"][:6],
            "note": "Researched with Gemini and Google Search."}

    def _unit_live(self, unit: str):
        wiki = self.wikipedia(unit)
        books = self.books(unit)
        if not wiki and not books:
            return None
        area, _ = self.match_area(unit + " " + (wiki["extract"] if wiki else ""))
        resources = []
        if wiki:
            resources.append({"title": f"Overview: {wiki['title']}", "type": "Encyclopedia", "url": wiki["url"],
                              "why": "Start here for the key ideas and vocabulary"})
        resources += books + [{"title": t, "type": "Course", "url": u, "why": w} for t, u, w in AREAS.get(area, ("", []))[1]][:2]
        resources += search_links(unit)[:3]
        return {"unit": unit, "mode": "live", "summary": (wiki["extract"][:420] if wiki else ""), "resources": resources[:8],
                "activities": ACTIVITIES.get(area, []) + ACTIVITIES["default"],
                "sources": ([{"title": wiki["title"] + " (Wikipedia)", "url": wiki["url"]}] if wiki else []) + [{"title": "Open Library", "url": "https://openlibrary.org"}] * bool(books),
                "note": "Researched live from Wikipedia and Open Library (no Gemini key set)."}

    def _unit_offline(self, unit: str) -> dict:
        area, score = self.match_area(unit)
        resources = []
        if area:
            resources += [{"title": t, "type": "Course", "url": u, "why": w} for t, u, w in AREAS[area][1]]
        resources += search_links(unit)
        return {"unit": unit, "mode": "offline", "summary": f"Matched to the subject area: {area}." if area else "No subject area matched; using searches built from the unit name.",
                "resources": resources[:7], "activities": ACTIVITIES.get(area, []) + ACTIVITIES["default"], "sources": [],
                "note": "Offline: could not reach the web. The links open live searches for this exact unit name."}

    # ------------------------------------------------------------------ open questions (support agent)
    def answer_question(self, question: str, context: str = "supermarket customer service") -> dict | None:
        """Answer a question the stored knowledge cannot. Returns None when nothing trustworthy was found."""
        q = re.sub(r"\s+", " ", question or "").strip()
        if len(q) < 3:
            return None
        key = "q:" + q.lower()
        if self.db:
            hit = self.db.cache_get(key, max_age=2 * 86400)
            if hit:
                return {**hit, "cached": True}
        res = self._q_gemini(q, context) or self._q_wiki(q)
        if res and self.db:
            self.db.cache_put(key, "question", res)
        return res

    def answer_study_question(self, question: str, student_summary: str = "") -> dict | None:
        """An academic question the advisor's rules cannot answer. Gemini sees the student's results for context."""
        q = re.sub(r"\s+", " ", question or "").strip()
        if len(q) < 3:
            return None
        if self.gemini.available():
            system = ("You are a university academic advisor. Use the student's results only as context. Answer in under 150 words, "
                      "practically, and do not invent facts about the university. Politely decline questions unrelated to studying.")
            r = self.gemini.generate(f"Student context: {student_summary}\n\nQuestion: {q}", system, grounded=True)
            if r["ok"]:
                return {"answer": r["text"], "sources": r["sources"][:4], "mode": "gemini"}
        return self._q_wiki(q)

    def _q_gemini(self, q: str, context: str):
        if not self.gemini.available():
            return None
        system = (f"You are a helpful assistant for a {context} team. Answer briefly (under 120 words) and practically. "
                  "Search the web for general facts (food storage, nutrition, cooking, safety, recipes, how products are used). "
                  "You do NOT know this specific store's prices, stock, opening hours or policies: never invent them; say the store team "
                  "will confirm. If the question is unrelated to shopping, food, household or store services, politely say so.")
        r = self.gemini.generate(q, system, grounded=True, cache_key="q:" + q.lower())
        if not r["ok"]:
            return None
        return {"answer": r["text"], "sources": r["sources"][:4], "mode": "gemini"}

    def _q_wiki(self, q: str):
        terms = re.sub(r"\b(how|what|why|when|where|which|who|do|does|did|is|are|can|could|should|i|you|we|the|a|an|to|of|for|in|on|my|your|it|long|much|many|best|way)\b", " ", q.lower())
        terms = re.sub(r"[^a-z0-9 ]", " ", terms).split()
        if not terms:
            return None
        w = self.wikipedia(" ".join(terms[:4]))
        if not w:
            return None
        overlap = len(set(terms) & set(re.findall(r"[a-z0-9]+", (w["title"] + " " + w["extract"]).lower())))
        if overlap < max(1, min(2, len(terms))):
            return None
        return {"answer": w["extract"][:600], "sources": [{"title": w["title"] + " (Wikipedia)", "url": w["url"]}], "mode": "live"}
