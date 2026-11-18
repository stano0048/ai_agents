"""
support_engine.py
-----------------
NLP engine for the Supermarket Customer Support Agent (Option 2).

For every customer message
  1. NLP pre-processing    - normalisation, synonym folding, entity extraction, urgency / safety / frustration cues
  2. Intent classification - TF-IDF (word + character n-grams) + Logistic Regression
  3. Retrieval             - TF-IDF cosine similarity over the knowledge base
  4. Product lookup        - the live product catalogue in the database (price, offer, stock, aisle)
  5. Decision              - answer / catalogue / clarify / research / escalate
  6. Research (NOT closed) - when the knowledge base has nothing reliable, the agent researches the question
                             (Gemini with Google Search if a key is set, otherwise Wikipedia) and cites its sources
  7. Learning              - thumbs up/down and corrections become training examples; helpful researched answers are
                             remembered; everything is saved in the database

The knowledge base is SAMPLE content for a fictional store. Replace the policies with the real store's.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

STORE = "the supermarket"
ANSWER_T = 0.42      # confident answer
CLARIFY_T = 0.24     # "did you mean ...?" between this and ANSWER_T
HUMAN_INTENT = "human_agent"
OOS_INTENT = "out_of_scope"
SMALLTALK = {"greeting", "goodbye", "thanks"}
PRODUCT_INTENTS = {"product_price", "product_availability", "product_location"}

# --------------------------------------------------------------------------
# Knowledge base (sample content)
# --------------------------------------------------------------------------
KB: list[dict] = [
    dict(id="greeting", title="Greeting", category="Small talk",
         answer="Hello! I am the store's customer assistant. I can help with prices, stock, offers, delivery, payments, "
                "loyalty points, returns and more. What do you need?",
         examples=["hello", "hi", "hi there", "good morning", "good afternoon", "good evening", "hey", "hello I need help",
                   "hi store", "habari", "greetings", "hello team", "hey there"],
         followups=["What are today's offers?", "Do you deliver?"]),
    dict(id="goodbye", title="Goodbye", category="Small talk",
         answer="Thank you for shopping with us. Have a lovely day!",
         examples=["bye", "goodbye", "see you later", "that is all", "i am done", "no more questions", "talk later",
                   "thats all for now", "cheers bye", "ok bye"],
         followups=[]),
    dict(id="thanks", title="Thanks", category="Small talk",
         answer="You are welcome. Is there anything else I can help you with?",
         examples=["thank you", "thanks", "thanks a lot", "i appreciate it", "that was helpful", "great thanks",
                   "many thanks", "awesome thank you", "cool thanks", "thank you very much"],
         followups=[]),
    dict(id="human_agent", title="Speak to a staff member", category="Escalation",
         answer="I will pass this to a staff member. I have logged a ticket with your conversation so you do not need to "
                "repeat yourself. Someone from the team will contact you.",
         examples=["i want to talk to a human", "speak to an agent", "connect me to a staff member", "i need a real person",
                   "escalate my issue", "call me please", "let me speak to the manager", "transfer me to someone",
                   "human please", "i want to speak to customer care"],
         followups=[]),
    dict(id="opening_hours", title="Opening hours", category="Store",
         answer="**Opening hours (sample)**\n- Monday to Saturday: 7:00 am to 10:00 pm\n- Sunday and public holidays: 8:00 am to 8:00 pm\n"
                "- The pharmacy counter and deli close one hour earlier.\n- Hours may change on holidays; the notice at the entrance is final.",
         examples=["what time do you open", "what time do you close", "opening hours", "are you open on sunday",
                   "are you open today", "do you open on public holidays", "closing time", "store hours",
                   "when do you close on saturday", "are you open late at night"],
         followups=["Where is the nearest branch?"]),
    dict(id="store_locations", title="Branches, parking and directions", category="Store",
         answer="**Our branches**\n- Branch addresses and maps are on the website and at the Customer Service Desk.\n"
                "- Free parking is available for customers; spend the minimum amount shown at the exit to have a parking ticket validated.\n"
                "- Ask the guard at the gate for trolleys and help carrying goods to your car.",
         examples=["where is your nearest branch", "store location", "where are you located", "do you have parking",
                   "directions to the supermarket", "is there parking at the store", "address of the shop",
                   "how many branches do you have", "which towns are you in", "where can i park"],
         followups=["What are your opening hours?"]),
    dict(id="payment_methods", title="Payment methods", category="Payments",
         answer="**Ways to pay**\n- Cash, M-Pesa (Lipa na M-Pesa till), Visa and Mastercard are accepted at all tills.\n"
                "- Loyalty points can be redeemed at the till.\n- Online orders can be paid by M-Pesa or card at checkout.\n"
                "- Always keep your receipt or M-Pesa confirmation message.",
         examples=["how can i pay", "do you accept mpesa", "can i pay with card", "payment methods at the till",
                   "do you take visa", "pay using mpesa till number", "is cash accepted", "can i pay with airtel money",
                   "do you accept credit cards", "paybill number for payments"],
         followups=["How do loyalty points work?"]),
    dict(id="loyalty_card", title="Loyalty card and points", category="Loyalty",
         answer="**Loyalty programme (sample)**\n- Register for a loyalty card at the Customer Service Desk with your ID and phone number.\n"
                "- Earn 1 point for every KES 100 spent; 100 points are worth KES 100 at the till.\n"
                "- Check your balance with the store app or at any till. Points expire after 12 months.",
         examples=["how do i get a loyalty card", "how do loyalty points work", "check my points balance", "redeem my points",
                   "join the rewards programme", "i lost my loyalty card", "how many points do i have",
                   "do points expire", "register for the card", "rewards for shopping"],
         followups=["What are today's offers?"]),
    dict(id="returns_refunds", title="Returns and refunds", category="Customer care",
         answer="**Returns and refunds (sample policy)**\n- Bring the item, unopened where possible, with your receipt within 7 days.\n"
                "- Faulty, damaged or expired goods are refunded or replaced.\n- Fresh produce, meat and opened hygiene products are not returnable unless faulty.\n"
                "- Refunds go back to the original payment method; M-Pesa refunds take about 24 hours.",
         examples=["what is your return policy", "can i return an item", "i want a refund", "how do i get my money back",
                   "exchange a product i bought", "return goods without a receipt", "refund for a damaged item",
                   "can i bring back opened items", "how long do i have to return", "replace a faulty product"],
         followups=["I want to make a complaint"]),
    dict(id="delivery", title="Online shopping and delivery", category="Online",
         answer="**Delivery (sample)**\n- Order on the website or app, or by phone, and choose a delivery slot.\n"
                "- Free delivery over KES 3,000 within the delivery zone; otherwise a small fee applies.\n"
                "- Same-day delivery for orders placed before 2:00 pm. You can pay on delivery with M-Pesa.",
         examples=["do you deliver", "how does delivery work", "delivery charges", "can i order groceries online",
                   "do you deliver to my area", "same day delivery", "how much is the delivery fee",
                   "order by phone and deliver", "home delivery of shopping", "delivery time slots"],
         followups=["Where is my order?"]),
    dict(id="order_status", title="Where is my order", category="Online",
         answer="**Tracking an order**\n- Use the link in your order confirmation or open **My orders** in the app.\n"
                "- Delivery windows are two hours long; the rider calls before arriving.\n"
                "- If your order is late by more than one hour, tell me your order number (for example ORD-12345) and I will alert the delivery team.",
         examples=["where is my order", "track my order", "my delivery has not arrived", "my groceries are late",
                   "order status", "when will my order arrive", "rider has not come", "delivery is delayed",
                   "i ordered yesterday and nothing came", "cancel my online order"],
         followups=["I want to talk to a human"]),
    dict(id="offers_promotions", title="Offers and promotions", category="Offers",
         answer="**Offers**\n- Weekly offers are on the shelf tags, the flyer and the app. Offer prices are shown below from the catalogue.\n"
                "- Loyalty members get extra discounts on selected products.\n- Offers are while stocks last.",
         examples=["what offers do you have", "any discounts this week", "special offers", "promotions today",
                   "what is on sale", "weekly deals", "any specials", "cheap deals this week", "discount on groceries",
                   "what is on promotion"],
         followups=["How do loyalty points work?"]),
    dict(id="product_price", title="Product price", category="Products",
         answer="Tell me the product name (for example *cooking oil 1L*) and I will check the price in the catalogue.",
         examples=["how much is sugar", "price of cooking oil", "what does milk cost", "cost of a loaf of bread",
                   "how much is a kilo of rice", "what is the price of eggs", "price of unga", "how much for tomatoes",
                   "how much does it cost", "what is the price"],
         followups=["Is it in stock?"]),
    dict(id="product_availability", title="Product availability", category="Products",
         answer="Tell me the product name and I will check whether it is in stock.",
         examples=["do you have sugar", "is rice in stock", "do you sell fresh fish", "is milk available",
                   "do you stock baby formula", "are tomatoes available today", "do you have this item",
                   "is it out of stock", "when will it be back in stock", "do you sell gas refills"],
         followups=["Where can I find it?"]),
    dict(id="product_location", title="Where to find a product", category="Products",
         answer="Tell me the product name and I will tell you which aisle it is in.",
         examples=["where can i find flour", "which aisle is the cooking oil", "where is the bakery section",
                   "where do i find baby products", "i cannot find the rice", "what aisle are the drinks in",
                   "where are the fresh vegetables", "direct me to the dairy section", "where is the butcher", "location of soap"],
         followups=["What is the price?"]),
    dict(id="complaint", title="Complaint about a product or service", category="Customer care",
         answer="I am sorry about that. Please share what happened, the date and the branch, and your receipt number if you have it. "
                "I have logged a complaint so the store manager can follow up with you.",
         examples=["i was overcharged at the till", "the product was expired", "i bought a damaged item",
                   "the cashier was rude", "i want to make a complaint", "the price on the shelf is different from the till",
                   "i was charged twice", "my food was spoilt", "poor customer service at your branch", "wrong change given"],
         followups=["I want to talk to a human"]),
    dict(id="receipts_invoices", title="Receipts and invoices", category="Payments",
         answer="**Receipts and invoices**\n- Reprint a receipt at the Customer Service Desk within 30 days using the card, M-Pesa message or loyalty number.\n"
                "- For a tax invoice with your KRA PIN, give the PIN at the till before payment or visit the Accounts desk within 7 days.",
         examples=["i lost my receipt", "reprint my receipt", "i need a tax invoice", "invoice with kra pin",
                   "receipt for my purchase", "copy of my receipt", "vat invoice for my business", "how do i get an invoice",
                   "proof of purchase", "email me my receipt"],
         followups=["What is your return policy?"]),
    dict(id="gift_cards", title="Gift cards and vouchers", category="Payments",
         answer="**Gift cards**\n- Gift cards from KES 500 can be bought at the Customer Service Desk and used at any branch.\n"
                "- They are valid for 12 months and cannot be exchanged for cash.\n- Corporate gift cards need a letter on company letterhead.",
         examples=["do you sell gift cards", "buy a voucher", "how do i use my gift card", "check gift card balance",
                   "corporate gift vouchers", "voucher for a present", "does the gift card expire", "gift card for christmas",
                   "where do i buy a shopping voucher", "can i get a gift hamper"],
         followups=["What are your payment methods?"]),
    dict(id="bulk_orders", title="Bulk and wholesale orders", category="Sales",
         answer="**Bulk orders (sample)**\n- Hotels, schools, caterers and events can order in bulk at discounted prices.\n"
                "- Send the product list and delivery date to the Sales team; they reply with a quotation within one working day.\n"
                "- A deposit may be required for large orders.",
         examples=["do you do bulk orders", "wholesale prices", "i need groceries for an event", "catering supplies in large quantities",
                   "buy in bulk for my restaurant", "discount for large orders", "supplies for a school", "order for a wedding",
                   "quotation for bulk purchase", "carton prices"],
         followups=["Do you deliver?"]),
    dict(id="careers", title="Jobs and careers", category="Company",
         answer="**Careers**\n- Vacancies are published on the website careers page and on the notice board at each branch.\n"
                "- Apply with your CV and copies of your certificates; shortlisted candidates are contacted by phone.\n"
                "- We do not charge any fee for applications.",
         examples=["are you hiring", "how do i apply for a job", "job vacancies", "send my cv", "work at your supermarket",
                   "cashier jobs", "internship opportunities", "recruitment", "do you have any vacancies", "careers page"],
         followups=[]),
    dict(id="supplier_registration", title="Becoming a supplier", category="Company",
         answer="**Supplying the store**\n- Email the Procurement team your company profile, product list, price list and certificates "
                "(business registration, KEBS or food safety licences where they apply).\n- Samples are reviewed monthly.\n"
                "- Local farmers and small producers are welcome.",
         examples=["how do i become a supplier", "i want to supply my products", "supplier registration", "sell my products in your stores",
                   "procurement contact", "i am a farmer and want to supply", "list my product in your shop",
                   "vendor application", "how do suppliers get paid", "supply vegetables to the supermarket"],
         followups=[]),
    dict(id="food_safety", title="Allergens, ingredients and food safety", category="Quality",
         answer="**Food safety and allergens**\n- Ingredients and allergens are listed on the product label; check it before you buy.\n"
                "- If you have a serious allergy, ask the Customer Service Desk for the supplier's details.\n"
                "- Recalled products are announced at the entrance and on the website; bring any recalled item back for a full refund.\n"
                "- If you feel unwell after eating a product, keep the packaging and contact us right away.",
         examples=["does this contain nuts", "is this product halal", "allergen information", "ingredients in the biscuits",
                   "gluten free products", "is there a product recall", "expiry date on the label", "is it safe for diabetics",
                   "i got sick after eating your product", "lactose free milk"],
         followups=["I want to make a complaint"]),
    dict(id="out_of_scope", title="Out of scope", category="Fallback",
         answer="That is outside what I can help with. I support questions about shopping with the supermarket.",
         examples=["what is the weather today", "tell me a joke", "who won the football match", "bitcoin price",
                   "recommend a movie", "what is the capital of france", "sing a song", "best phone to buy",
                   "latest news", "what is your favourite colour", "write me a poem", "who is the president of kenya"],
         followups=[]),
]

EXTRA_EXAMPLES = {
    "opening_hours": ["hours", "open", "closing", "what time do you close today"],
    "store_locations": ["location", "branches", "parking", "where is the shop"],
    "payment_methods": ["mpesa", "pay", "till number", "card payment"],
    "loyalty_card": ["points", "loyalty", "rewards", "loyalty card"],
    "returns_refunds": ["refund", "return", "exchange", "money back"],
    "delivery": ["delivery", "deliver", "online order"],
    "order_status": ["track order", "late order", "where is my delivery"],
    "offers_promotions": ["offers", "discounts", "promo", "deals"],
    "product_price": ["price", "how much", "cost"],
    "product_availability": ["in stock", "available", "do you sell"],
    "product_location": ["aisle", "where is", "where do i find"],
    "complaint": ["complaint", "complain", "overcharged", "expired"],
    "receipts_invoices": ["receipt", "invoice", "kra pin"],
    "gift_cards": ["gift card", "voucher"],
    "bulk_orders": ["bulk", "wholesale", "event catering"],
    "careers": ["jobs", "vacancy", "hiring"],
    "supplier_registration": ["supplier", "vendor", "supply"],
    "food_safety": ["allergen", "halal", "ingredients", "recall"],
    "human_agent": ["human", "agent", "manager", "speak to a person"],
}
for _a in KB:
    _a["examples"] = _a["examples"] + EXTRA_EXAMPLES.get(_a["id"], [])

DEPARTMENT = {
    "opening_hours": "Customer Service Desk", "store_locations": "Customer Service Desk", "payment_methods": "Front-end Tills",
    "loyalty_card": "Loyalty Desk", "returns_refunds": "Customer Service Desk", "delivery": "Online Orders and Delivery",
    "order_status": "Online Orders and Delivery", "offers_promotions": "Marketing", "product_price": "Store Floor Team",
    "product_availability": "Store Floor Team", "product_location": "Store Floor Team", "complaint": "Store Manager",
    "receipts_invoices": "Accounts", "gift_cards": "Customer Service Desk", "bulk_orders": "Sales and Wholesale",
    "careers": "Human Resources", "supplier_registration": "Procurement", "food_safety": "Quality and Safety",
    "human_agent": "Customer Service Desk",
}
URGENT_RE = re.compile(r"\b(urgent|asap|immediately|today|tomorrow|tonight|deadline|emergency|right now|expired|spoilt|rotten)\b", re.I)
SAFETY_RE = re.compile(r"\b(sick|food poisoning|poisoned|vomit\w*|allergic reaction|allergy attack|hospital|contaminated|recall|expired|rotten)\b", re.I)

SAMPLE_INBOX = [
    "Hello, do you deliver to Kasarani?",
    "I bought milk this morning and it is already expired, I feel sick. This is urgent!",
    "How much is cooking oil 1L?",
    "My online order ORD-48213 has not arrived and it is 3 hours late",
    "I was charged twice at the till for the same bread, this is useless",
    "Do you have quinoa in stock?",
    "Is lactose free milk safe for a 2 year old baby?",
    "How long can I keep cooked rice in the fridge?",
    "Who will win the premier league this year?",
    "thanks",
]

# Fresh phrasings that were NOT used while building the training data (kept for honest evaluation).
HELD_OUT = [
    ("what time do you shut on sundays", "opening_hours"),
    ("is there somewhere to park at your branch", "store_locations"),
    ("can i swipe my visa at the till", "payment_methods"),
    ("how do i earn points when i shop", "loyalty_card"),
    ("can i bring back an item i bought if i changed my mind", "returns_refunds"),
    ("will you bring groceries to my house", "delivery"),
    ("my online shopping has not arrived yet", "order_status"),
    ("any specials this week", "offers_promotions"),
    ("how much is a loaf of brown bread", "product_price"),
    ("do you still have eggs", "product_availability"),
    ("which aisle has the washing powder", "product_location"),
    ("the cashier charged me for the same item twice", "complaint"),
    ("i need a tax invoice for my purchase", "receipts_invoices"),
    ("do you sell vouchers i can give as presents", "gift_cards"),
    ("i am organising a wedding and need food in large quantities", "bulk_orders"),
    ("are you looking for shelf stackers", "careers"),
    ("i want to supply tomatoes to your stores", "supplier_registration"),
    ("does the bread contain nuts", "food_safety"),
    ("good evening", "greeting"),
    ("who is the president of brazil", "out_of_scope"),
]

SYNONYMS = {
    "m-pesa": "mpesa", "fees": "fee", "supermarket": "store", "shop": "store", "discounts": "offer", "discount": "offer",
    "offers": "offer", "promo": "offer", "promotion": "offer", "promotions": "offer", "specials": "offer", "special": "offer",
    "deals": "offer", "deal": "offer", "refunds": "refund", "vouchers": "voucher", "groceries": "grocery", "delivered": "delivery",
    "deliver": "delivery", "delivers": "delivery", "cheaper": "cheap",
}
FRUSTRATION = {"useless", "frustrated", "angry", "terrible", "worst", "waste", "annoyed", "ridiculous", "stupid",
               "nonsense", "unacceptable", "fed up", "not helping", "still not", "again and again", "pathetic", "rude"}
ORDER_RE = re.compile(r"\b(?:ORD|INV|TKT)[- ]?\d{3,8}\b", re.I)
MONEY_RE = re.compile(r"\b(?:kes|ksh|kshs)\.?\s?[\d,]+(?:\.\d+)?\b|\b[\d,]{3,}\s?(?:kes|ksh|kshs)\b", re.I)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?254|0)7\d{8}\b")

PRODUCT_CUE_RE = re.compile(r"\b(price|cost|how much|stock|have|sell|stock|available|aisle|where|buy|offer|cheap)\b", re.I)
STOP = {"the", "a", "an", "do", "does", "you", "your", "have", "has", "is", "are", "in", "stock", "how", "much", "what", "price", "of",
        "for", "to", "i", "me", "my", "any", "some", "where", "can", "find", "which", "aisle", "cost", "sell", "still", "get", "buy",
        "available", "it", "this", "that", "and", "or", "on", "at", "with", "please", "there", "much", "many", "kg", "kilo", "litre"}
GENERIC = {"fresh", "white", "brown", "whole", "instant", "bar", "tray", "each", "pack", "bunch", "refill", "leaves", "with", "bone"}


def normalise(text: str) -> str:
    t = text.lower().replace("can't", "cannot").replace("won't", "will not").replace("don't", "do not")
    t = re.sub(r"[^a-z0-9/\s\-']", " ", t)
    words = [SYNONYMS.get(w, w) for w in t.split()]
    return " ".join(words)


def extract_entities(text: str) -> dict:
    ents = {}
    if m := ORDER_RE.findall(text):
        ents["order_number"] = m[0].upper()
    if m := MONEY_RE.findall(text):
        ents["amount"] = m[0].strip()
    if m := DATE_RE.findall(text):
        ents["date"] = m[0]
    if m := PHONE_RE.findall(text):
        ents["phone"] = m[0]
    return ents


def is_frustrated(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in FRUSTRATION) or ("!!" in text)


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower())]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class SupportBot:
    """db: Database for memory (optional). researcher: Researcher for open questions (optional)."""

    def __init__(self, db=None, researcher=None):
        self.db = db
        self.researcher = researcher
        self.kb: list[dict] = [dict(a) for a in KB]
        self.extra: list[tuple[str, str]] = []          # learned (text, intent)
        self.penalty: dict[str, float] = {}
        self.qa: list[dict] = []                         # approved researched answers
        self.log: list[dict] = []
        self.tickets: list[dict] = []
        self._load()
        self.train()

    # ---------------- persistence (database)
    def _load(self):
        if not self.db:
            return
        try:
            m = self.db.support_load()
            self.extra = [tuple(x) for x in m["examples"]]
            self.penalty = m["penalty"]
            self.qa = m["qa"]
            for a in m["topics"]:
                if all(a["id"] != k["id"] for k in self.kb):
                    self.kb.insert(len(self.kb) - 1, a)
            self.tickets = [dict(t, transcript=[]) for t in m["tickets"]]
        except Exception:
            pass

    def reset_learning(self):
        self.extra.clear(); self.penalty.clear(); self.qa.clear()
        if self.db:
            self.db.support_reset()
        self.train()

    # ---------------- training
    def _training_data(self):
        X, y = [], []
        for a in self.kb:
            for ex in a["examples"] + [a["title"]]:
                X.append(normalise(ex)); y.append(a["id"])
        for t, i in self.extra:
            X.append(normalise(t)); y.append(i)
        return X, y

    def _pipeline(self):
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        return Pipeline([("f", feats), ("c", LogisticRegression(C=40, max_iter=2000))])

    def train(self):
        X, y = self._training_data()
        self.clf = self._pipeline().fit(X, y)
        self.classes = list(self.clf.classes_)
        docs = [normalise(" ".join([a["title"], " ".join(a["examples"]), a["answer"]])) for a in self.kb]
        self.ret = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(docs)
        self.doc_matrix = self.ret.transform(docs)
        self.ids = [a["id"] for a in self.kb]
        self._qa_vec = None
        if self.qa:
            self._qa_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([normalise(q["question"]) for q in self.qa])
            self._qa_mat = self._qa_vec.transform([normalise(q["question"]) for q in self.qa])

    def evaluate(self) -> dict:
        """Cross-validated accuracy of the intent classifier on the training utterances."""
        X, y = self._training_data()
        counts = pd.Series(y).value_counts()
        k = int(min(5, counts.min()))
        if k < 2:
            return {"accuracy": None, "n": len(X), "intents": len(counts)}
        scores = cross_val_score(self._pipeline(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=1))
        return {"accuracy": float(scores.mean()), "n": len(X), "intents": int(len(counts))}

    # ---------------- scoring
    def score(self, text: str) -> list[tuple[str, float, float, float]]:
        """Return [(intent, blended, p_intent, cosine)] sorted best first."""
        q = normalise(text)
        probs = dict(zip(self.classes, self.clf.predict_proba([q])[0]))
        sims = linear_kernel(self.ret.transform([q]), self.doc_matrix)[0]
        sim_by = dict(zip(self.ids, sims))
        out = []
        for i in self.ids:
            p, c = probs.get(i, 0.0), float(sim_by[i])
            s = 0.6 * p + 0.4 * min(1.0, c * 2.0) - self.penalty.get(i, 0.0)
            out.append((i, s, p, c))
        return sorted(out, key=lambda r: -r[1])

    def article(self, intent: str) -> dict:
        return next(a for a in self.kb if a["id"] == intent)

    # ---------------- product catalogue (live from the database)
    def products(self) -> list[dict]:
        if not self.db:
            return []
        return self.db.products()

    def find_products(self, text: str, limit: int = 3) -> list[dict]:
        """Match product names / categories in the message. Needs a distinctive word in common with the product name."""
        prods = self.products()
        if not prods:
            return []
        q = {t for t in _tokens(text) if t not in STOP and len(t) >= 3}
        if not q:
            return []
        scored = []
        for p in prods:
            name_t = [t for t in _tokens(p["name"]) if t not in GENERIC and len(t) >= 3]
            if not name_t:
                continue
            hit = len(q & set(name_t))
            if hit:
                scored.append((hit / min(len(name_t), 2), p))
        if not scored:
            cat = [p for p in prods if _stem(p["category"].lower().split()[0]) in q]
            return cat[:4]
        scored.sort(key=lambda r: -r[0])
        best = scored[0][0]
        return [p for s, p in scored if s >= max(0.5, best - 1e-9)][:limit]

    @staticmethod
    def product_line(p: dict) -> str:
        price = f"KES {p['price']:,.0f}"
        if p.get("offer_price"):
            price = f"KES {p['offer_price']:,.0f} (offer, normally KES {p['price']:,.0f})"
        stock = ("**out of stock**" if p["stock"] <= 0 else "low stock" if p["stock"] < 30 else "in stock")
        aisle = p["aisle"] if str(p["aisle"]).lower().startswith("customer") else f"aisle {p['aisle']}"
        return f"- **{p['name']}** ({p['unit']}): {price}, {stock}, {aisle}"

    def catalogue_reply(self, items: list[dict]) -> str:
        out = [self.product_line(p) for p in items]
        txt = "Here is what the catalogue shows:\n" + "\n".join(out)
        if any(p["stock"] <= 0 for p in items):
            txt += "\n\nFor an item that is out of stock, the Customer Service Desk can tell you when it is expected back."
        return txt + "\n\n*Prices and stock are from the store catalogue and can change during the day.*"

    # ---------------- learned Q&A
    def _learned_answer(self, text: str):
        if not self.qa or self._qa_vec is None:
            return None
        sims = linear_kernel(self._qa_vec.transform([normalise(text)]), self._qa_mat)[0]
        i = int(np.argmax(sims))
        return (self.qa[i], float(sims[i])) if sims[i] >= 0.7 else None

    # ---------------- main entry point
    def respond(self, text: str, history: list[dict] | None = None, research: bool = True) -> dict:
        history = history or []
        ranked = self.score(text)
        context_used = False
        effective = text

        # short follow-up with weak signal -> blend with the previous question
        clearly_off_topic = ranked[0][0] == OOS_INTENT and ranked[0][1] >= 0.15
        if ranked[0][1] < ANSWER_T and len(text.split()) <= 4 and not clearly_off_topic:
            prev = next((m for m in reversed(history) if m["role"] == "user"), None)
            if prev:
                combo = prev["text"] + " " + text
                r2 = self.score(combo)
                if (r2[0][1] > ranked[0][1] + 0.05 and r2[0][1] >= CLARIFY_T
                        and r2[0][0] not in SMALLTALK and r2[0][0] != OOS_INTENT):
                    ranked, effective, context_used = r2, combo, True

        top_id, top_score, p, c = ranked[0]
        art = self.article(top_id)
        frustrated = is_frustrated(text)
        cands = [{"intent": i, "title": self.article(i)["title"], "score": round(s, 3)}
                 for i, s, _, _ in ranked if i not in SMALLTALK and i != OOS_INTENT][:3]

        if top_id == OOS_INTENT:
            status = "fallback"
        elif top_id == HUMAN_INTENT:
            status = "escalate"
        elif top_id in SMALLTALK and top_score >= CLARIFY_T:
            status = "smalltalk"
        elif top_score >= ANSWER_T:
            status = "answered"
        elif top_score >= CLARIFY_T:
            status = "clarify"
        else:
            status = "fallback"

        reply, followups, sources, mode = "", [], [], ""
        found = []
        safety = bool(SAFETY_RE.search(text))
        if safety and status not in ("escalate", "smalltalk"):     # food-safety words always go to a person, never to research
            top_id, status = "complaint", "answered"
            art = self.article(top_id)

        # product lookup: a product question, or nothing else in the knowledge base fits
        if status not in ("escalate", "smalltalk") and top_id != "complaint":
            found = self.find_products(effective)
            product_q = top_id in PRODUCT_INTENTS and top_score >= CLARIFY_T
            cue = bool(PRODUCT_CUE_RE.search(effective)) or len(effective.split()) <= 3
            if found and (product_q or (status in ("fallback", "clarify") and cue)):
                status, mode = "catalogue", "catalogue"
                reply = self.catalogue_reply(found)
                followups = ["Where can I find it?", "What are today's offers?"]
                top_id = top_id if top_id in PRODUCT_INTENTS else "product_availability"
                art = self.article(top_id)

        if status in ("answered", "smalltalk"):
            reply, followups = art["answer"], art.get("followups", [])
            if top_id == "offers_promotions":
                on_offer = [p_ for p_ in self.products() if p_.get("offer_price")][:6]
                if on_offer:
                    reply += "\n\n**On offer now**\n" + "\n".join(self.product_line(x) for x in on_offer)
        elif status == "escalate":
            reply = art["answer"]
        elif status == "clarify":
            reply = "I am not completely sure what you mean. Did you want help with one of these?"
        elif status == "fallback":
            reply = ""

        # not closed: learned answers first, then research
        if status in ("fallback", "clarify") and research:
            learned = self._learned_answer(effective)
            if learned:
                status, mode, reply, sources = "learned", "learned", learned[0]["answer"], learned[0].get("sources", [])
            elif self.researcher and (status == "fallback"):
                try:
                    got = self.researcher.answer_question(effective)
                except Exception:
                    got = None
                if got and got.get("answer"):
                    status, mode, reply, sources = "researched", got["mode"], got["answer"], got.get("sources", [])
        if status == "fallback":
            reply = ("I could not find a reliable answer to that, and I would rather not guess. "
                     "Try rephrasing with a keyword (for example *delivery*, *price of sugar*, *refund*), pick a topic below, "
                     "or ask for a staff member.")
        if frustrated and status != "escalate":
            reply = "I am sorry for the trouble. " + reply
            if status in ("fallback", "clarify"):
                reply += "\n\nIf you prefer, type **human** and I will pass this to a staff member."

        ticket = None
        if status == "escalate" or top_id == "complaint" and status == "answered" or (frustrated and status == "fallback") or safety:
            dept = "Quality and Safety" if safety else DEPARTMENT.get(top_id, "Customer Service Desk")
            ticket = self._ticket(text, history, dept, "High" if (frustrated or safety or URGENT_RE.search(text)) else "Normal")
            reply += f"\n\nTicket **{ticket['id']}** has been logged for {ticket['department']} (simulated in this demo)."

        entry = {"ts": time.time(), "text": text, "intent": top_id, "confidence": round(top_score, 3),
                 "status": status, "feedback": 0, "context_used": context_used, "mode": mode, "reply": reply, "sources": sources}
        self.log.append(entry)
        return {"text": reply, "intent": top_id, "title": art["title"], "confidence": top_score, "status": status,
                "candidates": cands, "entities": extract_entities(text), "sentiment": "frustrated" if frustrated else "neutral",
                "context_used": context_used, "followups": followups, "query": effective, "mode": mode, "sources": sources,
                "products": found, "retrieval": c, "classifier": p, "ticket": ticket, "log_index": len(self.log) - 1}

    def answer_for(self, intent: str) -> dict:
        a = self.article(intent)
        return {"text": a["answer"], "intent": intent, "title": a["title"], "followups": a.get("followups", [])}

    def _ticket(self, text, history, department="Customer Service Desk", priority="Normal"):
        n = (self.db.support_next_ticket_number() if self.db else len(self.tickets) + 1)
        t = {"id": f"TKT-{n:04d}", "question": text, "department": department, "priority": priority, "status": "open",
             "transcript": [f"{m['role']}: {m['text']}" for m in history[-6:]]}
        self.tickets.append(t)
        if self.db:
            self.db.support_add_ticket(t)
        return t

    # ---------------- learning from feedback
    def feedback(self, log_index: int, helpful: bool):
        """Thumbs up/down. Up = learn the phrasing (or remember a researched answer), down = lower that topic's score."""
        e = self.log[log_index]
        e["feedback"] = 1 if helpful else -1
        i = e["intent"]
        if helpful:
            if e["status"] == "answered" and (e["text"].lower(), i) not in self.extra:
                self.extra.append((e["text"], i))
                if self.db:
                    self.db.support_add_example(e["text"], i)
                self.train()
            elif e["status"] == "researched" and e.get("reply"):
                self.qa.append({"question": e["text"], "answer": e["reply"].split("\n\nTicket")[0], "sources": e.get("sources", [])})
                if self.db:
                    self.db.support_add_qa(e["text"], self.qa[-1]["answer"], self.qa[-1]["sources"])
                self.train()
            self.penalty[i] = max(0.0, self.penalty.get(i, 0.0) - 0.02)
        elif e["status"] in ("answered", "clarify"):
            self.penalty[i] = min(0.2, self.penalty.get(i, 0.0) + 0.04)
        if self.db:
            self.db.support_set_penalty(i, self.penalty.get(i, 0.0))

    def correct(self, log_index: int, right_intent: str):
        """The customer tells the agent what they really meant -> new labelled training example."""
        e = self.log[log_index]
        self.extra.append((e["text"], right_intent))
        e["feedback"] = 1
        e["intent_corrected"] = right_intent
        self.penalty[right_intent] = max(0.0, self.penalty.get(right_intent, 0.0) - 0.04)
        if self.db:
            self.db.support_add_example(e["text"], right_intent)
            self.db.support_set_penalty(right_intent, self.penalty[right_intent])
        self.train()

    def add_article(self, title: str, answer: str, examples: list[str], category: str = "Custom") -> str:
        base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "custom"
        aid, n = base, 2
        while any(a["id"] == aid for a in self.kb):
            aid, n = f"{base}_{n}", n + 1
        art = dict(id=aid, title=title, category=category, answer=answer, examples=examples, followups=[], custom=True)
        self.kb.insert(len(self.kb) - 1, art)
        if self.db:
            self.db.support_save_topic(art)
        self.train()
        return aid

    # ---------------- autonomous inbox triage
    def triage(self, messages: list[str]) -> list[dict]:
        """Process a batch of messages with no human in the loop: answer, research, route, prioritise or escalate."""
        rows = []
        for text in [m.strip() for m in messages if m.strip()]:
            r = self.respond(text, [])
            safety = bool(SAFETY_RE.search(text))
            urgent = bool(URGENT_RE.search(text)) or r["sentiment"] == "frustrated" or safety
            dept = "Quality and Safety" if safety else DEPARTMENT.get(r["intent"], "Customer Service Desk")
            ticket = r["ticket"]
            if r["status"] in ("answered", "smalltalk", "catalogue", "researched", "learned"):
                action = {"researched": "Researched and replied", "catalogue": "Answered from the product catalogue"}.get(r["status"], "Auto-replied")
                if urgent and r["status"] in ("answered", "catalogue", "researched"):      # answered, but time-sensitive -> alert the team
                    ticket = ticket or self._ticket(text, [], dept, "High")
                    action += f" and flagged urgent to {dept}"
            elif r["status"] == "clarify":
                action = "Asked the customer to clarify"
            elif r["status"] == "escalate":
                action = f"Escalated to {ticket['department']}" if ticket else "Escalated"
            else:
                ticket = ticket or self._ticket(text, [], "Customer Service Desk", "High" if urgent else "Normal")
                action = "Could not answer, logged for staff review"
            rows.append({"message": text, "intent_id": r["intent"], "intent": r["title"] if r["status"] != "fallback" else "No reliable match",
                         "confidence": r["confidence"], "status": r["status"], "action": action,
                         "department": (ticket or {}).get("department", dept if r["status"] != "fallback" else "-"),
                         "priority": "High" if urgent else "Normal", "ticket": (ticket or {}).get("id", "-"),
                         "reply": r["text"], "sentiment": r["sentiment"], "mode": r["mode"]})
        return rows

    def evaluate_heldout(self) -> dict:
        """Accuracy on phrasings that were never used for training (reported honestly, not tuned to)."""
        correct, wrong = 0, []
        for text, expected in HELD_OUT:
            got = self.score(text)[0][0]
            if got == expected:
                correct += 1
            else:
                wrong.append((text, expected, got))
        return {"accuracy": correct / len(HELD_OUT), "n": len(HELD_OUT), "wrong": wrong}

    # ---------------- analytics
    def stats(self) -> dict:
        df = pd.DataFrame(self.log)
        if df.empty:
            return {"df": df, "n": 0}
        rated = df[df["feedback"] != 0]
        return {"df": df, "n": len(df),
                "answered": int((df["status"].isin(["answered", "smalltalk", "catalogue", "researched", "learned"])).sum()),
                "researched": int((df["status"] == "researched").sum()),
                "fallback": int((df["status"] == "fallback").sum()),
                "satisfaction": float((rated["feedback"] == 1).mean()) if len(rated) else None,
                "avg_conf": float(df["confidence"].mean())}


def triage_run(rows: list[dict]) -> dict:
    """Summarise a triage batch as one perceive -> analyse -> decide -> recommend -> act cycle."""
    if not rows:
        return {}
    n = len(rows)
    replied = [r for r in rows if r["action"].startswith(("Auto-replied", "Researched", "Answered"))]
    researched = [r for r in rows if r["status"] == "researched"]
    catalog = [r for r in rows if r["status"] == "catalogue"]
    urgent = [r for r in rows if r["priority"] == "High"]
    escal = [r for r in rows if r["action"].startswith("Escalated")]
    unans = [r for r in rows if r["status"] == "fallback"]
    clar = [r for r in rows if r["status"] == "clarify"]
    frustr = [r for r in rows if r["sentiment"] == "frustrated"]
    topics = pd.Series([r["intent"] for r in rows if r["status"] in ("answered", "clarify", "catalogue")]).value_counts()
    tickets = sorted({r["ticket"] for r in rows if r["ticket"] != "-"})
    perceive = [f"Received {n} customer messages with no human triage.",
                "Read each message for intent, products, urgency, food-safety words and frustration."]
    analyse = [f"Most common topic: {topics.index[0]} ({topics.iloc[0]})." if len(topics) else "No clear topics found.",
               f"{len(urgent)} message(s) were urgent, safety-related or from frustrated customers.",
               f"{len(catalog)} message(s) were product questions answered from the live catalogue.",
               f"{len(researched) + len(unans)} message(s) were outside the knowledge base"
               + (f"; {len(researched)} were researched online." if researched else ".")]
    decide = [f"Answer directly: {len(replied)} message(s).", f"Escalate to staff: {len(escal) + len([r for r in unans if r not in escal])}.",
              f"Ask for clarification: {len(clar)}."]
    for r in urgent[:3]:
        decide.append(f"Raised priority for '{r['message'][:50]}' and routed it to {r['department']}.")
    recommend = []
    gaps = [r for r in unans if r.get("intent_id") != OOS_INTENT]
    if gaps:
        recommend.append("Add knowledge-base topics for: " + "; ".join(f"'{r['message'][:40]}'" for r in gaps[:3]) + ".")
    elif unans:
        recommend.append("The unanswered message(s) were off-topic, so no new knowledge-base topic is needed.")
    if researched:
        recommend.append("Review the researched answers and give a thumbs up to the good ones so the agent remembers them.")
    if frustr:
        recommend.append(f"Follow up personally with {len(frustr)} frustrated customer(s) today.")
    recommend.append("Review low-confidence answers and teach the agent the right topic to improve accuracy.")
    act = [f"Sent {len(replied)} automatic repl{'y' if len(replied) == 1 else 'ies'}.",
           f"Opened {len(tickets)} ticket(s): {', '.join(tickets) if tickets else 'none'} (simulated in this demo)."]
    return {"perceive": perceive, "analyse": analyse, "decide": decide, "recommend": recommend, "act": act}
