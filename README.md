# AI Agents Suite

One Streamlit app with three agents. Flat colours (no gradients), no emojis, Material icons for actions,
and a theme control that follows the system (System / Light / Dark).

```
pip install -r requirements.txt
streamlit run app.py
python tests/test_agents.py        # optional quick checks
```

| Page | Assignment option | Intelligence |
|---|---|---|
| Academic Advisor | Option 1 - Rule-based reasoning and prediction | Ridge regression predicts unfinished exams; an expert-system rule base diagnoses weak subjects, picks study activities and a risk level; a resource recommender and study-time planner; a chat agent with what-if and goal calculations; HTML report |
| Support Agent | Option 2 - NLP and conversational AI | TF-IDF + Logistic Regression intent classifier, TF-IDF retrieval over a knowledge base, blended confidence with answer / clarify / fallback / human hand-off, conversation memory for follow-ups, entity and frustration detection, learning from thumbs up/down and corrections, analytics |
| Finance Agent | Option 3 - Machine learning and predictive analytics | Text classifier auto-categorises expenses and learns from corrections; pattern recognition (weekday profile, recurring bills, small-purchase leakage, outliers, duplicates); 50/30/20 excess detection; Ridge forecast chosen by back-testing; ranked saving strategies; savings-goal planner; HTML and CSV export |

## Files
- `app.py` router, theme switch  |  `theme.py` flat light/dark design system
- `views/advisor.py`, `views/support.py`, `views/finance.py` pages
- `advisor_engine.py`, `support_engine.py`, `finance_engine.py` the AI logic (testable without the UI)

## Theme
The default **System** mode reads the browser/OS theme (`st.context.theme`); **Light** and **Dark** override it.
Charts, tables and cards are drawn with the theme module so they follow the switch.

## Honest notes
- The advisor's prediction model is trained on synthetic data; the CA/exam weights (30/70) and grade bands are
  assumptions in `advisor_engine.py`. Change them to match your university.
- The support knowledge base is **sample content**. Replace the answers with your institution's official information.
  Learned feedback is saved to `learned_support.json` (delete it to reset).
- Finance categories and guideline percentages are standard defaults, and the sample data is generated. Entered data
  stays in the browser session only.
