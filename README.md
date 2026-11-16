# AI-Powered Student Academic Advisory Agent

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `advisor_engine.py` - the AI: ML prediction, rule-based reasoning, recommender, chat agent, report generator
- `app.py` - Streamlit interface (marks entry, dashboard, predictions, recommendations, chat, report)

## How it meets the AI requirement
1. **Machine learning** - Ridge regression predicts the final mark for subjects whose exam is not yet sat, from CA average, CA trend, attendance and study hours (80% prediction interval shown). It is trained on synthetic data; replace with real historical records when available.
2. **Rule-based reasoning** - an expert-system rule base diagnoses why a subject is weak (declining trend, inconsistent marks, exam-technique gap, low attendance/study time) and picks targeted study activities and a risk level.
3. **Recommender** - matches subjects to learning resources and splits weekly study hours in proportion to need.
4. **Conversational agent** - intent matching with fuzzy subject detection; supports what-if scenarios ("What if I score 70 in X exam?") and goal maths ("What do I need to get a B in X?").
5. **Report** - downloadable personalised HTML report (open in a browser, Print > Save as PDF).

## Marking scheme assumed
CA (CAT1, CAT2, Assignment) = 30%, Exam = 70%. Grades: A 70+, B 60-69, C 50-59, D 40-49, E below 40. Change `CA_WEIGHT`, `EXAM_WEIGHT` and `GRADE_BANDS` in `advisor_engine.py` to match your university.
# ai_agents