"""Labeled rewrite pairs for exercising meaning-drift detection.

Shared by test_verify.py and the benchmark used to choose the embedding model.
Every case is a rewrite of data/fixtures/business_register.txt. The hard case
is `legit_full_reword`: it preserves meaning while changing nearly every word,
which is what a good grade-6 rewrite looks like and what a weak similarity
metric reports as drift.
"""

from reading_level import config

FIXTURES = config.FIXTURES_DIR

ORIGINAL = (FIXTURES / "business_register.txt").read_text().strip()
UNRELATED = (FIXTURES / "astronaut_mixed.txt").read_text().strip()

# Meaning preserved, nearly every word changed.
LEGIT_FULL_REWORD = (
    "The long talks finally ended after many private letters between the two "
    "companies. Leaders said the result was a planned effort to win better "
    "terms. Experts were still unsure about the real reason behind this "
    "unusual deal. The new contract sets up a careful plan for settling future "
    "business fights. Officers must check the needed papers every three months "
    "with great care. Poor reporting could bring heavy fines and lasting harm "
    "to a company's name. Business pressures more and more shape how modern "
    "companies are built."
)

# Meaning preserved, moderate rewording.
LEGIT_MILD_REWORD = (
    "The long negotiation finally ended after extensive private letters "
    "between both companies. Executives described the outcome as a planned "
    "attempt to negotiate better terms. Analysts still remained doubtful about "
    "the reason behind this unusual concession. The revised agreement sets up a "
    "detailed framework for settling later commercial disputes. Compliance "
    "officers must check the required quarterly documents with considerable "
    "care. Poor disclosure could cause heavy regulatory penalties and lasting "
    "damage to reputation. Commercial pressures increasingly shape the basic "
    "structure of modern corporate deals."
)

# Same register and sentence shape, different subject entirely.
DRIFT_TOPIC_SWAPPED = (
    "The long science fair finally ended after many weeks of careful "
    "preparation by both classes. Teachers described the winning project as a "
    "clever attempt to explain plant growth. Students were still curious about "
    "the real method behind this unusual experiment. The new schedule sets up a "
    "detailed plan for judging future science projects. Volunteers must check "
    "the required safety forms each week with great care. Poor labeling could "
    "cause serious safety problems and lasting confusion for judges. Classroom "
    "routines increasingly shape how modern school events are run."
)

# Same topic and vocabulary, opposite claims. Embeddings are blind to this.
DRIFT_FACTS_INVERTED = (
    "The long talks finally collapsed after almost no correspondence between "
    "the two companies. Leaders said the result was an accident rather than an "
    "effort to win better terms. Experts were completely convinced about the "
    "honest reason behind this ordinary refusal. The cancelled agreement "
    "removes any plan for settling future business fights. Officers may ignore "
    "the optional quarterly papers without any real care. Full disclosure could "
    "prevent all regulatory penalties and lasting harm to a company's name. "
    "Business pressures rarely shape how modern companies are built."
)

# --------------------------------------------------------------------------
# Passage-type cases. These exercise how much freedom the rewrite prompt grants
# rather than the embedding model, so they are rewrites of the narrative and
# informational fixtures instead of the business one.
# --------------------------------------------------------------------------

NARRATIVE_ORIGINAL = (FIXTURES / "narrative_recital.txt").read_text().strip()
INFORMATIONAL_ORIGINAL = (FIXTURES / "informational_boiling.txt").read_text().strip()

# Same theme, same protected term, different story: the audition is won rather
# than lost and the fumbled notes are gone. Under the narrative tier this is a
# legitimate rewrite, not drift.
NARRATIVE_NEW_ENDING = (
    "Maya played her violin every day in the small flat above the bakery. Her "
    "teacher said many kids would try out for the same spot. On the day of the "
    "show, she felt very nervous. She played her whole song and smiled at the "
    "end. The judges picked her for the youth orchestra."
)

# Fully reorganised -- the definition now leads -- but every fact intact,
# including the number a comprehension question would ask about.
INFORMATIONAL_FACT_KEPT = (
    "Evaporation is when water turns into a gas. At sea level, water boils at "
    "100 degrees Celsius. If you add salt to water, it must get even hotter "
    "before it will boil."
)

# Identical to the above except the boiling point. This is the failure the
# informational tier exists to prevent and, as test_verify records, the one no
# current gate can see.
INFORMATIONAL_NUMBER_ALTERED = INFORMATIONAL_FACT_KEPT.replace(
    "100 degrees", "50 degrees"
)

# (label, rewrite, should_pass)
CASES = (
    ("identical", ORIGINAL, True),
    ("legit_full_reword", LEGIT_FULL_REWORD, True),
    ("legit_mild_reword", LEGIT_MILD_REWORD, True),
    ("drift_topic_swapped", DRIFT_TOPIC_SWAPPED, False),
    ("drift_facts_inverted", DRIFT_FACTS_INVERTED, False),
    ("unrelated_passage", UNRELATED, False),
)
