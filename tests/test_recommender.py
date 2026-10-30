"""Tests for the recommendation engine. Run from the project folder with: python -m pytest"""
import csv

import pytest

from recommender import RecommendationEngine, jaccard, read_purchases


def make_rows(pairs):
    """Build purchase rows from (customer_id, product_id) pairs."""
    return [
        {
            "customer_id": customer,
            "customer_name": f"Name {customer}",
            "product_id": product,
            "product_name": f"Product {product}",
            "category": "Test",
        }
        for customer, product in pairs
    ]


# ---------- jaccard ----------

def test_jaccard_partial_overlap():
    assert jaccard({"a", "b", "c", "d"}, {"a", "b", "e"}) == pytest.approx(0.4)


def test_jaccard_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_zero():
    assert jaccard(set(), set()) == 0.0


def test_jaccard_one_empty_is_zero():
    assert jaccard({"a"}, set()) == 0.0


# ---------- read_purchases ----------

def write_csv(path, lines):
    header = "purchase_id,customer_id,customer_name,product_id,product_name,category,purchase_date,quantity"
    path.write_text("\n".join([header] + lines) + "\n", encoding="utf-8")


def test_read_purchases_applies_every_rule(tmp_path):
    file = tmp_path / "purchases.csv"
    write_csv(file, [
        "T1,C1,Ann,P1,Milk,Dairy,2026-10-01,2",
        "T1,C1,Ann,P1,Milk,Dairy,2026-10-01,2",       # duplicate purchase_id
        "T2,C1,Ann,,,Dairy,2026-10-02,1",              # blank product
        "T3,C1,Ann,P2,Eggs,Fresh,2026-10-03,0",        # zero quantity
        "T4,C1,Ann,P3,Rice,Pantry,2026-10-04,abc",     # non-numeric quantity
        "T5, c2 ,Ben,P1,Milk,Dairy,2026-10-05,1",      # messy customer id
    ])
    valid, rejected = read_purchases(file)

    assert [r["purchase_id"] for r in valid] == ["T1", "T5"]
    assert valid[1]["customer_id"] == "C2"
    assert valid[0]["quantity"] == 2
    reasons = {r["purchase_id"]: r["reason"] for r in rejected}
    assert reasons == {
        "T1": "duplicate purchase_id",
        "T2": "missing product",
        "T3": "zero or negative quantity",
        "T4": "invalid quantity",
    }


def test_read_purchases_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_purchases(tmp_path / "nope.csv")


# ---------- engine ----------

def test_history_is_a_copy():
    engine = RecommendationEngine(make_rows([("C1", "P1")]))
    engine.history("C1").add("P99")
    assert engine.history("C1") == {"P1"}


def test_unknown_customer_raises_key_error():
    engine = RecommendationEngine(make_rows([("C1", "P1")]))
    for method in (engine.history, engine.recommend, engine.similar_customers,
                   engine.candidate_products, engine.customer_name):
        with pytest.raises(KeyError):
            method("C999")
    with pytest.raises(KeyError):
        engine.similarity("C1", "C999")


def test_similar_customers_excludes_self_and_zero_overlap():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"), ("C1", "P2"),
        ("C2", "P1"),
        ("C3", "P9"),
    ]))
    assert engine.similar_customers("C1") == [("C2", 0.5)]


def test_ties_are_broken_by_customer_id():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"),
        ("C3", "P1"), ("C3", "P2"),
        ("C2", "P1"), ("C2", "P3"),
    ]))
    neighbours = engine.similar_customers("C1")
    assert [c for c, _ in neighbours] == ["C2", "C3"]


def test_min_similarity_filters_neighbours():
    rows = make_rows([("C1", "P1"), ("C1", "P2"), ("C2", "P1")])
    assert engine_ids(RecommendationEngine(rows, min_similarity=0.5), "C1") == ["C2"]
    assert engine_ids(RecommendationEngine(rows, min_similarity=0.6), "C1") == []


def engine_ids(engine, customer_id):
    return [c for c, _ in engine.similar_customers(customer_id)]


def test_candidates_exclude_owned_products_and_record_supporters():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"),
        ("C2", "P1"), ("C2", "P2"),
        ("C3", "P1"), ("C3", "P2"), ("C3", "P3"),
    ]))
    candidates = engine.candidate_products("C1")
    assert set(candidates) == {"P2", "P3"}
    assert [c for c, _ in candidates["P2"]["supporters"]] == ["C2", "C3"]


def test_score_combines_similarity_and_popularity():
    engine = RecommendationEngine(make_rows([("C1", "P1")]))
    evidence = {"supporters": [("C2", 0.5), ("C3", 0.25)], "popularity": 4, "fallback": False}
    assert engine.score(evidence) == pytest.approx(0.5 + 0.25 + 0.01 * 4)


def test_more_supporters_rank_higher():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"),
        ("C2", "P1"), ("C2", "P2"),
        ("C3", "P1"), ("C3", "P2"), ("C3", "P3"),
    ]))
    ranked = [r["product_id"] for r in engine.recommend("C1")]
    assert ranked == ["P2", "P3"]


def test_fallback_when_customer_has_no_neighbours():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"),
        ("C2", "P2"), ("C3", "P2"), ("C4", "P3"),
    ]))
    recs = engine.recommend("C1")
    assert [r["product_id"] for r in recs] == ["P2", "P3"]   # most popular first
    assert all(r["fallback"] for r in recs)
    assert all(r["supporting_customers"] == [] for r in recs)


def test_customer_who_bought_everything_gets_no_recommendations():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"), ("C1", "P2"),
        ("C2", "P1"),
    ]))
    assert engine.recommend("C1") == []


def test_recommend_all_keeps_every_customer_even_with_nothing_to_recommend():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"), ("C1", "P2"),
        ("C2", "P1"),
    ]))
    rows = engine.recommend_all()
    by_customer = {}
    for row in rows:
        by_customer.setdefault(row["customer_id"], []).append(row)

    assert set(by_customer) == {"C1", "C2"}
    assert len(by_customer["C1"]) == 1
    assert by_customer["C1"][0]["product_id"] == ""      # blank row for C1
    assert by_customer["C2"][0]["product_id"] == "P2"


def test_recommend_respects_n():
    engine = RecommendationEngine(make_rows([
        ("C1", "P1"),
        ("C2", "P1"), ("C2", "P2"), ("C2", "P3"), ("C2", "P4"),
    ]))
    assert len(engine.recommend("C1", n=2)) == 2


def test_export_csv_writes_expected_columns(tmp_path):
    engine = RecommendationEngine(make_rows([("C1", "P1"), ("C2", "P1"), ("C2", "P2")]))
    path = tmp_path / "out" / "recommendations.csv"
    engine.export_csv(engine.recommend_all(), path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == [
        "customer_id", "customer_name", "rank", "product_id",
        "product_name", "category", "score", "supporting_customers",
    ]
    assert {r["customer_id"] for r in rows} == {"C1", "C2"}
    c1 = next(r for r in rows if r["customer_id"] == "C1")
    assert c1["product_id"] == "P2" and c1["supporting_customers"] == "C2"