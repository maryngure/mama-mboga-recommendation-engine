"""
Day 4 - Customer Recommendation Engine
Squad Python Specialization - Data Science, AI & Analytics

Core Python only for the engine (csv, collections, math are fine). No ML libraries.
"""
from __future__ import annotations
from collections import Counter
import csv
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"

POPULARITY_WEIGHT = 0.01

CSV_COLUMNS = [
    "customer_id", "customer_name", "rank", "product_id",
    "product_name", "category", "score", "supporting_customers",
]


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]: 
    valid_rows = []
    rejected_rows = []
    seen_ids = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean
            row = {key: value.strip() for key, value in row.items()}
            row["customer_id"] = row["customer_id"].upper()

            # Check
            reason = None
            if row["purchase_id"] in seen_ids:
                reason = "duplicate purchase_id"
            elif not row["product_id"]:
                reason = "missing product"
            else:
                try:
                    row["quantity"] = int(row["quantity"])
                except ValueError:
                    reason = "invalid quantity"
                else:
                    if row["quantity"] <= 0:
                        reason = "zero or negative quantity"

            # Sort into the right list
            if reason:
                rejected_rows.append({**row, "reason": reason})
            else:
                seen_ids.add(row["purchase_id"])
                valid_rows.append(row)

    return valid_rows, rejected_rows



def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|. Define (and test) what happens when both are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity
        self._history: dict[str, set[str]] = {}        # customer_id -> product_ids
        self._customer_names: dict[str, str] = {}      # customer_id -> name
        self._products: dict[str, dict] = {}           # product_id -> name, category

        for row in purchases:
            customer_id = row["customer_id"]
            product_id = row["product_id"]
            self._history.setdefault(customer_id, set()).add(product_id)
            self._customer_names[customer_id] = row["customer_name"]
            self._products[product_id] = {
                "name": row["product_name"],
                "category": row["category"],
            }

        # popularity = number of different customers who bought each product
        self._popularity: Counter = Counter(
            product_id
            for products in self._history.values()
            for product_id in products
        )
        

    def history(self, customer_id: str) -> set[str]:
        """Return a copy of the set of products this customer has bought."""
        if customer_id not in self._history:
            raise KeyError(f"Unknown customer: {customer_id}")
        return set(self._history[customer_id])
    

    def customer_ids(self) -> list[str]:
        """Return all customer IDs in sorted order."""
        return sorted(self._history)

    def product_ids(self) -> list[str]:
        """Return all product IDs in sorted order."""
        return sorted(self._products)

    def customer_name(self, customer_id: str) -> str:
        """Return a customer's name. Unknown customers raise a clear KeyError."""
        if customer_id not in self._customer_names:
            raise KeyError(f"Unknown customer: {customer_id}")
        return self._customer_names[customer_id]
    

    def similarity(self, customer_a: str, customer_b: str) -> float:
        """Jaccard similarity between two customers' purchase histories."""
        return jaccard(self.history(customer_a), self.history(customer_b))
    

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the k most similar other customers as (customer_id, score), best first.
         Rules:
        - The customer themself is excluded.
        - A neighbour must share at least one product (score > 0) and score
          at least min_similarity.
        - Ties are broken by customer ID (A to Z), so results are reproducible.
        """
        target = self.history(customer_id)  # raises KeyError for an unknown customer

        scored = []
        for other_id, other_products in self._history.items():
            if other_id == customer_id:
                continue
            score = jaccard(target, other_products)
            if score > 0 and score >= self.min_similarity:
                scored.append((other_id, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
    

    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by similar customers but not by this one, with evidence.

        Each candidate maps to:
        - supporters: list of (neighbour_id, similarity) who bought it
        - popularity: how many customers bought it overall
        - fallback: True if it came from the fallback rule

        Fallback rule: if the customer has no neighbours, or the neighbours
        have nothing new to offer, suggest the most popular products the
        customer has not bought yet. A customer who has bought everything
        gets an empty result.
        """
        owned = self.history(customer_id)  # raises KeyError for an unknown customer
        candidates: dict[str, dict] = {}

        for neighbour_id, similarity in self.similar_customers(customer_id, k):
            for product_id in sorted(self._history[neighbour_id] - owned):
                entry = candidates.setdefault(
                    product_id,
                    {
                        "supporters": [],
                        "popularity": self._popularity[product_id],
                        "fallback": False,
                    },
                )
                entry["supporters"].append((neighbour_id, similarity))

        if not candidates:
            for product_id in sorted(self._products):
                if product_id not in owned:
                    candidates[product_id] = {
                        "supporters": [],
                        "popularity": self._popularity[product_id],
                        "fallback": True,
                    }

        return candidates
      

    def score(self, evidence: dict) -> float:
        """Ranking rule: sum of supporting neighbours' similarities
        plus a small popularity bonus (POPULARITY_WEIGHT per customer who bought it)."""
        similarity_total = sum(similarity for _, similarity in evidence["supporters"])
        return similarity_total + POPULARITY_WEIGHT * evidence["popularity"]

    def recommend(self, customer_id: str, n: int = 3, k: int = 5) -> list[dict]:
        """Return up to n ranked recommendations for one customer.

        Ranked by score (highest first). Ties are broken by popularity
        (higher first), then product ID (A to Z), so results are reproducible.
        k is how many similar customers to draw candidates from.
        """
        candidates = self.candidate_products(customer_id, k)

        scored = [
            (self.score(evidence), product_id, evidence)
            for product_id, evidence in candidates.items()
        ]
        scored.sort(key=lambda item: (-item[0], -item[2]["popularity"], item[1]))

        recommendations = []
        for rank, (score, product_id, evidence) in enumerate(scored[:n], start=1):
            recommendations.append({
                "rank": rank,
                "product_id": product_id,
                "product_name": self._products[product_id]["name"],
                "category": self._products[product_id]["category"],
                "score": score,
                "supporting_customers": [c for c, _ in evidence["supporters"]],
                "fallback": evidence["fallback"],
            })
        return recommendations
    

    def recommend_all(self, n: int = 3) -> list[dict]:
        """Recommendations for every customer, one row per (customer, rank).

        A customer with no possible recommendation still gets one row with
        blank product columns, so nobody silently disappears from the file.
        """
        rows = []
        for customer_id in self.customer_ids():
            name = self._customer_names[customer_id]
            recommendations = self.recommend(customer_id, n)

            if not recommendations:
                rows.append({
                    "customer_id": customer_id,
                    "customer_name": name,
                    "rank": "",
                    "product_id": "",
                    "product_name": "",
                    "category": "",
                    "score": "",
                    "supporting_customers": "",
                })
                continue

            for r in recommendations:
                rows.append({
                    "customer_id": customer_id,
                    "customer_name": name,
                    "rank": r["rank"],
                    "product_id": r["product_id"],
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "score": f'{r["score"]:.3f}',
                    "supporting_customers": ";".join(r["supporting_customers"]),
                })
        return rows
    

    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        """Write rows to a CSV file, creating the output folder if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)











    