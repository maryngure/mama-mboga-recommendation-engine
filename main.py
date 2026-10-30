"""Run the recommendation engine end to end."""
from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases


def show_customer(engine: RecommendationEngine, customer_id: str) -> None:
    """Print one customer's history, nearest neighbours and recommendations."""
    products = ", ".join(sorted(engine.history(customer_id)))
    print(f"Customer {customer_id} ({engine.customer_name(customer_id)}) bought: {{{products}}}")

    print("Most similar customers:")
    for neighbour_id, score in engine.similar_customers(customer_id, k=3):
        print(f"  {neighbour_id}  jaccard={score:.2f}")

    print("Recommendations:")
    for r in engine.recommend(customer_id, n=3):
        if r["fallback"]:
            reason = "it is popular with other customers"
        else:
            count = len(r["supporting_customers"])
            noun = "customer" if count == 1 else "customers"
            reason = f"{count} similar {noun} bought it"
        print(
            f"  {r['rank']}. {r['product_name']} ({r['category']})  "
            f"score={r['score']:.3f}  because {reason}"
        )


def main() -> None:
    valid, rejected = read_purchases(DATA_PATH)
    engine = RecommendationEngine(valid)

    show_customer(engine, "C106")
    print()
    show_customer(engine, "C101")
    print()

    rows = engine.recommend_all(n=3)
    engine.export_csv(rows, OUTPUT_PATH)
    written = sum(1 for row in rows if row["product_id"])

    print(f"Purchases read: {len(valid) + len(rejected)} | used: {len(valid)} | rejected: {len(rejected)}")
    print(f"Customers: {len(engine.customer_ids())} | Products: {len(engine.product_ids())} | Recommendations written: {written}")


if __name__ == "__main__":
    main()