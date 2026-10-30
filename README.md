# Mama Mboga Online: Customer Recommendation Engine

A "you might also need" recommender for checkout, built from first principles in core Python (no machine-learning libraries). It looks at what similar customers bought and suggests products a customer has not tried yet, using a rule simple enough to explain to a shop owner.

> **In one sentence for a shop owner:** "We find customers whose baskets overlap most with yours, then suggest what they buy that you haven't, starting with the items the most similar customers buy."

## How it works

1. **Clean the order export.** Unusable rows are rejected with a reason; messy IDs are normalised.
2. **Represent each customer as a set** of product IDs they have bought (`dict` of customer → `set`).
3. **Measure similarity with Jaccard:** `|A ∩ B| / |A ∪ B|`, the share of products two customers have in common out of everything either of them bought. It runs from 0 (nothing in common) to 1 (identical baskets).
4. **Find neighbours:** the top-k most similar other customers, excluding the customer and anyone who shares no products.
5. **Collect candidates:** products the neighbours bought that this customer has not, with a record of which neighbours support each one.
6. **Rank candidates** with the scoring rule below and keep the top n.
7. **Export** every customer's recommendations to a CSV for the product team.

## Project structure

```
day4-recommendation-engine/
├── recommender.py          # read_purchases, jaccard, RecommendationEngine
├── main.py                 # runs the engine end to end
├── data/purchases.csv      # order export (input)
├── output/recommendations.csv   # recommendations (output)
├── tests/test_recommender.py    # 20 unit tests
├── requirements.txt
└── README.md
```

## How to run

Requires Python 3.10 or newer. The engine uses only the standard library.

```
python main.py
```

This prints recommendations for two sample customers, writes `output/recommendations.csv`, and prints a run summary. To run the tests (`pip install pytest` first):

```
python -m pytest
```

To use the engine in your own code:

```python
from recommender import read_purchases, RecommendationEngine

valid, rejected = read_purchases("data/purchases.csv")
engine = RecommendationEngine(valid)
engine.recommend("C106", n=3)      # recommendations for one customer
engine.recommend_all(n=3)          # rows for every customer
```

## Data-cleaning rules

`read_purchases(path)` returns `(valid_rows, rejected_rows)`. Every rejected row carries a `reason`.

| Problem | Rule | Example in this data |
|---|---|---|
| Repeated `purchase_id` | Keep the first, reject repeats as `duplicate purchase_id` | T9010 appears twice |
| Blank `product_id` | Reject as `missing product` (we cannot tell what was bought) | T9020 |
| Quantity of zero or less | Reject as `zero or negative quantity` | T9033 |
| Quantity that is not a whole number | Reject as `invalid quantity` | none in this file |
| Inconsistent customer ID | Strip spaces and uppercase, so `c106` becomes `C106` | T9041 |

**Result on the supplied file:** 63 rows read, 60 used, 3 rejected.

## Ranking rule

```
score = (sum of the similarity of each neighbour who bought the product)
        + 0.01 × (number of customers who bought the product overall)
```

In plain words: a product scores higher when more of your close neighbours bought it, and the closer they are, the more each one counts. Overall popularity adds a small nudge. The weight is deliberately small so popularity breaks near-ties but cannot override what similar customers actually bought.

The rule combines two signals: **neighbour similarity and agreement** (the sum rewards both closeness and the number of supporters) and **popularity**.

**Ties** are broken deterministically: higher score first, then higher popularity, then product ID A to Z. Neighbour ties are broken by customer ID A to Z, so the same input always gives the same output.

### Worked example: Mwangi (C106)

Mwangi bought rice (P06), Kimbo (P05) and chicken (P15). Her five closest neighbours are:

| Neighbour | Jaccard with Mwangi |
|---|---|
| C118 Kariuki | 0.50 |
| C102 Otieno | 0.25 |
| C105 Njeri | 0.25 |
| C114 Omondi | 0.20 |
| C107 Chebet | 0.14 |

Candidate scores:

- **Kienyeji Eggs:** bought by Kariuki (0.50), Otieno (0.25) and Omondi (0.20). Sum = 0.95. Five customers bought eggs overall, so the popularity bonus is 0.01 × 5 = 0.05. **Score = 1.000**
- **Brookside Milk:** bought by Njeri (0.25) and Chebet (0.14). Sum = 0.39 (0.393 before rounding). Five customers bought milk overall, so the bonus is 0.05. **Score = 0.443**
- **Royco Mchuzi Mix:** bought by Omondi (0.20) only. Two customers bought it overall, so the bonus is 0.02. **Score = 0.220**

So Mwangi's top three are eggs, milk and Royco. Eggs win because three of her neighbours, including the closest, bought them.

## Edge cases

- **Customer with no useful neighbours, or neighbours with nothing new:** the engine falls back to the most popular products the customer has not bought. These rows are marked as fallback internally and have a blank `supporting_customers` field. In this data, Chebet (C107) uses the fallback.
- **Customer who bought everything:** no recommendations are possible. They still appear in the CSV once, with blank product columns, so nobody silently disappears from the file.
- **Unknown customer ID:** raises a clear `KeyError` rather than returning nonsense. A brand-new customer with no history is an unknown ID; the checkout system should catch this and show the store's most popular items.
- **Two empty histories:** Jaccard is defined as 0.0 instead of dividing by zero.
- **Fewer than k neighbours:** the engine uses whatever neighbours exist.

## Output

`output/recommendations.csv` has one row per (customer, rank):

```
customer_id,customer_name,rank,product_id,product_name,category,score,supporting_customers
C101,Wanjiru,1,P03,Festive Bread,Bakery,1.070,C117;C109;C113
```

On the supplied data: 20 customers, 15 products, 56 recommendation rows. Every customer appears in the file.

## Assumptions and limitations

- Only *whether* a customer bought a product matters, not how often or how much, so quantity is only used to reject zero or negative values.
- A customer is identified by `customer_id`, and a product by `product_id`.
- A neighbour must share at least one product with the customer. Customers with nothing in common are never treated as similar.
- Purchase dates are not used, so there is no recency weighting.
- The data is small and sparse (most customers have 2 to 5 distinct products), so many customer pairs have a similarity of 0. With more order history, recommendations would become more reliable, and `k` and the popularity weight could be tuned.
- Products are not scored by category, so the engine can recommend several items from the same category.