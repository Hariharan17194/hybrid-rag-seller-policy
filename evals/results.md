# Retrieval evaluation

- Generated: 2026-10-03 05:50 UTC
- Embedder: `sentence-transformers:sentence-transformers/all-MiniLM-L6-v2`
- Reranker: off
- Questions: 16 (`evals/questions.json`)

| Mode | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| bm25 | 0.62 | 0.94 | 0.755 |
| vector | 0.81 | 1.00 | 0.906 |
| hybrid | 0.75 | 1.00 | 0.844 |

## Hit@5 by question style

| Style | Questions | bm25 | vector | hybrid |
|---|---|---|---|---|
| keyword | 9 | 1.00 | 1.00 | 1.00 |
| paraphrase | 7 | 0.86 | 1.00 | 1.00 |

## Per question (rank of first correct chunk, - = not in top 5)

| id | style | bm25 | vector | hybrid | question |
|---|---|---|---|---|---|
| q01 | keyword | 1 | 1 | 1 | How many days does a seller have to appeal an account deactivation? |
| q02 | keyword | 1 | 1 | 1 | What ODR percentage must sellers stay under? |
| q03 | keyword | 1 | 1 | 1 | What are the three parts of a good Plan of Action? |
| q04 | paraphrase | - | 1 | 3 | The customer says the parcel never showed up. When can they open a guarantee claim against me? |
| q05 | keyword | 1 | 1 | 1 | How long does the seller have to respond to an A-to-z claim notification? |
| q06 | paraphrase | 1 | 1 | 1 | Does a claim caused by the carrier losing the package hurt my defect rate? |
| q07 | paraphrase | 1 | 1 | 1 | Can I keep part of the money if the buyer sends back something they already used? |
| q08 | keyword | 2 | 2 | 3 | What is the deadline to file a SAFE-T claim? |
| q09 | keyword | 1 | 1 | 1 | Within how many business days must a refund be issued after the return is received? |
| q10 | paraphrase | 4 | 2 | 3 | The warehouse lost some of my units and I was never paid for them. How long do I have to ask? |
| q11 | keyword | 1 | 1 | 1 | What proof of ownership is needed for an FBA reimbursement claim? |
| q12 | paraphrase | 3 | 1 | 1 | Why can't my product be found in search even though it still exists in my inventory? |
| q13 | keyword | 2 | 1 | 2 | How do I get approval to sell jewellery in a gated category? |
| q14 | paraphrase | 1 | 1 | 1 | Why is some of my balance being held back instead of paid out? |
| q15 | keyword | 1 | 1 | 1 | How many business days does a bank transfer take after a disbursement? |
| q16 | paraphrase | 2 | 2 | 1 | When should an associate send the case to another team instead of answering? |
