import pandas as pd

def main():
    queries = pd.read_json(
        "data/hotpotqa/queries.jsonl",
        lines=True
    )

    train_qrels = pd.read_csv(
        "data/hotpotqa/qrels/train.tsv",
        sep="\t"
    )

    train_query_ids = set(train_qrels["query-id"].astype(str))

    train_queries = queries[
        queries["_id"].astype(str).isin(train_query_ids)
    ]

    train_queries.to_json(
        "data/hotpotqa/train_queries.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )
if __name__ == '__main__':
    main()