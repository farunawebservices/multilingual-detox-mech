from datasets import load_dataset

for config in ["yor", "xho"]:
    print("\n" + "=" * 80)
    print("CONFIG:", config)

    dataset = load_dataset(
        "afrihate/afrihate",
        config,
        token=True
    )

    print(dataset)

    for split_name, split in dataset.items():
        print("\nSplit:", split_name)
        print("Rows:", len(split))
        print("Columns:", split.column_names)
        print("First example:", split[0])

        if "label" in split.column_names:
            labels = split["label"]
            counts = {}
            for label in labels:
                counts[str(label)] = counts.get(str(label), 0) + 1
            print("Label counts:", counts)