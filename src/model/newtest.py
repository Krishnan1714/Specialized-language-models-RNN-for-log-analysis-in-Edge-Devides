import pandas as pd

# -------------------------------
# 1. Load the Original Cleaned Dataset
# -------------------------------
# Update this path if needed.
original_csv = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df_original = pd.read_csv(original_csv)
print(f"Original dataset loaded with {len(df_original)} examples.")

# -------------------------------
# 2. Define Synthetic Negative Examples
# -------------------------------
# Each negative example is a dictionary with "tokens" and "ner_tags".
# Here, tokens and ner_tags are stored as string representations of lists.
negative_examples = [
    {
        "tokens": "['feature', 'training', 'needed']",
        "ner_tags": "['O', 'O', 'O']"
    },
    {
        "tokens": "['this', 'is', 'random', 'text']",
        "ner_tags": "['O', 'O', 'O', 'O']"
    },
    {
        "tokens": "['no', 'entities', 'here']",
        "ner_tags": "['O', 'O', 'O']"
    },
    {
        "tokens": "['unrelated', 'information']",
        "ner_tags": "['O', 'O']"
    },
    {
        "tokens": "['random', 'words', 'with', 'no', 'meaning']",
        "ner_tags": "['O', 'O', 'O', 'O', 'O']"
    },
    {
        "tokens": "['this', 'sentence', 'has', 'nothing', 'special']",
        "ner_tags": "['O', 'O', 'O', 'O', 'O']"
    },
    # Additional negative examples:
    {
        "tokens": "['completely', 'irrelevant', 'data']",
        "ner_tags": "['O', 'O', 'O']"
    },
    {
        "tokens": "['nothing', 'to', 'see', 'here']",
        "ner_tags": "['O', 'O', 'O', 'O']"
    },
    {
        "tokens": "['just', 'a', 'random', 'phrase']",
        "ner_tags": "['O', 'O', 'O', 'O']"
    },
    {
        "tokens": "['bland', 'and', 'uninspiring', 'text']",
        "ner_tags": "['O', 'O', 'O', 'O']"
    },
    {
        "tokens": "['nonsense', 'words', 'only']",
        "ner_tags": "['O', 'O', 'O']"
    },
    {
        "tokens": "['empty', 'prompt']",
        "ner_tags": "['O', 'O']"
    }
]

# -------------------------------
# 3. Convert Negative Examples to a DataFrame
# -------------------------------
df_negative = pd.DataFrame(negative_examples)
print(f"Negative examples dataset created with {len(df_negative)} examples.")

# -------------------------------
# 4. Append Negative Examples to the Original Dataset
# -------------------------------
df_combined = pd.concat([df_original, df_negative], ignore_index=True)
print(f"Combined dataset has {len(df_combined)} examples.")

# -------------------------------
# 5. Save the Combined Dataset to a New CSV File
# -------------------------------
output_csv = "assets/log_analysis_ner_with_negatives.csv"
df_combined.to_csv(output_csv, index=False)
print(f"Combined dataset saved to {output_csv}")
