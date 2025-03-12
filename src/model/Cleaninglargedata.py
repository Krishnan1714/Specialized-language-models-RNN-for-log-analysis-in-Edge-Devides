import ast
import re
import string
import pandas as pd

# Allowed NER tags.
ALLOWED_TAGS = {"B-FUNC", "B-PARAM", "I-PARAM", "O"}

def process_ner_tags(tag_str):
    """
    Cleans a ner_tags string that may look like:
      "['O', 'O', 'B-FUNC', 'B-PARAM', 'I-PARAM', 'O']"
    or even as a list of comma‐separated strings.
    This function uses regex to extract only the allowed tags.
    """
    # Find all occurrences of allowed tags.
    tags = re.findall(r"B-FUNC|B-PARAM|I-PARAM|O", tag_str)
    return tags

def process_tokens(token_str):
    """
    Converts a tokens string to a list of tokens.
    If the evaluated list contains only one string (the whole sentence),
    it splits that string into tokens.
    """
    try:
        tokens = ast.literal_eval(token_str)
        if isinstance(tokens, list):
            # If there is only one element that contains spaces, split it further.
            if len(tokens) == 1 and isinstance(tokens[0], str) and " " in tokens[0]:
                tokens = tokens[0].split()
            return tokens
        else:
            return token_str.split()
    except Exception as e:
        # Fallback: split on whitespace.
        return token_str.split()

def split_punctuation(tokens, tags):
    """
    For each token, if it ends with punctuation (e.g. a period, comma, etc.)
    and its length is > 1, split it into the word (without punctuation)
    and the punctuation (as a separate token with tag "O").
    Returns the new tokens and new tags lists.
    """
    new_tokens = []
    new_tags = []
    for token, tag in zip(tokens, tags):
        if token and token[-1] in string.punctuation and len(token) > 1:
            # Separate the punctuation from the token.
            word = token[:-1]
            punct = token[-1]
            new_tokens.append(word)
            new_tags.append(tag)
            new_tokens.append(punct)
            new_tags.append("O")
        else:
            new_tokens.append(token)
            new_tags.append(tag)
    return new_tokens, new_tags

# -------------------------------
# Load the Dataset
# -------------------------------
csv_path = "assets/log_analysis_ner_large_dataset.csv"
df = pd.read_csv(csv_path)

# Process the "tokens" column.
df["tokens"] = df["tokens"].apply(process_tokens)

# Process the "ner_tags" column using our regex extractor.
df["ner_tags"] = df["ner_tags"].apply(process_ner_tags)

# Now, for each row, split punctuation from tokens and adjust the ner_tags accordingly.
def process_row(row):
    tokens, tags = row["tokens"], row["ner_tags"]
    new_tokens, new_tags = split_punctuation(tokens, tags)
    return pd.Series({"tokens": new_tokens, "ner_tags": new_tags})

df[["tokens", "ner_tags"]] = df.apply(process_row, axis=1)

# -------------------------------
# Inspect a Sample of the Cleaned Data
# -------------------------------
print("Cleaned dataset sample:")
print(df.head())

# Check unique labels from the cleaned ner_tags.
unique_labels = set()
for tags in df["ner_tags"]:
    unique_labels.update(tags)
unique_labels = sorted(list(unique_labels))
print("Unique labels (cleaned):", unique_labels)

# -------------------------------
# Save the Cleaned Dataset to a New CSV File
# -------------------------------
cleaned_csv_path = "assets/log_analysis_ner_large_dataset_cleaned.csv"
df.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned dataset saved to {cleaned_csv_path}")
