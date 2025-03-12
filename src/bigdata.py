import itertools
import pandas as pd

# Define lists for generating prompts
functions = ["calculate", "determine", "find", "show", "identify", "measure"]
function_types = ["average", "minimum", "maximum", "trend", "latest"]
parameters = ["temperature", "power consumption", "system load", "rotation speed", "torque", "tool wear", "air pressure"]
specials = [("detect", "fault detection"), ("summarize", "log summary")]

synthetic_data = []

def annotate_single(prompt, ft, param):
    """
    Given a prompt with a single function-type and parameter,
    produce tokens and corresponding BIO tags.
    The function type (e.g. average) is tagged as B-FUNC,
    and the parameter is tagged with B-PARAM for its first token and I-PARAM for the remaining tokens.
    """
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # Tag the function type token (assume it appears exactly once)
    for i, token in enumerate(tokens):
        if token.lower() == ft:
            ner_tags[i] = "B-FUNC"
            break
    # Tag the parameter tokens
    param_tokens = param.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param_tokens[0]:
            match = True
            for j in range(1, len(param_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param_tokens[j]:
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    return " ".join(tokens), " ".join(ner_tags)

def annotate_double(prompt, pair1, pair2):
    """
    Annotate a prompt containing two function–parameter pairs.
    Each pair is given the same annotation as in annotate_single.
    """
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # For each pair, find and tag the function type and parameter.
    for ft, param in [ (pair1[1], pair1[2]), (pair2[1], pair2[2]) ]:
        # Tag function type token:
        for i, token in enumerate(tokens):
            if token.lower() == ft and ner_tags[i] == "O":
                ner_tags[i] = "B-FUNC"
                break
        # Tag parameter tokens:
        param_tokens = param.split()
        for i in range(len(tokens)):
            if tokens[i].lower() == param_tokens[0] and ner_tags[i] == "O":
                match = True
                for j in range(1, len(param_tokens)):
                    if i+j >= len(tokens) or tokens[i+j].lower() != param_tokens[j]:
                        match = False
                        break
                if match:
                    ner_tags[i] = "B-PARAM"
                    for j in range(1, len(param_tokens)):
                        ner_tags[i+j] = "I-PARAM"
                    break
    return " ".join(tokens), " ".join(ner_tags)

def annotate_special(prompt, func, special_label):
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # Tag the function word (e.g. detect or summarize)
    for i, token in enumerate(tokens):
        if token.lower() == func:
            ner_tags[i] = "B-FUNC"
            break
    # Tag the special label tokens (e.g. fault detection)
    special_tokens = special_label.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == special_tokens[0]:
            match = True
            for j in range(1, len(special_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != special_tokens[j]:
                    match = False
                    break
            if match:
                ner_tags[i] = f"B-{special_label.upper().replace(' ', '_')}"
                for j in range(1, len(special_tokens)):
                    ner_tags[i+j] = f"I-{special_label.upper().replace(' ', '_')}"
                break
    return " ".join(tokens), " ".join(ner_tags)

# --- Generate synthetic sentences for single function–parameter pairs ---
for func in functions:
    for ft in function_types:
        for param in parameters:
            prompt = f"{func} the {ft} {param}."
            tokens, tags = annotate_single(prompt, ft, param)
            synthetic_data.append((tokens, tags))

# --- Generate synthetic sentences for two function–parameter pairs ---
# We generate two pairs only if the parameters are different to avoid duplicates.
for (func1, ft1, param1), (func2, ft2, param2) in itertools.product(
        itertools.product(functions, function_types, parameters),
        repeat=2):
    if param1 != param2:
        prompt = f"{func1} the {ft1} {param1} and {func2} the {ft2} {param2}."
        tokens, tags = annotate_double(prompt, (func1, ft1, param1), (func2, ft2, param2))
        synthetic_data.append((tokens, tags))

# --- Include special queries ---
for func, special_label in specials:
    prompt = f"{func} {special_label}."
    tokens, tags = annotate_special(prompt, func, special_label)
    synthetic_data.append((tokens, tags))

# Convert to DataFrame and save as CSV
df_synthetic = pd.DataFrame(synthetic_data, columns=["tokens", "ner_tags"])
output_path = "synthetic_ner_dataset.csv"
df_synthetic.to_csv(output_path, index=False)
print(f"Synthetic NER dataset CSV generated and saved as {output_path}")
