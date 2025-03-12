import itertools
import pandas as pd

# ------------------ Define lists for generating prompts ------------------
functions = ["calculate", "determine", "find", "show", "identify", "measure"]
function_types = ["average", "minimum", "maximum", "trend", "latest"]
parameters = ["temperature", "power consumption", "system load", "rotation speed", "torque", "tool wear", "air pressure"]

templates_single = [
    "{func} the {func_type} {param}.",
    "What is the {func_type} {param}?",
    "Please {func} the {func_type} {param}."
]

templates_double = [
    "{func} the {func_type1} {param1} and {func_type2} {param2}.",
    "What is the {func_type1} {param1} and {func_type2} {param2}?",
    "Please {func} the {func_type1} {param1} and the {func_type2} {param2}."
]

templates_triple = [
    "{func} the {func_type1} {param1}, {func_type2} {param2} and {func_type3} {param3}.",
    "What is the {func_type1} {param1}, {func_type2} {param2} and {func_type3} {param3}?",
    "Please {func} the {func_type1} {param1}, the {func_type2} {param2} and the {func_type3} {param3}."
]

# ------------------ Annotation Functions ------------------
def annotate_single(prompt, ft, param):
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # Tag the function type token (first occurrence of ft)
    for i, token in enumerate(tokens):
        if token.lower() == ft.lower():
            ner_tags[i] = "B-FUNC"
            break
    # Tag the parameter tokens (first occurrence of param tokens)
    param_tokens = param.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param_tokens[0].lower():
            match = True
            for j in range(1, len(param_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    return " ".join(tokens), " ".join(ner_tags)

def annotate_double(prompt, ft1, param1, ft2, param2):
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # Tag first pair
    for i, token in enumerate(tokens):
        if token.lower() == ft1.lower() and ner_tags[i] == "O":
            ner_tags[i] = "B-FUNC"
            break
    param1_tokens = param1.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param1_tokens[0].lower() and ner_tags[i] == "O":
            match = True
            for j in range(1, len(param1_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param1_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param1_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    # Tag second pair
    for i, token in enumerate(tokens):
        if token.lower() == ft2.lower() and ner_tags[i] == "O":
            ner_tags[i] = "B-FUNC"
            break
    param2_tokens = param2.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param2_tokens[0].lower() and ner_tags[i] == "O":
            match = True
            for j in range(1, len(param2_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param2_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param2_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    return " ".join(tokens), " ".join(ner_tags)

def annotate_triple(prompt, ft1, param1, ft2, param2, ft3, param3):
    tokens = prompt.split()
    ner_tags = ["O"] * len(tokens)
    # Tag first pair
    for i, token in enumerate(tokens):
        if token.lower() == ft1.lower() and ner_tags[i] == "O":
            ner_tags[i] = "B-FUNC"
            break
    param1_tokens = param1.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param1_tokens[0].lower() and ner_tags[i] == "O":
            match = True
            for j in range(1, len(param1_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param1_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param1_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    # Tag second pair
    for i, token in enumerate(tokens):
        if token.lower() == ft2.lower() and ner_tags[i] == "O":
            ner_tags[i] = "B-FUNC"
            break
    param2_tokens = param2.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param2_tokens[0].lower() and ner_tags[i] == "O":
            match = True
            for j in range(1, len(param2_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param2_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param2_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    # Tag third pair
    for i, token in enumerate(tokens):
        if token.lower() == ft3.lower() and ner_tags[i] == "O":
            ner_tags[i] = "B-FUNC"
            break
    param3_tokens = param3.split()
    for i in range(len(tokens)):
        if tokens[i].lower() == param3_tokens[0].lower() and ner_tags[i] == "O":
            match = True
            for j in range(1, len(param3_tokens)):
                if i+j >= len(tokens) or tokens[i+j].lower() != param3_tokens[j].lower():
                    match = False
                    break
            if match:
                ner_tags[i] = "B-PARAM"
                for j in range(1, len(param3_tokens)):
                    ner_tags[i+j] = "I-PARAM"
                break
    return " ".join(tokens), " ".join(ner_tags)

# ------------------ Generate Synthetic Data ------------------
rows = []

# Single pair rows
for param in parameters:
    for func in functions:
        for ft in function_types:
            for tmpl in templates_single:
                prompt = tmpl.format(func=func, func_type=ft, param=param)
                tokens, tags = annotate_single(prompt, ft, param)
                rows.append((tokens, tags))

# Double pair rows (for distinct parameters)
for param_pair in itertools.combinations(parameters, 2):
    for func in functions:
        for ft1 in function_types:
            for ft2 in function_types:
                for tmpl in templates_double:
                    prompt = tmpl.format(func=func, func_type1=ft1, param1=param_pair[0],
                                           func_type2=ft2, param2=param_pair[1])
                    tokens, tags = annotate_double(prompt, ft1, param_pair[0], ft2, param_pair[1])
                    rows.append((tokens, tags))

# Triple pair rows (for distinct parameters)
for param_trip in itertools.combinations(parameters, 3):
    for func in functions:
        for ft1 in function_types:
            for ft2 in function_types:
                for ft3 in function_types:
                    for tmpl in templates_triple:
                        prompt = tmpl.format(func=func, func_type1=ft1, param1=param_trip[0],
                                               func_type2=ft2, param2=param_trip[1],
                                               func_type3=ft3, param3=param_trip[2])
                        tokens, tags = annotate_triple(prompt, ft1, param_trip[0], ft2, param_trip[1], ft3, param_trip[2])
                        rows.append((tokens, tags))

# Convert to DataFrame and save as CSV
df_large = pd.DataFrame(rows, columns=["tokens", "ner_tags"])
output_path = "synthetic_ner_dataset_large.csv"
df_large.to_csv(output_path, index=False)
print(f"Synthetic NER dataset with {len(df_large)} rows saved as {output_path}")
