import pandas as pd
import itertools
from sklearn.preprocessing import MultiLabelBinarizer

# ------------------ Step 1: Load Original Dataset and Process Labels ------------------
df_orig = pd.read_csv("oversampled_log_analysis.csv")

def process_category(cat):
    """
    Convert a category string into a list of raw labels.
    Splits by comma and removes common prefixes.
    """
    parts = [x.strip() for x in cat.split(",")]
    processed = []
    for part in parts:
        # Remove common prefixes if present
        for prefix in ["parameter_query:", "multi_parameter_query:", "value:"]:
            if part.lower().startswith(prefix):
                part = part[len(prefix):].strip()
        processed.append(part)
    return processed

# Process the original Category column into lists
df_orig["Category_list"] = df_orig["Category"].apply(process_category)

# ------------------ Step 2: Generate Synthetic Data with Multi-Label (Multi-Hot) Targets ------------------
# Define the list of parameters (these represent the available measurement names)
parameters = [
    "air_temperature",
    "power_consumption",
    "process_temperature",
    "rotational_speed",
    "system_load",
    "torque",
    "tool_wear"
]

# Define function keywords for function-based queries
function_keywords = ["average", "maximum", "minimum", "latest", "trend"]

# Define prompt templates to add variety
templates = [
    "What is the {func} {param}?",
    "Calculate the {func} {param}.",
    "Find the {func} {param}.",
    "Determine the {func} {param}.",
    "Show me the {func} {param}.",
    "Give me the {func} {param}.",
    "Provide the {func} {param}."
]

synthetic_data = []

# Synthetic Single-Parameter Queries
for param in parameters:
    label_list = [param]  # single label as a list
    for func in function_keywords:
        for tmpl in templates:
            prompt = tmpl.format(func=func, param=param)
            synthetic_data.append({"Prompt": prompt, "Category_list": label_list})

# Synthetic Two-Parameter Queries
for params in itertools.combinations(parameters, 2):
    label_list = list(params)  # two labels in a list
    for func in function_keywords:
        for tmpl in templates:
            # Example prompt: "Calculate the maximum power_consumption and system_load."
            prompt = tmpl.format(func=func, param=f"{params[0]} and {params[1]}")
            synthetic_data.append({"Prompt": prompt, "Category_list": label_list})

# Synthetic Three-Parameter Queries
for params in itertools.combinations(parameters, 3):
    label_list = list(params)  # three labels
    for func in function_keywords:
        for tmpl in templates:
            prompt = tmpl.format(func=func, param=f"{params[0]}, {params[1]} and {params[2]}")
            synthetic_data.append({"Prompt": prompt, "Category_list": label_list})

df_synthetic = pd.DataFrame(synthetic_data)

# ------------------ Step 3: Combine Original and Synthetic Data ------------------
# Keep only the necessary columns from the original data
df_orig_processed = df_orig[["Prompt", "Category_list"]].copy()
combined_df = pd.concat([df_orig_processed, df_synthetic], ignore_index=True)

# ------------------ Step 4: Multi-Hot Encode the Labels ------------------
mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(combined_df["Category_list"])
df_encoded = pd.DataFrame(multi_hot, columns=mlb.classes_)

# Combine the Prompt column with the multi-hot encoded labels
df_final = pd.concat([combined_df["Prompt"], df_encoded], axis=1)

# ------------------ Step 5: Save the Final CSV ------------------
output_path = "synthetic_multi_hot_dataset.csv"
df_final.to_csv(output_path, index=False)
print(f"Multi-hot synthetic dataset saved to {output_path}")
