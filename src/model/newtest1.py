import pandas as pd

# List of new, domain-specific prompts with tokens and NER tags.
# Each entry is a dictionary with keys "tokens" and "ner_tags".
new_prompts = [
    {
        "tokens": ["Summarize", "the", "rotational", "speed", "trend", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "torque", "variations", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "air", "temperature", "fluctuations", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "process", "temperature", "changes", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "tool", "wear", "data", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "O"]
    },
    {
        "tokens": ["Summarize", "the", "vibration", "levels", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "failure", "type", "occurrences", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "rotational", "speed", "and", "torque", "measurements", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "air", "temperature", "and", "process", "temperature", "trends", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "torque", "fluctuation", "statistics", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "sensor", "data", "for", "rotational", "speed", "and", "air", "temperature", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "process", "parameters", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "trends", "in", "tool", "wear", "and", "rotational", "speed", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "fluctuation", "of", "torque", "over", "time", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "O", "O", "O"]
    },
    {
        "tokens": ["Digest", "the", "process", "temperature", "readings", "from", "the", "last", "shift", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Summarize", "the", "hourly", "variations", "in", "air", "temperature", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "torque", "and", "rotational", "speed", "data", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "B-PARAM", "O", "B-PARAM", "I-PARAM", "O", "O"]
    },
    {
        "tokens": ["Recap", "the", "sensor", "measurements", "for", "process", "temperature", "and", "tool", "wear", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "trend", "of", "rotational", "speed", "and", "its", "fluctuations", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "latest", "readings", "of", "tool", "wear", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "air", "temperature", "measurements", "recorded", "today", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "O", "O", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "process", "temperature", "fluctuations", "during", "peak", "hours", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "variations", "in", "rotational", "speed", "over", "the", "last", "24", "hours", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "torque", "readings", "captured", "this", "morning", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "summary", "of", "rotational", "speed", "fluctuations", "and", "torque", "changes", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "brief", "recap", "of", "air", "temperature", "and", "tool", "wear", "data", "."],
        "ner_tags": ["O", "O", "O", "B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O", "O"]
    },
    {
        "tokens": ["Summarize", "the", "trend", "in", "process", "temperature", "during", "the", "operation", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "sensor", "performance", "focusing", "on", "rotational", "speed", "and", "torque", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "latest", "measurements", "of", "air", "temperature", "and", "process", "temperature", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "recorded", "values", "of", "tool", "wear", "over", "the", "past", "cycle", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "variation", "in", "torque", "and", "its", "impact", "on", "performance", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "O", "O", "O", "O", "B-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "rotational", "speed", "fluctuations", "during", "high", "load", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "air", "temperature", "trend", "observed", "in", "the", "morning", "shift", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "process", "temperature", "statistics", "from", "the", "last", "quarter", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "recent", "changes", "in", "tool", "wear", "measurements", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "torque", "and", "rotational", "speed", "statistics", "for", "the", "current", "cycle", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-PARAM", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "concise", "overview", "of", "air", "temperature", "readings", "."],
        "ner_tags": ["O", "O", "O", "B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "O"]
    },
    {
        "tokens": ["Recap", "the", "fluctuations", "in", "process", "temperature", "over", "the", "past", "week", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "key", "performance", "parameters", "including", "torque", "and", "tool", "wear", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "O", "B-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "summary", "of", "rotational", "speed", "data", "collected", "yesterday", "."],
        "ner_tags": ["B-SUMMARY", "O", "B-SUMMARY", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "overall", "sensor", "data", "focusing", "on", "process", "temperature", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "report", "summarizing", "torque", "fluctuations", "and", "rotational", "speed", "."],
        "ner_tags": ["O", "O", "O", "B-SUMMARY", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "changes", "in", "air", "temperature", "during", "the", "operation", "period", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "trend", "in", "tool", "wear", "observed", "this", "quarter", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "performance", "metrics", "for", "rotational", "speed", "and", "torque", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "variation", "in", "tool", "wear", "over", "multiple", "cycles", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Provide", "a", "summary", "of", "the", "sensor", "outputs", "focusing", "on", "process", "temperature", "."],
        "ner_tags": ["O", "O", "B-SUMMARY", "O", "O", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Recap", "the", "latest", "torque", "and", "tool", "wear", "readings", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "B-PARAM", "O", "B-PARAM", "I-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Outline", "the", "data", "trends", "for", "air", "temperature", "and", "rotational", "speed", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Digest", "the", "recorded", "sensor", "metrics", "focusing", "on", "torque", "fluctuations", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "O", "O", "O", "O", "B-PARAM", "I-PARAM", "O"]
    },
    {
        "tokens": ["Summarize", "the", "hourly", "rotational", "speed", "and", "tool", "wear", "data", "."],
        "ner_tags": ["B-SUMMARY", "O", "O", "B-PARAM", "I-PARAM", "O", "B-PARAM", "I-PARAM", "O", "O"]
    }
]

# Convert the list of new prompts into a DataFrame.
new_prompts_df = pd.DataFrame(new_prompts)

# Save the new prompts to a CSV file.
output_file_path = "assets/expanded_prompts_domain.csv"
new_prompts_df.to_csv(output_file_path, index=False)
print("Expanded domain-specific prompts saved to:", output_file_path)
