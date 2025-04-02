import re

# File paths
input_file = "assets/logs.csv"  # Change to your actual file path
output_file = "assets/logs2.csv"  # Change to your desired output file

# Read the file
with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

# Remove excessive blank lines
cleaned_content = re.sub(r'\n+', '\n', content).strip()

# Write the cleaned content back to a new file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(cleaned_content)

print(f"Excess blank lines removed. Cleaned content saved to {output_file}")
