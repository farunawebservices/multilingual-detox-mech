from pathlib import Path

p = Path("scripts/07_llama_lora_finetune.py")
s = p.read_text()

# Remove the nested QLoRASafeTrainer class from inside main()
# and move it to module level (after imports, before Config class)

# First, extract the nested class definition
import re

# Find the nested class
nested_class_pattern = r"(    class QLoRASafeTrainer\(Trainer\):.*?return super\(\)\.compute_loss\(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch\))"
match = re.search(nested_class_pattern, s, re.DOTALL)

if not match:
    raise RuntimeError("Nested QLoRASafeTrainer class not found.")

nested_class = match.group(1)

# Dedent it to module level (remove 4 spaces from each line)
module_level_class = "\n".join(
    line[4:] if line.startswith("    ") else line
    for line in nested_class.split("\n")
)

# Remove the nested class from inside main()
s = s.replace(nested_class, "")

# Insert the module-level class after the imports (before "class Config:")
insert_marker = "\nclass Config:"
s = s.replace(insert_marker, f"\n{module_level_class}\n\nclass Config:")

# Also remove the blank lines left behind
s = s.replace("\n\n\n\n", "\n\n")

p.write_text(s)
print("Moved QLoRASafeTrainer to module level.")
