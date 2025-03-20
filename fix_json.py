import json

FEEDBACK_FILE = "feedback.json"

def fix_feedback_json():
    """Ensure feedback.json is properly formatted as a list."""
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
            content = file.read().strip()
        
        if content.startswith("{"):  # If it's a single JSON object, wrap it in a list
            fixed_data = [json.loads(content)]
        else:
            fixed_data = json.loads(content)  # Try loading as a list

        if not isinstance(fixed_data, list):
            print("❌ JSON is still not a list! Fix manually.")
            return

        # Save the corrected JSON
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as file:
            json.dump(fixed_data, file, indent=4)

        print("✅ Fixed feedback.json successfully!")

    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")

# Run the fixer
fix_feedback_json()
