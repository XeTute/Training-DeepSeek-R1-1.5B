import json

try:
    with open("data.json", "r") as f:
        json.load(f)  # This loads the file to validate it
    print("JSON is valid.")
except json.JSONDecodeError as e:
    print(f"JSON is invalid: {e}")
