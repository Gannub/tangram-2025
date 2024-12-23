import json

def save_logs(logs, filename):
    with open(filename, "w") as f:
        json.dump(logs, f, indent=4)
