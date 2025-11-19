import json
import glob
import os

INPUT_DIR = "data/connect4/game_logs"


def get_action_index(legal_actions, action):
    # legal_actions is something like [[0],[1],[2]...]
    flat = [row[0] for row in legal_actions]
    return flat.index(action)


def process_file(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        changed = False

        for entry in data:
            if "action" not in entry:
                continue

            action = entry["action"]
            legal = entry["state"]["legal_actions"]["_data"]
            idx = get_action_index(legal, action)

            entry["action_index"] = idx
            del entry["action"]
            changed = True

        if changed:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Skipping {path}: {e}")


def main():
    for path in glob.glob(os.path.join(INPUT_DIR, "**/*.json")):
        process_file(path)


if __name__ == "__main__":
    main()
