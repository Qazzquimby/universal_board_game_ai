from __future__ import annotations
import json
from hcloud import Client

from remote_play.hetzner import HETZNER_KEY

client = Client(token=HETZNER_KEY)


def main():
    path = "servers.json"
    with open(path) as f:
        server_details = json.load(f)

        for server_detail in server_details:
            server = client.servers.get_by_id(server_detail["id"])
            print(f"Deleting server {server.name} ({server_detail})")
            server.delete()

    with open(path, "w") as f:
        pass  # Clears the file

    print("All servers deleted.")
