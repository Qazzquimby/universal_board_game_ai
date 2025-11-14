from __future__ import annotations
import json
from hcloud import Client

from remote_play.hetzner import HETZNER_KEY

client = Client(token=HETZNER_KEY)


def main():
    with open("servers.json") as f:
        server_details = json.load(f)

    for server_detail in server_details:
        server = client.servers.get_by_id(server_detail["id"])
        print(f"Deleting server {server.name} ({server_detail})")
        server.delete()

    print("All servers deleted.")
