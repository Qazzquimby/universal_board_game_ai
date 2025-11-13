from __future__ import annotations
import json
from hcloud import Client

from remote_play.hetzner import HETZNER_KEY

client = Client(token=HETZNER_KEY)

with open("servers.json") as f:
    ids = json.load(f)

for sid in ids:
    server = client.servers.get_by_id(sid)
    print(f"Deleting server {server.name} ({sid})")
    server.delete()

print("All servers deleted.")
