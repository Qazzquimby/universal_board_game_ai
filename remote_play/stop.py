from __future__ import annotations
import json
from os import environ
from hcloud import Client

assert "HCLOUD_TOKEN" in environ, "Please set HCLOUD_TOKEN"
token = environ["HCLOUD_TOKEN"]
client = Client(token=token)

with open("servers.json") as f:
    ids = json.load(f)

for sid in ids:
    server = client.servers.get_by_id(sid)
    print(f"Deleting server {server.name} ({sid})")
    server.delete()

print("All servers deleted.")
