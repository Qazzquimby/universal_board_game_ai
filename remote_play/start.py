from __future__ import annotations
import asyncio
import aiohttp
import asyncssh
import time
import json
from os import environ
from hcloud import Client
from hcloud.images import Image
from hcloud.server_types import ServerType
from pathlib import Path

from remote_play.hetzner import HETZNER_KEY

client = Client(token=HETZNER_KEY)

N_SERVERS = 1
APP_FILES = [
    "requirements.txt",
    "factories.py",
    "remote_play/server_app.py",
    "config.yaml",
]
APP_DIRS = ["agents", "algorithms", "core", "environments", "models", "utils"]
SERVER_TYPE = "cx23"
IMAGE = "ubuntu-24.04"


async def wait_for_ssh(ip: str, port: int = 22, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with asyncssh.connect(ip, username="root", known_hosts=None):
                return True
        except Exception:
            await asyncio.sleep(5)
    raise TimeoutError(f"SSH not available on {ip}")


async def setup_server(server):
    ip = server.public_net.ipv4.ip
    await wait_for_ssh(ip)
    async with asyncssh.connect(ip, username="root", known_hosts=None) as conn:
        await conn.run("apt update -y && apt install -y python3 python3-pip")

        for d in APP_DIRS:
            await asyncssh.scp(d, (conn, d), recurse=True)
        for f in APP_FILES:
            target_name = "main.py" if "server_app.py" in f else Path(f).name
            await asyncssh.scp(f, (conn, target_name))

        await conn.run("pip install -r requirements.txt")
        # Ensure uvicorn, fastapi, etc. are in requirements.txt
        conn.create_process(
            "nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &"
        )
    return ip


async def send_requests(ips):
    async with aiohttp.ClientSession() as session:
        while True:
            for ip in ips:
                try:
                    async with session.get(f"http://{ip}:8000") as resp:
                        text = await resp.text()
                        print(f"[{ip}] {resp.status}: {text[:60]}")
                except Exception as e:
                    print(f"[{ip}] Error: {e}")
                await asyncio.sleep(1)


async def main():
    print("Creating servers...")
    servers = []
    for i in range(N_SERVERS):
        resp = client.servers.create(
            name=f"fastapi-node-{i}",
            server_type=ServerType(name=SERVER_TYPE),
            image=Image(name=IMAGE),
        )
        servers.append(resp.server)
    print("Waiting for servers...")
    client.servers.wait_for_actions([resp.action_index for resp in servers])

    servers_data = []
    for s in servers:
        ip = await setup_server(s)
        servers_data.append({"id": s.id, "ip": ip})

    ips = [s["ip"] for s in servers_data]
    print("Servers ready:", ips)

    with open("servers.json", "w") as f:
        json.dump(servers_data, f)

    await send_requests(ips)


if __name__ == "__main__":
    asyncio.run(main())
