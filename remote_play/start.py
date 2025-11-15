from __future__ import annotations
import asyncio
import aiohttp
import asyncssh
import time
import json
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
]
APP_DIRS = [
    "agents",
    "algorithms",
    "core",
    "environments",
    "models",
    "utils",
    "remote_play",
]
SERVER_TYPE = "cx23"
IMAGE = "ubuntu-24.04"

SSH_KEYS = [str(Path.home() / ".ssh" / "hetzner")]


async def wait_for_ssh(ip: str, port: int = 22, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with asyncssh.connect(
                ip, username="root", known_hosts=None, client_keys=SSH_KEYS
            ):
                return True
        except Exception:
            await asyncio.sleep(5)
    raise TimeoutError(f"SSH not available on {ip}")


async def setup_server(server):
    ip = server.public_net.ipv4.ip
    await wait_for_ssh(ip)
    async with asyncssh.connect(
        ip, username="root", known_hosts=None, client_keys=SSH_KEYS
    ) as conn:
        update_result = await conn.run(
            "apt update -y && apt install -y python3 python3-venv python3-pip"
        )
        print("update")  # , update_result.stdout, update_result.stderr)

        print("copying files")
        for d in APP_DIRS:
            await asyncssh.scp(d, (conn, d), recurse=True)
        for f in APP_FILES:
            target_name = "main.py" if "server_app.py" in f else Path(f).name
            await asyncssh.scp(f, (conn, target_name))
        await conn.run("touch .git")

        app_env_result = await conn.run("python3 -m venv /opt/appenv")
        install_result = await conn.run(
            "/opt/appenv/bin/pip install -r requirements.txt"
        )
        run_result = await conn.run(
            "nohup /opt/appenv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &",
            check=False,
        )

        # install_result = await conn.run(
        #     "pip install --break-system-packages -r requirements.txt"
        # )
        # print("install", install_result.stdout, install_result.stderr)
        #
        # run_result = await conn.run(
        #     "nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &",
        #     check=False,
        # )
        # print("run", run_result.stdout, run_result.stderr)
    return ip


async def send_requests(ips):
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.sleep(30)
            for ip in ips:
                try:
                    async with session.get(f"http://{ip}:8000") as resp:
                        text = await resp.text()
                        print(f"[{ip}] {resp.status}: {text[:60]}")
                except Exception as e:
                    print(f"[{ip}] Error: {e}")


async def main(local: bool = False):
    if local:
        servers_data = [{"id": "local-0", "ip": "127.0.0.1"}]
        with open("servers.json", "w") as f:
            json.dump(servers_data, f)
        print("Created servers.json for local server at http://127.0.0.1:8000")
        return

    print("Creating servers...")
    create_responses = []
    for i in range(N_SERVERS):
        create_response = client.servers.create(
            name=f"fastapi-node-{i}",
            server_type=ServerType(name=SERVER_TYPE),
            image=Image(name=IMAGE),
            ssh_keys=[client.ssh_keys.get_by_name("hetzner")],
        )
        create_responses.append(create_response)
        print(f"Server password: {create_response.root_password}")

    print("Waiting for servers...")
    servers_data = []
    for create_response in create_responses:
        create_response.action.wait_until_finished()
        ip = await setup_server(create_response.server)
        servers_data.append({"id": create_response.server.id, "ip": ip})

    ips = [s["ip"] for s in servers_data]
    print("Servers ready:", ips)

    with open("servers.json", "w") as f:
        json.dump(servers_data, f)

    print("ssh -i C:/users/user/.ssh/hetzner root@a.b.c.d")
    print("cat app.log")
    print("Ready!")

    await send_requests(ips)
