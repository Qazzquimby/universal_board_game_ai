import asyncio

import uvicorn

from remote_play.start import main

if __name__ == "__main__":
    asyncio.run(main(local=True))
    print("Starting uvicorn server...")
    uvicorn.run("remote_play.server_app:app", host="127.0.0.1", port=8000, reload=True)
