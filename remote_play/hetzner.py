from pathlib import Path

my_dir = Path(__file__).parent
HETZNER_KEY = (my_dir / Path("../hetzner_key")).read_text()
