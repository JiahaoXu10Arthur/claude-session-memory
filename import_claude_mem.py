import json
import sqlite3
import sys
import tomllib
from pathlib import Path

import core

SRC = str(Path.home() / ".claude-mem" / "claude-mem.db")
UPTO = 37233  # spec：按 id 排第 37062 条
BATCH = 256


def read_observations(src: str, upto: int) -> list[dict]:
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT id, memory_session_id, project, type, title, subtitle, narrative, "
        "facts, files_read, files_modified, created_at "
        "FROM observations WHERE id <= ? ORDER BY id",
        (upto,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": row[0], "session_id": row[1] or "", "project": row[2] or "",
            "type": row[3] or "", "title": row[4] or "", "subtitle": row[5] or "",
            "narrative": row[6] or "",
            "facts": json.loads(row[7]) if row[7] else [],
            "files": list(dict.fromkeys(
                (json.loads(row[8]) if row[8] else []) + (json.loads(row[9]) if row[9] else [])
            )),
            "created_at": row[10] or "",
        }
        for row in rows
    ]


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    if Path(cfg["db"]).exists():
        sys.exit(f"{cfg['db']} 已经存在；要重灌先删掉它")
    observations = read_observations(SRC, UPTO)
    mem = core.Memory(cfg)
    total = len(observations)
    for i in range(0, total, BATCH):
        mem.add(observations[i:i + BATCH])
        print(f"已灌 {min(i + BATCH, total)}/{total}", flush=True)
    print(mem.store.stats())


if __name__ == "__main__":
    main(sys.argv[1])
