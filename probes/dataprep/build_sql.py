import sqlite3 as sqlite
import logging
import fire


def create_sql_cache(dblx_path, sql_path):
    """
    Create sql cache containing audio_path, label, start_time, end_time
    """
    with open(dblx_path, "r") as f, sqlite.connect(f"{sql_path}") as conn:
        num_entries = 0
        create_str = (
            "CREATE TABLE IF NOT EXISTS "
            "data(key TEXT PRIMARY KEY, audio_path TEXT, label TEXT, start_time FLOAT, end_time FLOAT)"
        )
        logging.info("Generating SQLITE db from utterances")
        cur = conn.cursor()
        cur.execute(create_str)
        while True:
            line = f.readline()
            if not line:
                break
            _, audio_path, start_time, end_time, label = line.split(" ")
            if (
                audio_path
                == "/data/audio/mirrored/en/mirror_wav/cantab/data/AMI/processed/data-joined/IS1002b.Mix-Headset.wav.wav"
            ):
                continue
            label = label.strip("\n")
            cur.execute(
                "INSERT INTO data VALUES(?,?,?,?,?);",
                (
                    f"{num_entries}",
                    audio_path,
                    label,
                    float(start_time),
                    float(end_time),
                ),
            )
            num_entries += 1

    print(num_entries)


if __name__ == "__main__":
    fire.Fire(create_sql_cache)
