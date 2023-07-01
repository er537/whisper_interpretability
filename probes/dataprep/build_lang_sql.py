import sqlite3 as sqlite
import logging
import fire


def init_sql(sql_path):
    with sqlite.connect(f"{sql_path}") as conn:
        create_str = (
            "CREATE TABLE IF NOT EXISTS "
            "data(key TEXT PRIMARY KEY, audio_path TEXT, label TEXT, start_time FLOAT, end_time FLOAT)"
        )
        logging.info("Generating SQLITE db from utterances")
        cur = conn.cursor()
        cur.execute(create_str)


def append_to_sql_cache(dblx_path: str, lang: str, sql_path: str, total_entries):
    """
    Create sql cache containing audio_path, label, start_time, end_time
    """
    with open(dblx_path, "r") as f, sqlite.connect(f"{sql_path}") as conn:
        cur = conn.cursor()
        num_entries = 0
        while True:
            line = f.readline()
            if not line:
                break
            id, audio_path, start_time, end_time, *_ = line.split(" ")
            cur.execute(
                "INSERT INTO data VALUES(?,?,?,?,?);",
                (
                    f"{total_entries+num_entries}",
                    audio_path,
                    lang,
                    float(start_time),
                    float(end_time),
                ),
            )
            num_entries += 1

    return num_entries


def build_sql_table(
    sql_path: str,
    dblx_paths: list = [
        "/home/ellenar/probes/fr_val_cut.dblx",
        "/home/ellenar/probes/de_val_cut.dblx",
    ],
    langs: list = ["fr", "de"],
):
    init_sql(sql_path)
    total_entries = 0
    for dblx_path, lang in zip(dblx_paths, langs):
        total_entries += append_to_sql_cache(dblx_path, lang, sql_path, total_entries)

    print(total_entries)


if __name__ == "__main__":
    fire.Fire(build_sql_table)
