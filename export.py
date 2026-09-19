#!/usr/bin/env python3
"""Export erovi.db to CSV / JSONL / SQL dump.

  python export.py csv    videos.csv
  python export.py jsonl  videos.jsonl
  python export.py sql    erovi_dump.sql        # full schema+data SQL dump
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).with_name("erovi.db")

QUERY = """
SELECT v.cid, v.product_code, v.url, v.title_ja, v.title_en, v.storyline_ja, v.storyline_en,
       v.release_date, v.runtime_min, v.director_ja, v.director_en, v.series_ja, v.series_en,
       v.maker_ja, v.maker_en, v.label_ja, v.label_en, v.cover_url, v.rating,
       (SELECT group_concat(a.name_ja, ' | ') FROM video_actresses va JOIN actresses a ON a.id=va.actress_id WHERE va.cid=v.cid) AS actresses_ja,
       (SELECT group_concat(a.name_en, ' | ') FROM video_actresses va JOIN actresses a ON a.id=va.actress_id WHERE va.cid=v.cid) AS actresses_en,
       (SELECT group_concat(g.name_ja, ' | ') FROM video_genres vg JOIN genres g ON g.id=vg.genre_id WHERE vg.cid=v.cid) AS genres_ja,
       (SELECT group_concat(g.name_en, ' | ') FROM video_genres vg JOIN genres g ON g.id=vg.genre_id WHERE vg.cid=v.cid) AS genres_en
FROM videos v ORDER BY v.release_date, v.cid
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fmt, out = sys.argv[1], Path(sys.argv[2])
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    if fmt == "sql":
        with out.open("w", encoding="utf-8") as fh:
            for line in conn.iterdump():
                fh.write(line + "\n")
    else:
        cur = conn.execute(QUERY)
        cols = [d[0] for d in cur.description]
        with out.open("w", encoding="utf-8", newline="") as fh:
            if fmt == "csv":
                w = csv.writer(fh)
                w.writerow(cols)
                for row in cur:
                    w.writerow(list(row))
            elif fmt == "jsonl":
                for row in cur:
                    fh.write(json.dumps(dict(zip(cols, row)), ensure_ascii=False) + "\n")
            else:
                sys.exit("format must be csv | jsonl | sql")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

