#!/usr/bin/env python3
"""
erovi.jp catalogue crawler
==========================

Enumerates every video on https://erovi.jp via the release-date list pages
(/list/dv_saledate-YYYYMMDD_nN.html), scrapes every item page
(/item/ddv-<cid>.html) and stores the result in a SQLite database.

Design goals
------------
* Resumable: every URL / date is checkpointed, you can Ctrl-C at any time and
  restart with the same command.
* Polite but fast: async HTTP (httpx) with a configurable concurrency limit and
  automatic back-off on 429 / 5xx.
* Storyline (and title / genres) translation JA -> EN is a separate stage
  (translate.py) so it can run in parallel with crawling or be re-run with a
  different engine later.

Usage
-----
  python erovi_crawler.py discover  --from 2002-01-01 --to 2026-09-17
  python erovi_crawler.py scrape    --concurrency 8
  python erovi_crawler.py all       --from 2002-01-01            # both stages
  python erovi_crawler.py stats
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import re
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx
from selectolax.lexbor import LexborHTMLParser

BASE = "https://erovi.jp"
UTC = dt.timezone.utc
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
DB_PATH = Path(__file__).with_name("erovi.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS videos (
    cid            TEXT PRIMARY KEY,          -- e.g. fabs00097
    product_code   TEXT,                      -- e.g. FABS-097
    url            TEXT NOT NULL,
    title_ja       TEXT,
    title_en       TEXT,
    storyline_ja   TEXT,
    storyline_en   TEXT,
    release_date   TEXT,                      -- YYYY-MM-DD
    runtime_min    INTEGER,
    director_ja    TEXT,
    director_en    TEXT,
    series_ja      TEXT,
    series_en      TEXT,
    maker_ja       TEXT,
    maker_en       TEXT,
    label_ja       TEXT,
    label_en       TEXT,
    cover_url      TEXT,
    rating         REAL,                      -- 0-5 stars if present
    scraped_at     TEXT,
    translated_at  TEXT
);

CREATE TABLE IF NOT EXISTS actresses (
    id     INTEGER PRIMARY KEY,               -- erovi/DMM actress id
    name_ja TEXT,
    name_en TEXT
);
CREATE TABLE IF NOT EXISTS video_actresses (
    cid TEXT, actress_id INTEGER,
    PRIMARY KEY (cid, actress_id)
);

CREATE TABLE IF NOT EXISTS genres (
    id     INTEGER PRIMARY KEY,               -- erovi keyword id
    name_ja TEXT,
    name_en TEXT
);
CREATE TABLE IF NOT EXISTS video_genres (
    cid TEXT, genre_id INTEGER,
    PRIMARY KEY (cid, genre_id)
);

-- crawl bookkeeping -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS queue (            -- item URLs discovered, pending scrape
    url        TEXT PRIMARY KEY,
    cid        TEXT,
    status     TEXT DEFAULT 'pending',        -- pending | done | error
    attempts   INTEGER DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);

CREATE TABLE IF NOT EXISTS dates_done (        -- saledate lists fully paged
    date  TEXT PRIMARY KEY,
    items INTEGER,
    pages INTEGER
);
CREATE INDEX IF NOT EXISTS idx_videos_release ON videos(release_date);
CREATE INDEX IF NOT EXISTS idx_videos_untranslated ON videos(translated_at) WHERE translated_at IS NULL;
"""

ITEM_RE = re.compile(r'href="(/item/ddv-([^"/]+)\.html)"')
DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


# ----------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------
def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.executescript(SCHEMA)
    return conn


# ----------------------------------------------------------------------------
# HTTP with retry / back-off
# ----------------------------------------------------------------------------
class Fetcher:
    """httpx AsyncClient with HTTP/2 multiplexing, Brotli, bounded concurrency,
    exponential back-off with jitter and Retry-After support."""

    def __init__(self, concurrency: int, delay: float):
        self.sem = asyncio.Semaphore(concurrency)
        self.delay = delay
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": UA,
                "Accept-Language": "ja,en;q=0.8",
                # server supports br (~20% smaller than gzip); httpx[brotli] decodes it
                "Accept-Encoding": "br, gzip",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            http2=True,
            limits=httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency),
        )
        self.requests = 0
        self.bytes = 0          # bytes on the wire (compressed)
        self.errors = 0

    async def get(self, url: str, retries: int = 6) -> str | None:
        backoff = 2.0
        for _ in range(retries):
            async with self.sem:
                try:
                    r = await self.client.get(url)
                    self.requests += 1
                    self.bytes += r.num_bytes_downloaded
                    if self.delay:
                        await asyncio.sleep(self.delay * random.uniform(0.5, 1.5))
                except (httpx.HTTPError, asyncio.TimeoutError):
                    self.errors += 1
                    await asyncio.sleep(backoff * random.uniform(0.8, 1.2))
                    backoff = min(backoff * 2, 60)
                    continue
            if r.status_code == 200:
                return r.text
            if r.status_code in (404, 410):
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                self.errors += 1
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.isdigit() else backoff
                await asyncio.sleep(wait * random.uniform(0.8, 1.2))
                backoff = min(backoff * 2, 60)
                continue
            return None
        return None

    async def close(self):
        await self.client.aclose()


# ----------------------------------------------------------------------------
# Stage 1: discovery via release-date lists
# ----------------------------------------------------------------------------
def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


async def discover_date(f: Fetcher, conn: sqlite3.Connection, day: dt.date, max_pages=50):
    ymd = day.strftime("%Y%m%d")
    found: dict[str, str] = {}
    page = 1
    while page <= max_pages:
        html = await f.get(f"{BASE}/list/dv_saledate-{ymd}_n{page}.html")
        if not html:
            break
        hits = ITEM_RE.findall(html)
        new = {u: cid for u, cid in hits}
        found.update(new)
        # 100 items per page; fewer than 100 unique => last page
        if len(set(new)) < 100:
            break
        page += 1
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO queue(url, cid) VALUES (?, ?)",
            [(BASE + u, cid) for u, cid in found.items()],
        )
        conn.execute(
            "INSERT OR REPLACE INTO dates_done(date, items, pages) VALUES (?,?,?)",
            (day.isoformat(), len(found), page),
        )
    return len(found)


async def discover(args):
    conn = open_db(args.db)
    start = dt.date.fromisoformat(args.date_from)
    end = dt.date.fromisoformat(args.date_to) if args.date_to else dt.date.today()
    done = {r[0] for r in conn.execute("SELECT date FROM dates_done")}
    todo = [d for d in daterange(start, end) if d.isoformat() not in done]
    print(f"[discover] {len(todo)} days to scan ({start} -> {end}), {len(done)} already done")
    f = Fetcher(args.concurrency, args.delay)
    t0 = time.time()
    c = {"days": 0, "items": 0}
    it = iter(todo)

    async def worker():
        for day in it:                      # shared iterator = simple work-stealing
            c["items"] += await discover_date(f, conn, day)
            c["days"] += 1
            el = time.time() - t0
            rate = c["days"] / el if el else 0
            eta = (len(todo) - c["days"]) / rate if rate else 0
            print(f"\r[discover] {c['days']}/{len(todo)} days  +{c['items']} items  "
                  f"{f.requests} req  {rate*60:.0f} days/min  ETA {eta/60:.1f} min", end="", flush=True)

    try:
        async with asyncio.TaskGroup() as tg:
            for _ in range(args.concurrency):
                tg.create_task(worker())
    finally:
        await f.close()
    q = conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    print(f"\n[discover] finished. queue size = {q}")


# ----------------------------------------------------------------------------
# Stage 2: item page parsing
# ----------------------------------------------------------------------------
def _txt(el) -> str | None:
    if el is None:
        return None
    t = " ".join(el.text(separator=" ", strip=True).split())
    return t if t and t != "--" else None


def _id_from(href: str | None) -> int | None:
    if not href:
        return None
    m = re.search(r"-(\d+)_", href)
    return int(m.group(1)) if m else None


def parse_item(html: str, url: str) -> dict:
    """Parse an item page with selectolax/Lexbor (C HTML5 parser, ~15x faster than bs4)."""
    tree = LexborHTMLParser(html)
    out: dict = {"url": url}

    m = re.search(r"/item/ddv-([^/]+)\.html", url)
    out["cid"] = m.group(1) if m else None

    h1 = tree.css_first("h1")
    title = " ".join(h1.text(separator=" ", strip=True).split()) if h1 else ""
    # strip leading "cid [CODE] " and trailing " @動画"
    title = re.sub(r"^\S+\s+\[[^\]]+\]\s*", "", title)
    title = re.sub(r"\s*@動画\s*$", "", title)
    out["title_ja"] = title or None

    # storyline: first <p> of .captext; drop the first line (repeated title)
    cap = tree.css_first(".captext p")
    if cap:
        # <br> -> newline: unwrap into text nodes
        for br in cap.css("br"):
            br.replace_with("\n")
        lines = [ln.strip() for ln in cap.text(separator="").split("\n")]
        lines = [ln for ln in lines if ln]
        if lines and (out["cid"] or "") and lines[0].startswith(out["cid"]):
            lines = lines[1:]
        out["storyline_ja"] = "\n".join(lines) or None
    else:
        out["storyline_ja"] = None

    # spec table
    actresses, genres = [], []
    for tr in tree.css("table.buy_table tr"):
        tds = tr.css("td")
        if len(tds) < 2:
            continue
        key = tds[0].text(strip=True).rstrip("：:")
        val = tds[1]
        if key == "配信日":
            mm = DATE_RE.search(val.text())
            if mm:
                y, mo, d = map(int, mm.groups())
                out["release_date"] = f"{y:04d}-{mo:02d}-{d:02d}"
        elif key == "収録時間":
            mm = re.search(r"([\d,]+)", val.text())
            out["runtime_min"] = int(mm.group(1).replace(",", "")) if mm else None
        elif key == "女優":
            for a in val.css("a"):
                aid = a.attributes.get("data-actressid") or _id_from(a.attributes.get("href"))
                if aid:
                    actresses.append((int(aid), a.text(strip=True)))
        elif key == "監督":
            out["director_ja"] = _txt(val)
        elif key == "シリーズ":
            out["series_ja"] = _txt(val)
        elif key == "メーカー":
            out["maker_ja"] = _txt(val)
        elif key == "レーベル":
            out["label_ja"] = _txt(val)
        elif key == "ジャンル":
            for a in val.css("a"):
                gid = _id_from(a.attributes.get("href"))
                if gid:
                    genres.append((gid, a.text(strip=True)))
        elif key == "品番":
            code = val.css_first("#chkcd")
            out["product_code"] = code.text(strip=True) if code else None

    # cover image from JSON-LD Product
    out["cover_url"] = None
    for sc in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(sc.text() or "")
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "Product":
            img = data.get("image")
            out["cover_url"] = img[0] if isinstance(img, list) and img else img
            break

    # rating: count filled stars in .ratingdetail (site uses width% style sometimes)
    out["rating"] = None
    rd = tree.css_first(".ratingdetail")
    style = rd.attributes.get("style") if rd else None
    if style:
        mm = re.search(r"width:\s*([\d.]+)%", style)
        if mm:
            out["rating"] = round(float(mm.group(1)) / 20, 2)

    out["actresses"] = actresses
    out["genres"] = genres
    return out


def save_item(conn: sqlite3.Connection, it: dict):
    with conn:
        conn.execute(
            """INSERT INTO videos(cid, product_code, url, title_ja, storyline_ja, release_date,
                   runtime_min, director_ja, series_ja, maker_ja, label_ja, cover_url, rating, scraped_at)
               VALUES(:cid,:product_code,:url,:title_ja,:storyline_ja,:release_date,
                   :runtime_min,:director_ja,:series_ja,:maker_ja,:label_ja,:cover_url,:rating,:scraped_at)
               ON CONFLICT(cid) DO UPDATE SET
                   product_code=excluded.product_code, title_ja=excluded.title_ja,
                   storyline_ja=excluded.storyline_ja, release_date=excluded.release_date,
                   runtime_min=excluded.runtime_min, director_ja=excluded.director_ja,
                   series_ja=excluded.series_ja, maker_ja=excluded.maker_ja, label_ja=excluded.label_ja,
                   cover_url=excluded.cover_url, rating=excluded.rating, scraped_at=excluded.scraped_at""",
            {**{k: it.get(k) for k in (
                "cid", "product_code", "url", "title_ja", "storyline_ja", "release_date",
                "runtime_min", "director_ja", "series_ja", "maker_ja", "label_ja", "cover_url", "rating")},
             "scraped_at": dt.datetime.now(UTC).isoformat(timespec="seconds")},
        )
        conn.executemany("INSERT OR IGNORE INTO actresses(id, name_ja) VALUES(?,?)", it["actresses"])
        conn.executemany("INSERT OR IGNORE INTO video_actresses(cid, actress_id) VALUES(?,?)",
                         [(it["cid"], a) for a, _ in it["actresses"]])
        conn.executemany("INSERT OR IGNORE INTO genres(id, name_ja) VALUES(?,?)", it["genres"])
        conn.executemany("INSERT OR IGNORE INTO video_genres(cid, genre_id) VALUES(?,?)",
                         [(it["cid"], g) for g, _ in it["genres"]])
        conn.execute("UPDATE queue SET status='done' WHERE url=?", (it["url"],))


async def scrape_one(f: Fetcher, conn: sqlite3.Connection, url: str) -> bool:
    html = await f.get(url)
    if html is None:
        with conn:
            conn.execute("UPDATE queue SET status='error', attempts=attempts+1, "
                         "last_error='fetch failed / 404' WHERE url=?", (url,))
        return False
    try:
        item = parse_item(html, url)
        if not item.get("cid"):
            raise ValueError("no cid")
        save_item(conn, item)
        return True
    except Exception as e:  # noqa
        with conn:
            conn.execute("UPDATE queue SET status='error', attempts=attempts+1, last_error=? WHERE url=?",
                         (repr(e)[:300], url))
        return False


async def scrape(args):
    conn = open_db(args.db)
    f = Fetcher(args.concurrency, args.delay)
    stop = asyncio.Event()

    def _sig(*_):
        print("\n[scrape] stopping after in-flight requests...", flush=True)
        stop.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    pending_sql = "status='pending' OR (status='error' AND attempts<3)"
    total_pending = conn.execute(f"SELECT COUNT(*) FROM queue WHERE {pending_sql}").fetchone()[0]
    print(f"[scrape] {total_pending} pages to scrape, concurrency={args.concurrency}")
    t0 = time.time()
    counters = {"done": 0, "ok": 0}
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=args.concurrency * 4)

    async def producer():
        """Stream URLs from SQLite into the queue in pages, never loading all 550k at once."""
        last = ""
        sent = 0
        while not stop.is_set():
            rows = conn.execute(
                f"SELECT url FROM queue WHERE ({pending_sql}) AND url > ? ORDER BY url LIMIT 500",
                (last,)).fetchall()
            if not rows:
                break
            for (url,) in rows:
                if stop.is_set() or (args.limit and sent >= args.limit):
                    break
                await q.put(url)
                sent += 1
                last = url
            else:
                continue
            break
        for _ in range(args.concurrency):
            await q.put(None)

    async def worker():
        while True:
            url = await q.get()
            if url is None:
                return
            ok = await scrape_one(f, conn, url)
            counters["done"] += 1
            counters["ok"] += ok
            if counters["done"] % 20 == 0:
                el = time.time() - t0
                rate = counters["done"] / el if el else 0
                eta = (total_pending - counters["done"]) / rate if rate else 0
                print(f"\r[scrape] {counters['done']}/{total_pending}  ok={counters['ok']}  "
                      f"{rate:.2f} pages/s  {f.bytes/1e6:.0f} MB wire  err={f.errors}  "
                      f"ETA {eta/3600:.2f} h", end="", flush=True)

    try:
        async with asyncio.TaskGroup() as tg:      # Python 3.11+ structured concurrency
            tg.create_task(producer())
            for _ in range(args.concurrency):
                tg.create_task(worker())
    finally:
        await f.close()
    print(f"\n[scrape] finished: {counters['ok']}/{counters['done']} ok in {(time.time()-t0)/60:.1f} min")


# ----------------------------------------------------------------------------
def stats(args):
    conn = open_db(args.db)
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa
    print("dates scanned     :", q("SELECT COUNT(*) FROM dates_done"))
    print("queue pending     :", q("SELECT COUNT(*) FROM queue WHERE status='pending'"))
    print("queue error       :", q("SELECT COUNT(*) FROM queue WHERE status='error'"))
    print("videos scraped    :", q("SELECT COUNT(*) FROM videos"))
    print("  with storyline  :", q("SELECT COUNT(*) FROM videos WHERE storyline_ja IS NOT NULL"))
    print("  translated      :", q("SELECT COUNT(*) FROM videos WHERE translated_at IS NOT NULL"))
    print("actresses         :", q("SELECT COUNT(*) FROM actresses"))
    print("genres            :", q("SELECT COUNT(*) FROM genres"))
    print("storyline JA chars:", q("SELECT COALESCE(SUM(LENGTH(storyline_ja)),0) FROM videos"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["discover", "scrape", "all", "stats"])
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--from", dest="date_from", default="2002-01-01")
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--delay", type=float, default=0.0, help="extra per-request sleep (s)")
    p.add_argument("--limit", type=int, default=0, help="scrape at most N pages (testing)")
    args = p.parse_args()

    if args.cmd == "stats":
        stats(args)
    elif args.cmd == "discover":
        asyncio.run(discover(args))
    elif args.cmd == "scrape":
        asyncio.run(scrape(args))
    elif args.cmd == "all":
        asyncio.run(discover(args))
        asyncio.run(scrape(args))


if __name__ == "__main__":
    main()

