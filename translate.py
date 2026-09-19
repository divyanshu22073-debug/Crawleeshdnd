#!/usr/bin/env python3
"""
Translate Japanese fields in erovi.db to English.

Engines
-------
  google  : free, no key, via deep-translator (Google web endpoint). ~5k chars/req.
            Rate-limited by Google; ok for a few hundred thousand requests/day
            if you keep concurrency low (1-3) – risk of temporary IP blocks.
  openai  : OpenAI-compatible chat model (default gpt-5-mini, override with
            --model or OPENAI_MODEL). Best quality for
            adult-content storylines (idiomatic, keeps product codes intact).
            Needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL).
  deepl   : DeepL API. Needs DEEPL_API_KEY. Free tier = 500k chars/month.

Usage
-----
  python translate.py --engine google  --workers 2
  python translate.py --engine openai  --workers 8 --model gpt-4o-mini
  python translate.py --engine deepl
  python translate.py --engine google --only-titles      # skip storylines

All translations are cached in table `tcache(src, engine) -> dst` so that
identical strings (genre names, maker names, boilerplate) are only paid once.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).with_name("erovi.db")

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tcache (
    h      TEXT PRIMARY KEY,   -- sha1(engine + src)
    engine TEXT,
    src    TEXT,
    dst    TEXT
);
"""

SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator for adult-video catalogue "
    "metadata. Translate the user's text faithfully and idiomatically. Keep product codes "
    "(e.g. FAX-132, AOFR-004), numbers, durations and proper names intact; romanize "
    "Japanese personal names (surname first, e.g. 'Tsukamoto Henry' -> 'Henry Tsukamoto' "
    "is fine when it is a known stage name). Preserve line breaks. Output ONLY the translation."
)


# ----------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------
class GoogleEngine:
    """Free Google Translate web endpoint (client=gtx). No key required.

    Google enforces roughly 5 req/s and ~200k req/day per IP. Keep --workers <= 3
    and, for the full catalogue, spread the run over a few days or rotate IPs.
    """
    name = "google"
    max_chars = 4500
    URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, rps: float = 2.0):
        import httpx
        self.client = httpx.AsyncClient(
            http2=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0"},
            timeout=30,
        )
        # token bucket: smooth requests to `rps`/s instead of bursting
        self._rps = rps
        self._tokens = rps
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self._cooldown_until = 0.0     # shared pause after a 429 (all workers wait)

    async def _throttle(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                await asyncio.sleep(self._cooldown_until - now)
                now = time.monotonic()
            self._tokens = min(self._rps, self._tokens + (now - self._last) * self._rps)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rps
                await asyncio.sleep(wait)
                self._tokens = 0
                self._last = time.monotonic()
            else:
                self._tokens -= 1

    async def _one(self, text: str) -> str:
        params = {"client": "gtx", "sl": "ja", "tl": "en", "dt": "t", "q": text}
        backoff = 10.0
        for attempt in range(8):
            await self._throttle()
            r = await self.client.get(self.URL, params=params)
            if r.status_code == 200:
                data = r.json()
                return "".join(seg[0] for seg in data[0] if seg and seg[0])
            if r.status_code in (429, 403, 503):
                # global cooldown so every worker backs off together
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + backoff)
                print(f"\n[google] {r.status_code} - cooling down {backoff:.0f}s", file=sys.stderr)
                backoff = min(backoff * 2, 300)
                continue
            raise RuntimeError(f"google {r.status_code}: {r.text[:120]}")
        raise RuntimeError("google: rate limited, giving up for now")

    # Google's free endpoint silently drops content from very long run-on
    # "sentences" (e.g. lists of product codes with no punctuation). Split such
    # lines into <= LINE_MAX char pieces at Japanese punctuation / spaces first.
    LINE_MAX = 250
    # split before product codes like "FAX-132" / "X-1049" (they are usually
    # glued to the previous title with no punctuation), and after sentence ends
    _CODE_RE = __import__("re").compile(r"(?=(?<![A-Za-z0-9-])[A-Z]{1,6}-\d{2,5}\b)")
    _SENT_RE = __import__("re").compile(r"(?<=[。！？!?」』…])")

    @classmethod
    def _split_line(cls, line: str) -> list[str]:
        codes = [p for p in cls._CODE_RE.split(line) if p and p.strip()]
        if len(codes) <= 1 and len(line) <= cls.LINE_MAX:
            return [line]
        parts = []
        for seg in codes:
            if len(seg) <= cls.LINE_MAX:
                parts.append(seg)
                continue
            cur = ""
            for piece in cls._SENT_RE.split(seg):
                if not piece:
                    continue
                if cur and len(cur) + len(piece) > cls.LINE_MAX:
                    parts.append(cur)
                    cur = piece
                else:
                    cur += piece
            if cur:
                parts.append(cur)
        # hard split anything still too long
        out = []
        for p in parts:
            while len(p) > cls.LINE_MAX:
                out.append(p[:cls.LINE_MAX])
                p = p[cls.LINE_MAX:]
            out.append(p)
        return out

    async def translate(self, text: str) -> str:
        """Translate a multi-line text in as few requests as possible.

        Every line is split into safe pieces; all pieces are sent in ONE request
        separated by newlines (Google keeps newlines as segment boundaries). If
        the number of returned lines does not match, fall back to one request per
        piece. Pieces of the same original line are re-joined with a space.
        """
        lines = text.split("\n")
        plan = [self._split_line(ln) if ln.strip() else [] for ln in lines]
        flat = [p for pieces in plan for p in pieces]
        if not flat:
            return text

        results: list[str] | None = None
        if sum(len(p) for p in flat) + len(flat) <= self.max_chars:
            joined = await self._one("\n".join(flat))
            got = joined.split("\n")
            if len(got) == len(flat):
                results = got
        if results is None:  # fallback: piece by piece (parallel)
            results = list(await asyncio.gather(*(self._one(p) for p in flat)))

        out, i = [], 0
        for pieces in plan:
            if not pieces:
                out.append("")
                continue
            out.append(" ".join(r.strip() for r in results[i:i + len(pieces)]))
            i += len(pieces)
        return "\n".join(out)


class OpenAIEngine:
    """Official `openai` SDK (AsyncOpenAI). Uses the modern Responses API and
    falls back to Chat Completions for OpenAI-compatible proxies that lack it.
    Honors OPENAI_API_KEY / OPENAI_BASE_URL; SDK handles retries + back-off.
    """
    name = "openai"

    def __init__(self, model: str):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            sys.exit("pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        self.model = model
        self.client = AsyncOpenAI(max_retries=5, timeout=120)
        self.in_tokens = self.out_tokens = 0
        self._use_responses = True

    @staticmethod
    def _check_proxy_refusal(text: str, usage_total: int | None):
        # Some proxies answer HTTP 200 with a billing notice instead of a translation.
        if usage_total == 0 or "credits can't be used" in text:
            raise RuntimeError(f"LLM proxy refused request: {text[:160]}")

    async def translate(self, text: str) -> str:
        if self._use_responses:
            try:
                r = await self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=text,
                    store=False,
                )
                out = (r.output_text or "").strip()
                u = r.usage
                self._check_proxy_refusal(out, getattr(u, "total_tokens", None))
                self.in_tokens += getattr(u, "input_tokens", 0) or 0
                self.out_tokens += getattr(u, "output_tokens", 0) or 0
                return out
            except Exception as e:  # noqa - proxy without /responses -> fall back once
                from openai import NotFoundError, BadRequestError
                if isinstance(e, (NotFoundError, BadRequestError)):
                    self._use_responses = False
                else:
                    raise
        r = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
        )
        out = (r.choices[0].message.content or "").strip()
        u = r.usage
        self._check_proxy_refusal(out, getattr(u, "total_tokens", None))
        self.in_tokens += getattr(u, "prompt_tokens", 0) or 0
        self.out_tokens += getattr(u, "completion_tokens", 0) or 0
        return out


class DeepLEngine:
    name = "deepl"

    def __init__(self):
        import httpx
        self.key = os.environ.get("DEEPL_API_KEY")
        if not self.key:
            sys.exit("DEEPL_API_KEY not set")
        host = "api-free.deepl.com" if self.key.endswith(":fx") else "api.deepl.com"
        self.url = f"https://{host}/v2/translate"
        self.client = httpx.AsyncClient(timeout=60)

    async def translate(self, text: str) -> str:
        for attempt in range(6):
            r = await self.client.post(
                self.url,
                headers={"Authorization": f"DeepL-Auth-Key {self.key}"},
                data={"text": text, "source_lang": "JA", "target_lang": "EN-US"},
            )
            if r.status_code == 200:
                return r.json()["translations"][0]["text"]
            if r.status_code in (429, 456, 500, 503):
                await asyncio.sleep(2 ** attempt + 1)
                continue
            raise RuntimeError(f"deepl {r.status_code}: {r.text[:200]}")
        raise RuntimeError("deepl: too many retries")


# ----------------------------------------------------------------------------
class Translator:
    def __init__(self, conn: sqlite3.Connection, engine, workers: int):
        self.conn = conn
        self.engine = engine
        self.sem = asyncio.Semaphore(workers)
        self.calls = 0
        self.chars = 0
        conn.executescript(CACHE_SCHEMA)

    def _h(self, s: str) -> str:
        return hashlib.sha1(f"{self.engine.name}\0{s}".encode()).hexdigest()

    async def tr(self, s: str | None) -> str | None:
        if not s or not s.strip():
            return s
        h = self._h(s)
        row = self.conn.execute("SELECT dst FROM tcache WHERE h=?", (h,)).fetchone()
        if row:
            return row[0]
        async with self.sem:
            dst = await self.engine.translate(s)
        self.calls += 1
        self.chars += len(s)
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO tcache(h, engine, src, dst) VALUES(?,?,?,?)",
                              (h, self.engine.name, s, dst))
        return dst


async def translate_lookups(t: Translator, table: str):
    rows = t.conn.execute(f"SELECT id, name_ja FROM {table} WHERE name_en IS NULL AND name_ja IS NOT NULL").fetchall()
    if not rows:
        return
    print(f"[translate] {table}: {len(rows)} names")
    res = await asyncio.gather(*(t.tr(n) for _, n in rows))
    with t.conn:
        t.conn.executemany(f"UPDATE {table} SET name_en=? WHERE id=?", [(en, i) for (i, _), en in zip(rows, res)])


async def translate_video(t: Translator, row, only_titles: bool):
    cid, title, story, director, series, maker, label = row
    title_en = await t.tr(title)
    story_en = None if only_titles else await t.tr(story)
    director_en, series_en, maker_en, label_en = await asyncio.gather(
        t.tr(director), t.tr(series), t.tr(maker), t.tr(label))
    with t.conn:
        t.conn.execute(
            """UPDATE videos SET title_en=?, storyline_en=COALESCE(?, storyline_en), director_en=?, series_en=?,
                                 maker_en=?, label_en=?, translated_at=? WHERE cid=?""",
            (title_en, story_en, director_en, series_en, maker_en, label_en,
             dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), cid))


async def run(args):
    conn = sqlite3.connect(args.db, timeout=60)
    if args.engine == "google":
        engine = GoogleEngine(rps=args.rps)
    elif args.engine == "openai":
        engine = OpenAIEngine(args.model)
    else:
        engine = DeepLEngine()
    t = Translator(conn, engine, args.workers)

    await translate_lookups(t, "genres")
    await translate_lookups(t, "actresses")

    total = conn.execute("SELECT COUNT(*) FROM videos WHERE translated_at IS NULL").fetchone()[0]
    print(f"[translate] {total} videos to translate with {engine.name} (workers={args.workers})")
    t0 = time.time()
    c = {"done": 0, "err": 0}
    q: asyncio.Queue = asyncio.Queue(maxsize=args.workers * 4)

    async def producer():
        last, sent = "", 0
        while True:
            rows = conn.execute(
                "SELECT cid, title_ja, storyline_ja, director_ja, series_ja, maker_ja, label_ja "
                "FROM videos WHERE translated_at IS NULL AND cid > ? ORDER BY cid LIMIT 500", (last,)).fetchall()
            if not rows:
                break
            for row in rows:
                if args.limit and sent >= args.limit:
                    break
                await q.put(row)
                sent += 1
                last = row[0]
            else:
                continue
            break
        for _ in range(args.workers):
            await q.put(None)

    async def worker():
        while (row := await q.get()) is not None:
            try:
                await translate_video(t, row, args.only_titles)
            except Exception as e:  # noqa - keep going, row stays untranslated for next run
                c["err"] += 1
                print(f"\n[translate] {row[0]}: {e!r}", file=sys.stderr)
                if c["err"] > 50 and c["err"] > c["done"]:
                    raise SystemExit("too many consecutive translation errors - aborting")
            c["done"] += 1
            if c["done"] % 10 == 0:
                el = time.time() - t0
                rate = c["done"] / el if el else 0
                eta = (total - c["done"]) / rate if rate else 0
                extra = f"  tokens in/out={engine.in_tokens}/{engine.out_tokens}" if isinstance(engine, OpenAIEngine) else ""
                print(f"\r[translate] {c['done']}/{total}  {rate*60:.0f} videos/min  api calls={t.calls}  "
                      f"chars={t.chars}  err={c['err']}  ETA {eta/3600:.2f} h{extra}", end="", flush=True)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        for _ in range(args.workers):
            tg.create_task(worker())
    print(f"\n[translate] done: {c['done']} videos ({c['err']} errors) in {(time.time()-t0)/60:.1f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--engine", choices=["google", "openai", "deepl"], default="google")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                   help="OpenAI model (gpt-5-mini / gpt-5-nano / gpt-4.1-mini ...)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rps", type=float, default=2.0,
                   help="google engine only: max requests/second (Google tolerates ~2-4 sustained)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-titles", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()

