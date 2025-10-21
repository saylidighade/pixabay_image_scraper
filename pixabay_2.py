import requests
import time
import json
import math
import os
from pathlib import Path
from random import uniform

API_KEY = '52821590-15e7832576f29584d27ead552'
API_URL = "https://pixabay.com/api/"
PER_PAGE = 200               # max allowed by API
RATE_LIMIT_DELAY = 0.8       # seconds between requests (adjust conservatively)
MAX_RETRIES = 5
CHECKPOINT_FILE = "pixabay_checkpoint.json"
OUTPUT_METADATA = "pixabay_results.jsonl"  # newline-delimited JSON (appendable)
DOWNLOAD_IMAGES = True      # set True if you want the script to also download images
DOWNLOAD_DIR = "pixabay_images"

# search building blocks tuned for "retail ecommerce beauty cosmetics"
seed_keywords = [
    "retail ecommerce beauty cosmetics",
    "beauty product",
    "cosmetics flatlay",
    "makeup",
    "skincare",
    "lipstick",
    "foundation",
    "cosmetics packaging",
    "beauty ecommerce",
    "online beauty store",
    "cosmetics display",
    "beauty model",
    "skincare routine",
    "cosmetic bottle",
    "beauty influencer"
]

# modifiers to slice the result space; many combinations will be generated.
colors = ["", "black", "white", "red", "pink", "gold", "blue", "green"]
orientations = ["all"]
image_types = ["photo"]
orders = ["popular", "latest"]
min_widths = [0]  # helps split images by resolution

# small helper functions
def safe_request(params):
    """Request with retry and backoff. Returns JSON or raises."""
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            # handle rate-limit-style or 400 errors gracefully
            if r.status_code == 400:
                raise ValueError(f"Bad request (400). Params: {params}")
            r.raise_for_status()
        except Exception as e:
            attempt += 1
            sleep = (2 ** attempt) + uniform(0, 0.5)
            print(f"[WARN] request failed (attempt {attempt}/{MAX_RETRIES}): {e}. sleeping {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts. Last params: {params}")


def write_checkpoint(state):
    with open(CHECKPOINT_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(CHECKPOINT_FILE + ".tmp", CHECKPOINT_FILE)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def append_metadata(hit):
    # store each hit as one JSON line so file can be appended/resumed easily
    with open(OUTPUT_METADATA, "a", encoding="utf-8") as f:
        f.write(json.dumps(hit, ensure_ascii=False) + "\n")


def download_image(url, dest_path):
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as w:
            for chunk in r.iter_content(1024 * 32):
                if chunk:
                    w.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] failed to download {url}: {e}")
        return False


def build_query_list():
    combos = []
    # create combinations but avoid the full cartesian explosion by limiting some combos:
    for kw in seed_keywords:
        for color in colors:
            for orient in orientations:
                for itype in image_types:
                    for order in orders:
                        for w in min_widths:
                            # skip the trivial all-empty combo
                            if not (color or orient or itype or w):
                                combos.append({"q": kw, "colors": "", "orientation": "", "image_type": "", "order": order, "min_width": 0})
                            else:
                                combos.append({"q": kw, "colors": color, "orientation": orient, "image_type": itype, "order": order, "min_width": w})
    # dedupe combos by string representation and return
    unique = []
    seen = set()
    for c in combos:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    print(f"[INFO] built {len(unique)} distinct queries")
    return unique


def normalize_params(c):
    params = {
        "key": API_KEY,
        "q": c["q"],
        "per_page": PER_PAGE,
        "safesearch": "true",
        "page": 1,
        "order": c.get("order", "popular"),
    }
    if c.get("colors"):
        params["colors"] = c["colors"]
    if c.get("orientation"):
        params["orientation"] = c["orientation"]
    if c.get("image_type"):
        params["image_type"] = c["image_type"]
    if c.get("min_width", 0):
        params["min_width"] = c["min_width"]
    return params


def harvest_all():
    if API_KEY == "YOUR_PIXABAY_KEY":
        raise RuntimeError("Please set your PIXABAY API key in API_KEY before running.")

    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    checkpoint = load_checkpoint() or {"processed": [], "seen_ids": [], "stats": {"collected": 0}}
    seen_ids = set(checkpoint.get("seen_ids", []))
    processed = set(checkpoint.get("processed", []))

    queries = build_query_list()

    try:
        for idx, combo in enumerate(queries):
            combo_key = json.dumps(combo, sort_keys=True)
            if combo_key in processed:
                continue

            params = normalize_params(combo)
            print(f"\n[INFO] Query {idx+1}/{len(queries)}: {params['q'][:60]} | colors={params.get('colors','')} | orient={params.get('orientation','')} | type={params.get('image_type','')} | mw={params.get('min_width',0)} | order={params.get('order')}")

            # initial request to get totalHits
            data = safe_request(params)
            total_hits = data.get("totalHits", 0)
            accessible = min(total_hits, 500)  # Pixabay cap per query
            if accessible == 0:
                print("[INFO] no hits for this slice.")
                processed.add(combo_key)
                checkpoint["processed"] = list(processed)
                write_checkpoint({**checkpoint, "processed": list(processed), "seen_ids": list(seen_ids)})
                time.sleep(RATE_LIMIT_DELAY)
                continue

            max_pages = math.ceil(accessible / PER_PAGE)
            print(f"[INFO] totalHits={total_hits} (accessible={accessible}), will fetch up to {max_pages} pages")

            # iterate pages (we already have page=1 data)
            for page in range(1, max_pages + 1):
                params["page"] = page
                data = safe_request(params)
                hits = data.get("hits", [])
                if not hits:
                    print(f"[INFO] page {page} returned 0 hits; breaking.")
                    break
                new_count = 0
                for h in hits:
                    hid = h.get("id")
                    if hid is None:
                        continue
                    if hid in seen_ids:
                        continue
                    seen_ids.add(hid)
                    append_metadata(h)
                    new_count += 1
                    # optional download (largeImageURL or fullHDURL depending on availability)
                    if DOWNLOAD_IMAGES:
                        url = h.get("largeImageURL") or h.get("webformatURL")
                        if url:
                            ext = os.path.splitext(url)[1].split("?")[0] or ".jpg"
                            fname = f"{hid}{ext}"
                            dest = os.path.join(DOWNLOAD_DIR, fname)
                            ok = download_image(url, dest)
                            if not ok:
                                # don't fail; continue
                                pass
                    checkpoint["stats"]["collected"] = checkpoint["stats"].get("collected", 0) + new_count

                print(f"[INFO] page {page}: fetched {len(hits)} hits, added {new_count} new unique images (total unique so far: {len(seen_ids)})")
                # polite delay between page requests
                time.sleep(RATE_LIMIT_DELAY)

            # finished this combo
            processed.add(combo_key)
            checkpoint["processed"] = list(processed)
            checkpoint["seen_ids"] = list(seen_ids)
            write_checkpoint(checkpoint)
            # short random sleep between combos to be polite
            time.sleep(RATE_LIMIT_DELAY + uniform(0, 0.6))

    except KeyboardInterrupt:
        print("[WARN] interrupted by user; saving checkpoint.")
        checkpoint["processed"] = list(processed)
        checkpoint["seen_ids"] = list(seen_ids)
        write_checkpoint(checkpoint)

    print(f"[DONE] Harvest complete (unique images collected: {len(seen_ids)})")
    return {"collected": len(seen_ids)}

if __name__ == "__main__":
    res = harvest_all()
    print(res)
