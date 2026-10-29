"""
phishjob_features.py
Ekstraksi 23 fitur URL untuk model LightGBM (Tahap 1) dan cabang numerik autoencoder (Tahap 2).
Port dari extract_features() pada app.R; urutan fitur mengikuti phishjob_lgbm_meta.json.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

POPULAR_TLDS = frozenset([
    "ac", "academy", "actor", "ad", "ae", "aero", "af", "africa", "ag", "ai", "al", "am",
    "amazon", "ao", "app", "apple", "aq", "ar", "arpa", "art", "as", "asia", "at", "au",
    "auction", "audi", "audio", "autos", "aws", "ax", "az", "ba", "baby", "bar", "basketball",
    "bd", "be", "beauty", "beer", "best", "bet", "bf", "bg", "bh", "bi", "bid", "bike",
    "bingo", "bio", "biz", "bj", "black", "blog", "blue", "bn", "bo", "boats", "bond",
    "boston", "bot", "box", "br", "bradesco", "bs", "bt", "build", "business", "buzz", "bw",
    "by", "bz", "ca", "cab", "cafe", "cam", "camera", "capital", "car", "care", "casa", "cash",
    "casino", "cat", "cc", "cd", "center", "ceo", "cern", "cf", "cfd", "cg", "ch", "chat",
    "chrome", "ci", "city", "cl", "click", "clinic", "cloud", "club", "cm", "cn", "co", "com",
    "community", "company", "cool", "coop", "courses", "cr", "credit", "cu", "cv", "cx", "cy",
    "cymru", "cyou", "cz", "dad", "dating", "day", "de", "delivery", "desi", "dev", "dhl",
    "digital", "direct", "directory", "dj", "dk", "dm", "do", "domains", "download", "dz",
    "earth", "ec", "edeka", "edu", "ee", "eg", "email", "energy", "engineering", "es", "et",
    "eu", "eus", "events", "exchange", "expert", "fail", "faith", "family", "fans", "farm",
    "fi", "finance", "fit", "fj", "fm", "fo", "food", "forum", "foundation", "fox", "fr",
    "fujitsu", "fun", "futbol", "fyi", "ga", "gal", "gallery", "game", "games", "garden", "gd",
    "gdn", "ge", "gf", "gg", "gh", "gi", "gl", "gle", "global", "globo", "gm", "gmbh",
    "godaddy", "gold", "golf", "goog", "google", "gov", "gp", "gr", "green", "group", "gs",
    "gt", "guru", "gy", "hair", "haus", "health", "help", "hk", "hn", "homes", "honda",
    "horse", "host", "hosting", "hot", "how", "hr", "ht", "hu", "icu", "id", "ie", "il", "im",
    "in", "inc", "info", "ing", "ink", "insure", "int", "international", "io", "iq", "ir",
    "is", "ist", "istanbul", "it", "je", "jetzt", "jo", "jobs", "jp", "ke", "kg", "kh", "ki",
    "kitchen", "koeln", "kr", "kw", "kz", "la", "land", "lat", "law", "lb", "leclerc", "legal",
    "li", "life", "limited", "limo", "link", "live", "living", "lk", "llc", "locker", "lol",
    "london", "love", "ls", "lt", "ltd", "lu", "luxe", "lv", "ly", "ma", "madrid", "makeup",
    "management", "market", "marketing", "markets", "md", "me", "med", "media", "meme", "men",
    "menu", "mg", "microsoft", "mil", "mk", "ml", "mn", "mo", "mobi", "moe", "mom", "money",
    "monster", "motorcycles", "mov", "movie", "mp", "ms", "mt", "mu", "museum", "mv", "mw",
    "mx", "my", "mz", "na", "name", "navy", "nc", "ne", "nec", "net", "network", "new", "news",
    "nf", "ng", "nhk", "ni", "ninja", "nl", "no", "now", "nr", "nrw", "nu", "nz", "om", "one",
    "onl", "online", "ooo", "org", "ovh", "pa", "page", "paris", "partners", "parts", "party",
    "pe", "pf", "ph", "phd", "photo", "photos", "pics", "pictures", "pink", "pk", "pl",
    "place", "plus", "pm", "pn", "porn", "pr", "press", "pro", "promo", "ps", "pt", "pub",
    "pw", "py", "qa", "qpon", "quest", "re", "realestate", "realtor", "red", "reisen", "rent",
    "report", "rest", "rip", "ro", "rocks", "rodeo", "rs", "ru", "run", "rw", "sa", "sale",
    "sap", "sarl", "saxo", "sb", "sbs", "sc", "school", "science", "scot", "se", "security",
    "services", "sex", "sg", "sh", "sharp", "shop", "shopping", "show", "si", "site", "sk",
    "ski", "skin", "sm", "sn", "so", "soccer", "social", "software", "solutions", "soy",
    "space", "sport", "sr", "ss", "st", "statefarm", "store", "stream", "studio", "study",
    "su", "support", "surf", "sv", "sx", "sy", "systems", "tax", "taxi", "tc", "team", "tech",
    "technology", "tel", "tg", "th", "tips", "tk", "tl", "tm", "tn", "to", "today", "tokyo",
    "tools", "top", "town", "toyota", "tr", "trade", "travel", "tt", "tube", "tv", "tw", "tz",
    "ua", "ug", "uk", "uno", "uol", "us", "uy", "uz", "va", "vc", "ve", "ventures", "vet",
    "vg", "vi", "video", "vin", "vip", "vn", "vote", "vu", "wales", "watch", "website", "wf",
    "wiki", "win", "work", "works", "world", "ws", "wtf", "xin", "xn--p1ai", "xxx", "xyz",
    "yandex", "ye", "you", "youtube", "zip", "zm", "zone", "zw"
])

SUSPICIOUS_EXTENSIONS = frozenset(["html", "htm", "php", "php5", "jsp", "js", "docx", "bat", "pdf"])

_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_TOKEN_SPLIT_RE = re.compile(r"[-/._?=&:#@]+")
_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)$")


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url
    return url


def extract_features(url: str) -> dict:
    """Mengembalikan dict fitur (sudah termasuk has_http; urutan akhir diatur oleh pemanggil)."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    try:
        host = (parsed.hostname or "").lower()
    except ValueError:
        host = ""
    path = parsed.path or ""
    query = parsed.query or ""

    host_parts = [p for p in host.split(".") if p != ""]

    urllength = len(url)
    digits = sum(ch.isdigit() and ch in "0123456789" for ch in url)

    freq = Counter(url)
    n = len(url)
    urlentropy = -sum((c / n) * math.log2(c / n) for c in freq.values()) if n else 0.0

    tld = host_parts[-1] if host_parts else ""
    ext_m = _EXT_RE.search(path)
    ext = ext_m.group(1).lower() if ext_m else ""

    https = int(scheme == "https")
    return {
        "urllength": urllength,
        "hasipaddress": int(bool(_IP_RE.match(host))),
        "dotcount": url.count("."),
        "httpsflag": https,
        "urlentropy": urlentropy,
        "tokencount": sum(1 for t in _TOKEN_SPLIT_RE.split(url) if t != ""),
        "subdomaincount": max(0, len(host_parts) - 2),
        "queryparamcount": sum(1 for q in query.split("&") if q != "") if query else 0,
        "tldlength": len(tld),
        "pathlength": len(path),
        "hash": int("#" in url),
        "hyphenindomain": int("-" in host),
        "numberofdigits": digits,
        "tldpopularity": int(tld.lower() in POPULAR_TLDS),
        "suspiciousfileextension": int(ext in SUSPICIOUS_EXTENSIONS),
        "domainnamelength": len(host_parts[-2]) if len(host_parts) >= 2 else len(host),
        "percentagenumericchars": (digits / n * 100) if n else 0.0,
        "has_http": 1 - https,
        "count_dashes": url.count("-"),
        "count_underscores": url.count("_"),
        "count_slashes": url.count("/"),
        "count_ques": url.count("?"),
        "count_non_alphanumeric": len(re.findall(r"[^A-Za-z0-9]", url)),
        "count_letters": len(re.findall(r"[A-Za-z]", url)),
    }


def feature_vector(url: str, feature_order):
    feats = extract_features(url)
    return [float(feats[f]) for f in feature_order]
