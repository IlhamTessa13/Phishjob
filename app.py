"""
PhishJob Detection — Shiny for Python
Tahap 1: LightGBM (phishing vs legitimate)
Tahap 2: jika phishing -> klasterisasi (Hybrid Self-Attention AE + K-Means, k=2): Phishing Job vs Phishing General
Jalankan:  shiny run --reload app.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from shiny import App, reactive, render, ui

from phishjob_features import feature_vector, normalize_url

BASE = Path(__file__).parent

# ====== Tahap 1: model LightGBM ======
booster = lgb.Booster(model_file=str(BASE / "phishjob_lgbm.txt"))
meta = json.loads((BASE / "phishjob_lgbm_meta.json").read_text(encoding="utf-8"))
FEATURE_ORDER = meta["features"]
THRESHOLD = float(meta["threshold"])  # threshold OOF-optimal dari notebook

# ====== Tahap 2: klasterisasi (opsional; aplikasi tetap jalan tanpa folder stage2_export) ======
stage2 = None
try:
    from phishjob_stage2 import Stage2Clusterer

    stage2 = Stage2Clusterer(BASE / "stage2_export")
    if stage2.features != FEATURE_ORDER:
        print("[WARN] Urutan fitur Tahap 2 berbeda dengan meta LightGBM; Tahap 2 memakai urutannya sendiri.")
    print("[OK] Model klasterisasi Tahap 2 dimuat.")
except Exception as e:  # noqa: BLE001
    print(f"[WARN] Tahap 2 tidak aktif: {e!r}")


def predict_url(raw_url: str) -> dict:
    url = normalize_url(raw_url)
    x = np.array([feature_vector(url, FEATURE_ORDER)], dtype=np.float64)
    p = float(booster.predict(x)[0])  # P(ClassLabel = 1 = phishing)

    if p >= THRESHOLD:
        res = {"url": url, "label": "Phishing", "confidence": p, "cluster": None}
        if stage2 is not None:
            try:
                res["cluster"] = stage2.predict(url)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] klasterisasi gagal: {e!r}")
        return res
    return {"url": url, "label": "Legitimate", "confidence": 1 - p, "cluster": None}


# ====== Helper: ikon SVG (gaya lucide) ======
def icon(inner: str, cls: str = "h-5 w-5"):
    return ui.HTML(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="{cls}">{inner}</svg>'
    )


ICON_SHIELD_CHECK = '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>'
ICON_ZAP = '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>'
ICON_LOCK = '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
ICON_TARGET = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
ICON_GRAD = '<path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/><path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>'
ICON_CODE = '<path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/>'
ICON_SEARCH = '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
ICON_BRIEFCASE = '<path d="M16 20V4a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/><rect width="20" height="14" x="2" y="6" rx="2"/>'
ICON_GLOBE = '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>'

T = ui.tags

HEAD_JS = """
tailwind.config = {
  theme: { extend: {
    colors: { primary: '#0b5fff', muted: '#64748b' },
    fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] }
  } }
};
"""

PAGE_JS = """
var scanTimeout = null;

function scrollToSection(id){ document.getElementById(id).scrollIntoView({behavior:'smooth'}); }

function doScan(){
  var v = document.getElementById('url_input').value.trim();
  var err = document.getElementById('url_error');
  if(!v){ err.classList.remove('hidden'); return; }
  err.classList.add('hidden');
  document.getElementById('scanning').classList.remove('hidden');
  document.getElementById('result_ui').classList.add('opacity-30');
  var btn = document.getElementById('scan_btn');
  btn.disabled = true; btn.classList.add('opacity-60','cursor-not-allowed');
  // safety net: restore UI if no result arrives within 20 seconds
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(scanFinished, 20000);
  Shiny.setInputValue('scan', {url: v, t: Date.now()}, {priority: 'event'});
}

function scanFinished(){
  clearTimeout(scanTimeout);
  document.getElementById('scanning').classList.add('hidden');
  document.getElementById('result_ui').classList.remove('opacity-30');
  var btn = document.getElementById('scan_btn');
  btn.disabled = false; btn.classList.remove('opacity-60','cursor-not-allowed');
}

function quickTest(url){
  document.getElementById('url_input').value = url;
  doScan();
}

document.addEventListener('keydown', function(e){
  if(e.key === 'Enter' && document.activeElement && document.activeElement.id === 'url_input'){ doScan(); }
});

// Attach after all scripts (jQuery & Shiny) are fully loaded
window.addEventListener('load', function(){
  $(document).on('shiny:value shiny:error', function(e){
    if(e.name === 'result_ui'){ scanFinished(); }
  });
});
"""

# ====== UI (Tailwind CDN) ======
app_ui = ui.TagList(
    ui.head_content(
        T.title("PhishJob Detection — Phishing Detector for Job Posting Links"),
        T.meta(name="viewport", content="width=device-width, initial-scale=1"),
        T.link(rel="preconnect", href="https://fonts.googleapis.com"),
        T.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"),
        T.script(src="https://cdn.tailwindcss.com"),
        T.script(ui.HTML(HEAD_JS)),
        T.style(ui.HTML("html { scroll-behavior: smooth; } body { background:#f8fafc; }")),
        T.script(ui.HTML(PAGE_JS)),
    ),
    T.div(
        # ---------- Navbar ----------
        T.header(
            T.div(
                T.div(
                    T.div(icon(ICON_SHIELD_CHECK, "h-5 w-5"),
                          class_="h-10 w-10 rounded-xl bg-primary text-white flex items-center justify-center"),
                    ui.HTML('<span class="text-xl font-bold">PhishJob <span class="text-primary">Detection</span></span>'),
                    class_="flex items-center gap-2",
                ),
                T.nav(
                    T.a("Home", href="#", class_="hover:text-primary", onclick="scrollToSection('home');return false;"),
                    T.a("Detection", href="#", class_="hover:text-primary", onclick="scrollToSection('detection');return false;"),
                    T.a("About", href="#", class_="hover:text-primary", onclick="scrollToSection('about');return false;"),
                    class_="hidden sm:flex items-center gap-8 text-sm font-medium text-slate-600",
                ),
                T.span(
                    T.span(class_="h-2 w-2 rounded-full bg-emerald-500"), "ML Model Ready",
                    class_="inline-flex items-center gap-2 text-xs font-medium text-emerald-700 bg-emerald-50 rounded-full px-3 py-1.5",
                ),
                class_="max-w-[1520px] mx-auto px-4 h-16 flex items-center justify-between",
            ),
            class_="sticky top-0 z-50 bg-white/90 backdrop-blur border-b",
        ),
        # ---------- Home ----------
        T.section(
            T.div(
                T.div(icon(ICON_SHIELD_CHECK, "h-8 w-8"),
                      class_="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-6 text-primary"),
                T.h1("Detect Job Posting Phishing with ", T.span("Machine Learning", class_="text-primary"),
                     class_="text-4xl sm:text-5xl font-bold tracking-tight text-slate-900"),
                T.p("PhishJob Detection analyzes job posting links in real time using a LightGBM model to spot phishing "
                    "and fake recruitment URLs, then clusters detected phishing into job-themed and general phishing.",
                    class_="text-lg text-muted mt-4"),
                T.button("Start Scanning Now", onclick="scrollToSection('detection')",
                         class_="mt-8 h-12 px-8 text-base font-medium rounded-lg bg-primary text-white hover:bg-primary/90 transition"),
                T.div(
                    T.div(T.span(icon(ICON_ZAP), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Real-time", class_="font-medium text-sm"),
                                T.p("Analysis results within seconds", class_="text-xs text-muted mt-0.5")),
                          class_="flex items-start gap-3 p-4 rounded-xl border bg-white"),
                    T.div(T.span(icon(ICON_SHIELD_CHECK), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Two-stage", class_="font-medium text-sm"),
                                T.p("LightGBM detection + phishing clustering", class_="text-xs text-muted mt-0.5")),
                          class_="flex items-start gap-3 p-4 rounded-xl border bg-white"),
                    T.div(T.span(icon(ICON_LOCK), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Free & Private", class_="font-medium text-sm"),
                                T.p("URLs are not stored after scanning", class_="text-xs text-muted mt-0.5")),
                          class_="flex items-start gap-3 p-4 rounded-xl border bg-white"),
                    class_="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-14 text-left",
                ),
                class_="max-w-2xl w-full text-center",
            ),
            id="home", class_="min-h-[85vh] flex items-center justify-center px-4 py-20",
        ),
        # ---------- Detection ----------
        T.section(
            T.div(
                T.div(T.h2("Detection", class_="text-2xl font-bold"),
                      T.p("Enter a job posting URL to be analyzed by the model", class_="text-sm text-muted mt-1"),
                      class_="text-center mb-6"),
                T.div(
                    T.div(T.h3("Job Link Phishing Scanner", class_="font-semibold"),
                          T.p("Enter a job posting URL below to analyze", class_="text-sm text-muted"),
                          class_="mb-4"),
                    T.div(
                        T.div(
                            T.span(icon(ICON_SEARCH, "h-4 w-4"),
                                   class_="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"),
                            T.input(id="url_input", type="text", placeholder="www.linkedin.com/jobs/view/123456",
                                    autocomplete="off",
                                    class_="w-full h-12 pl-10 pr-3 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"),
                            class_="relative flex-1",
                        ),
                        T.button(icon(ICON_SEARCH, "h-4 w-4"), "Scan URL", id="scan_btn", onclick="doScan()",
                                 class_="h-12 px-6 rounded-lg bg-primary text-white font-medium inline-flex items-center justify-center gap-2 hover:bg-primary/90 transition"),
                        class_="flex flex-col sm:flex-row gap-3",
                    ),
                    T.p("Please enter a URL first.", id="url_error", class_="hidden text-xs text-red-600 mt-2"),
                    T.div(
                        T.span("Quick test examples:", class_="text-muted"),
                        T.button("Phishing", class_="px-3 py-1.5 rounded-full bg-red-50 text-red-600 font-medium hover:bg-red-100",
                                 onclick="quickTest('http://secure-careers-apply.verify-account.xyz/job-offer.php?id=123456')"),
                        T.button("Safe", class_="px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 font-medium hover:bg-emerald-100",
                                 onclick="quickTest('https://www.linkedin.com/jobs/')"),
                        T.button("Suspicious", class_="px-3 py-1.5 rounded-full bg-amber-50 text-amber-600 font-medium hover:bg-amber-100",
                                 onclick="quickTest('http://192.168.1.1/apply-now.html')"),
                        T.button("Short URL", class_="px-3 py-1.5 rounded-full bg-blue-50 text-primary font-medium hover:bg-blue-100",
                                 onclick="quickTest('https://bit.ly/3xYz')"),
                        class_="flex flex-wrap items-center gap-2 mt-4 text-xs",
                    ),
                    T.div(
                        ui.HTML('<svg class="animate-spin h-5 w-5 text-primary" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path></svg>'),
                        "Analyzing URL...",
                        id="scanning", class_="hidden mt-6 flex items-center justify-center gap-3 py-6 text-sm text-muted",
                    ),
                    ui.output_ui("result_ui"),
                    class_="bg-white rounded-2xl border shadow-sm p-6",
                ),
                T.p("Powered by LightGBM + Hybrid Self-Attention Autoencoder", class_="text-center text-xs text-muted mt-6"),
                class_="w-full max-w-xl",
            ),
            id="detection", class_="min-h-[70vh] flex items-center justify-center px-4 py-20",
        ),
        # ---------- About ----------
        T.section(
            T.div(
                T.div(T.h2("About Us", class_="text-2xl font-bold"),
                      T.p("About the PhishJob Detection project", class_="text-sm text-muted mt-1"),
                      class_="text-center mb-8"),
                T.div(
                    T.div(T.span(icon(ICON_TARGET), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Project Objective", class_="font-medium text-sm"),
                                T.p("PhishJob Detection automatically detects phishing links disguised as job postings using a "
                                    "LightGBM model, then groups detected phishing into job-themed and general phishing clusters.",
                                    class_="text-sm text-muted mt-1")),
                          class_="flex items-start gap-3"),
                    T.div(T.span(icon(ICON_GRAD), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Tech Stack", class_="font-medium text-sm"),
                                T.p("Frontend: Shiny for Python + Tailwind CSS · Stage 1: LightGBM · "
                                    "Stage 2: Self-Attention Autoencoder + K-Means",
                                    class_="text-sm text-muted mt-1")),
                          class_="flex items-start gap-3"),
                    T.div(T.span(icon(ICON_CODE), class_="text-primary mt-0.5 shrink-0"),
                          T.div(T.p("Source Code", class_="font-medium text-sm"),
                                T.p("https://github.com/IlhamTessa13/phishing-detector", class_="text-sm text-muted mt-1")),
                          class_="flex items-start gap-3"),
                    class_="bg-white rounded-2xl border shadow-sm p-6 space-y-4",
                ),
                class_="w-full max-w-2xl",
            ),
            id="about", class_="min-h-[60vh] flex items-center justify-center px-4 py-20",
        ),
        class_="font-sans text-slate-900 antialiased",
    ),
)


# ====== Result card: blok jenis phishing (Tahap 2) ======
def cluster_block(cluster: dict):
    is_job = cluster["cluster"] == "job"
    pct = round(cluster["affinity"] * 100, 1)
    if is_job:
        wrap, ico_cls, badge = "border-amber-300 bg-amber-50", "text-amber-700", "bg-amber-100 text-amber-800"
        bar, ico, title = "bg-amber-500", ICON_BRIEFCASE, "Phishing Job"
        desc = ("This link falls in the job / recruitment phishing cluster — it resembles fake job offers. "
                "Never submit your CV, ID, or fees to it.")
    else:
        wrap, ico_cls, badge = "border-slate-300 bg-white/70", "text-slate-700", "bg-slate-200 text-slate-800"
        bar, ico, title = "bg-slate-500", ICON_GLOBE, "Phishing General"
        desc = "This link falls in the general (non-job) phishing cluster — it is not specifically themed around recruitment."
    return T.div(
        T.div(
            T.span(icon(ico, "h-5 w-5"), class_=ico_cls),
            T.div(
                T.div(T.span("Phishing type", class_="text-xs text-slate-500"),
                      T.span(title, class_=f"ml-2 text-xs font-semibold rounded-full px-2.5 py-0.5 {badge}")),
                T.p(desc, class_="text-xs text-slate-600 mt-1"),
            ),
            class_="flex items-start gap-3",
        ),
        T.div(
            T.div(T.span("Cluster affinity"), T.span(f"{pct}%", class_="font-semibold"),
                  class_="flex justify-between text-xs text-slate-600 mb-1"),
            T.div(T.div(class_=f"h-full rounded-full {bar}", style=f"width:{pct}%"),
                  class_="h-1.5 w-full rounded-full bg-white overflow-hidden"),
            class_="mt-3",
        ),
        T.p("Exploratory unsupervised clustering: shows similarity to a cluster, not a verified label.",
            class_="mt-2 text-[11px] text-slate-400"),
        class_=f"mt-4 rounded-lg border p-3 {wrap}",
    )


# ====== Server ======
def server(input, output, session):
    @reactive.calc
    @reactive.event(input.scan)
    def result():
        payload = input.scan()
        url = (payload or {}).get("url", "")
        if not url or not url.strip():
            return {"error": "empty"}
        try:
            return predict_url(url)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    @render.ui
    def result_ui():
        res = result()
        # penanda unik per scan agar scan ulang URL yang sama tetap dianggap hasil baru
        stamp = T.span(class_="hidden", **{"data-t": str((input.scan() or {}).get("t", ""))})

        if "error" in res:
            return ui.TagList(stamp, T.div(
                T.p("Failed to analyze the URL", class_="font-semibold"),
                T.p("Please make sure the URL format is valid and try again.", class_="mt-1 text-xs"),
                class_="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800",
            ))

        is_phish = res["label"] == "Phishing"
        pct = round(res["confidence"] * 100, 2)
        box = "border-red-200 bg-red-50" if is_phish else "border-emerald-200 bg-emerald-50"
        title = "text-red-700" if is_phish else "text-emerald-700"
        bar = "bg-red-500" if is_phish else "bg-emerald-500"
        msg = ("This job link looks like phishing. Do not enter personal data or credentials."
               if is_phish else "This job link appears to be safe based on the model's analysis.")

        return ui.TagList(stamp, T.div(
            T.div(
                T.span(icon(ICON_SHIELD_CHECK, "h-7 w-7"), class_=title),
                T.div(T.p("Phishing" if is_phish else "Legitimate", class_=f"text-lg font-bold {title}"),
                      T.p(msg, class_="text-xs text-slate-600")),
                class_="flex items-center gap-3",
            ),
            T.div(
                T.div(T.span("Confidence"), T.span(f"{pct}%", class_="font-semibold"),
                      class_="flex justify-between text-xs text-slate-600 mb-1"),
                T.div(T.div(class_=f"h-full rounded-full {bar}", style=f"width:{pct}%"),
                      class_="h-2 w-full rounded-full bg-white/70 overflow-hidden"),
                class_="mt-4",
            ),
            cluster_block(res["cluster"]) if (is_phish and res.get("cluster")) else None,
            T.p(T.span("URL: ", class_="font-medium"), res["url"], class_="mt-4 text-xs text-slate-500 break-all"),
            class_=f"mt-6 rounded-xl border p-5 {box}",
        ))


app = App(app_ui, server)
