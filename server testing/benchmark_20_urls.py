"""
Sequential benchmark untuk aplikasi Shiny di Posit Cloud.
Mengambil timing server-side dari atribut data-* yang dikirim app_benchmark.py.
"""

import time
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_URL = "https://01a18820-2551-2ec0-1a87-dd2045e6a2df.share.connect.posit.cloud/"

URLS = [
"https://www.metacareers.com/profile/job_details/848358355026771/",
"https://edacafe.techjobscafe.com/forgot_my_password.php",
"https://www.usajobs.gov/Applicant/Dashboard/savedsearches",
"http://relogiodepontohenry.com.br/wp-includes/pomo/entry/linkedin.com/linkedin.com/login.php",
"https://www.naukri.com/220kv-jobs-in-mumbai",
"https://www.loker.id/lokasi-pekerjaan/yogyakarta",
"http://skipper-spb.ru/hb/linkedin_/login.php?s=mail&redirect=http://mail3.nate.com/&email=3mail@b.c",
"https://www.monster.com/7647840/2326549.html",
"https://marketing-group.ir/linkedin/linkedin.com/linkedin.com/login.php?ejduh=%7b%7bemail%7d%7d",
"https://www.simplyhired.com/?s=f7c4ec8ae91b9ff965861adad8e906a9&u=25",
"http://jpasportsmedia.com/linkedin.com/account/sign-in/linkedin",
"https://www.naukri.com/3c-jobs-in-oman",
"http://guardable-workload.000webhostapp.com/linkedinnew/linkedin_/login.php?email=[[-email-",
"http://idisraelita.org/1234/linkedin2.0.6/cmd-login=d3993414284beca7c9d20faab65c690f/root.php?email=",
"https://id.indeed.com/BON-APPETIT-jobs-in-Jakarta",
"http://dpu.edu.krd/public/filemanager_store/dsdi/new/css2/linkedin/linknewin/linkedin",
"https://www.urbanhire.com/en/jobs/business-administration-partner-relation-dollak-nusa-indonesia-jmn",
"https://www.indeed.com/?vjk=359d7866e1b7c770&from=smart-apply&advn=6186628037999356",
"https://www.topkarir.com/article/detail/cara-agar-tetap-produktif-saat-bekerja-di-rumah",
"https://career-managements.com/wp-content/upgrade/hasny/Home/Billing.php"
]

TIMEOUT_MS = 60000

def read_timing(page):
    locator = page.locator("#result_ui span[data-t]")
    return {
        "stamp": locator.get_attribute("data-t"),
        "feature_extraction_ms": locator.get_attribute("data-feature-ms"),
        "prediction_ms": locator.get_attribute("data-prediction-ms"),
        "clustering_ms": locator.get_attribute("data-clustering-ms"),
        "total_ms": locator.get_attribute("data-total-ms"),
        "label": locator.get_attribute("data-label"),
        "cluster_error": locator.get_attribute("data-cluster-error"),
    }

rows = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    print("Membuka aplikasi Posit Cloud...")
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120000)

    # Pastikan UI Shiny sudah siap menerima input.
    page.locator("#url_input").wait_for(state="visible", timeout=120000)
    page.locator("#scan_btn").wait_for(state="visible", timeout=120000)

    # Satu URL diproses setelah URL sebelumnya selesai.
    for i, url in enumerate(URLS, start=1):
        print(f"[{i:02d}/{len(URLS)}] {url}")

        stamp_locator = page.locator("#result_ui span[data-t]")
        old_stamp = None
        if stamp_locator.count() > 0:
            old_stamp = stamp_locator.get_attribute("data-t")

        page.locator("#url_input").fill(url)
        page.locator("#scan_btn").click()

        try:
            page.wait_for_function(
                """oldStamp => {
                    const el = document.querySelector('#result_ui span[data-t]');
                    return el && el.getAttribute('data-t') !== oldStamp;
                }""",
                arg=old_stamp,
                timeout=TIMEOUT_MS,
            )
            timing = read_timing(page)

            rows.append({
                "URL": url,
                "Prediction": timing["label"],
                "Feature Extraction (ms)": float(timing["feature_extraction_ms"]),
                "Prediction Time (ms)": float(timing["prediction_ms"]),
                "Clustering (ms)": (
                    float(timing["clustering_ms"])
                    if timing["clustering_ms"] not in (None, "")
                    else np.nan
                ),
                "Total (ms)": float(timing["total_ms"]),
                "Clustering Error": timing["cluster_error"],
                "Status": "Success",
            })
        except (PlaywrightTimeoutError, TypeError, ValueError) as exc:
            rows.append({
                "URL": url,
                "Prediction": None,
                "Feature Extraction (ms)": np.nan,
                "Prediction Time (ms)": np.nan,
                "Clustering (ms)": np.nan,
                "Total (ms)": np.nan,
                "Clustering Error": str(exc),
                "Status": "Failed",
            })
            print(f"  Gagal mengambil hasil timing: {exc}")

        # Jeda kecil agar UI selesai memperbarui sebelum request berikutnya.
        time.sleep(0.25)

    browser.close()

raw = pd.DataFrame(rows)
raw.to_csv("server_processing_20_urls_raw.csv", index=False)

# Feature extraction, prediction, dan total dihitung dari semua scan sukses.
# Clustering hanya dihitung pada URL yang benar-benar menjalankan clustering.
summary_rows = []
for stage, column in [
    ("Feature extraction", "Feature Extraction (ms)"),
    ("Prediction", "Prediction Time (ms)"),
    ("Clustering", "Clustering (ms)"),
    ("Total", "Total (ms)"),
]:
    values = pd.to_numeric(raw[column], errors="coerce").dropna()
    if len(values) == 0:
        summary_rows.append({
            "Stage": stage, "Mean": np.nan, "Median": np.nan,
            "95th percentile": np.nan, "Max": np.nan
        })
    else:
        summary_rows.append({
            "Stage": stage,
            "Mean": values.mean(),
            "Median": values.median(),
            "95th percentile": values.quantile(0.95),
            "Max": values.max(),
        })

summary = pd.DataFrame(summary_rows)
numeric_cols = ["Mean", "Median", "95th percentile", "Max"]
summary[numeric_cols] = summary[numeric_cols].round(3)

summary.to_csv("server_processing_20_urls_summary.csv", index=False)

print("\nTABEL SERVER-SIDE PROCESSING TIME (ms)")
print(summary.to_string(index=False))
print("\nFile tersimpan:")
print("- server_processing_20_urls_raw.csv")
print("- server_processing_20_urls_summary.csv")
cluster_count = raw["Clustering (ms)"].notna().sum()
print(f"Jumlah URL yang benar-benar menjalankan clustering: {cluster_count}")
print("Catatan: jumlah ini dapat kurang dari 20 karena app hanya menjalankan")
print("clustering untuk URL yang diprediksi phishing.")
