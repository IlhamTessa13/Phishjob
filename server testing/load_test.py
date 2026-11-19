
import asyncio
import csv
import time
from datetime import datetime

from playwright.async_api import async_playwright

# URL aplikasi publik
APP_URL = (
    "https://01a18820-2551-2ec0-1a87-dd2045e6a2df"
    ".share.connect.posit.cloud/"
)

# Gunakan URL phishing yang sebelumnya berhasil diuji
TEST_URL = (
    "http://secure-careers-apply.verify-account.xyz/"
    "job-offer.php?id=123456"
)

# Ubah menjadi 1, 2, atau 5 untuk setiap skenario
CONCURRENT_USERS = 5

async def simulate_user(browser, user_id):
    context = await browser.new_context()
    page = await context.new_page()

    try:
        # Membuka aplikasi
        await page.goto(
            APP_URL,
            wait_until="domcontentloaded",
            timeout=120000
        )

        # Menunggu input URL siap
        await page.locator("#url_input").wait_for(
            state="visible",
            timeout=120000
        )

        # Memasukkan URL
        await page.locator("#url_input").fill(TEST_URL)

        # Mulai pengukuran saat tombol Scan ditekan
        start_time = time.perf_counter()

        await page.locator("#scan_btn").click()

        # Menunggu hasil prediksi muncul
        result_box = page.locator("#result_ui")

        await result_box.get_by_text(
            "URL:",
            exact=False
        ).wait_for(timeout=120000)

        elapsed = time.perf_counter() - start_time

        result_text = await result_box.inner_text()

        if "Legitimate" in result_text:
            status = "Legitimate"
        elif "Phishing" in result_text:
            status = "Phishing"
        else:
            status = "Unknown"

        return {
            "user": user_id,
            "status": status,
            "response_time": round(elapsed, 3),
            "success": True,
            "error": ""
        }

    except Exception as e:
        return {
            "user": user_id,
            "status": "Failed",
            "response_time": None,
            "success": False,
            "error": str(e)[:250]
        }

    finally:
        await context.close()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        print(f"\nConcurrent users: {CONCURRENT_USERS}")
        print("Starting load test...\n")

        # Semua pengguna memulai pemindaian secara bersamaan
        results = await asyncio.gather(
            *[
                simulate_user(browser, i + 1)
                for i in range(CONCURRENT_USERS)
            ]
        )

        await browser.close()

    # Ringkasan hasil
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("===== TEST RESULTS =====")

    for r in results:
        print(
            f"User {r['user']}: "
            f"Status={r['status']}, "
            f"Time={r['response_time']}, "
            f"Success={r['success']}"
        )

    print("\n===== SUMMARY =====")
    print(f"Total users: {CONCURRENT_USERS}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_time = sum(
            r["response_time"] for r in successful
        ) / len(successful)

        print(f"Average response time: {avg_time:.3f} seconds")

    # Simpan hasil ke CSV
    filename = (
        f"results_{CONCURRENT_USERS}_users_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user",
                "status",
                "response_time",
                "success",
                "error"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())

