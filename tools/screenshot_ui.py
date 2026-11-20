"""Screenshot of the Streamlit UI after a real /predict round-trip (needs: pip install playwright; uses installed Edge).

    python tools/screenshot_ui.py [trip_id]
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def wait(url):
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(url)


def main(trip="00000076"):
    log = open(ROOT / "results" / "logs" / "api_demo_ui.log", "w")
    api = subprocess.Popen([str(ROOT / "envs/api/Scripts/python.exe"), "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0",
                            "--port", "8000"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ui = subprocess.Popen([str(ROOT / "envs/ui/Scripts/python.exe"), "-m", "streamlit", "run", "ui/app.py", "--server.port",
                           "8501", "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait("http://127.0.0.1:8000/health")
        wait("http://127.0.0.1:8501")
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(viewport={"width": 1400, "height": 1000})
            pg.goto(f"http://127.0.0.1:8501/?trip={trip}")
            pg.get_by_text("GPS points").first.wait_for(timeout=120000)
            pg.wait_for_timeout(8000)  # map tiles
            pg.screenshot(path=str(ROOT / "report" / "screenshots" / "ui_streamlit.png"), full_page=True)
            b.close()
    finally:
        ui.terminate()
        api.terminate()
        log.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
