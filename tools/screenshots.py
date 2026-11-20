"""Screenshots for the report: Folium maps, the Streamlit UI and the API console log.

    python tools/screenshots.py
Starts uvicorn (port 8000) and streamlit (port 8501) temporarily, calls the API for every student-test
trip (so the console log contains real requests), and screenshots with headless Edge.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "report" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
API_PY = ROOT / "envs" / "api" / "Scripts" / "python.exe"
UI_PY = ROOT / "envs" / "ui" / "Scripts" / "python.exe"


def shot(url, out, w=1400, h=900, budget=10000, realtime_ms=None):
    """budget = virtual time for static pages; realtime_ms = real wait for websocket apps (Streamlit)."""
    Path(out).unlink(missing_ok=True)
    prof = ROOT / "report" / ".edge_profile" / Path(out).stem  # fresh profile per shot
    wait_flag = f"--timeout={realtime_ms}" if realtime_ms else f"--virtual-time-budget={budget}"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                    f"--window-size={w},{h}",
                    wait_flag, f"--screenshot={out}", url], check=True, timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # msedge.exe returns before the headless browser has written the file: wait for it
    out = Path(out)
    t0, last = time.time(), -1
    while time.time() - t0 < 90:
        if out.exists() and out.stat().st_size == last and last > 0:
            break
        last = out.stat().st_size if out.exists() else -1
        time.sleep(1.5)
    print("shot ->", out, out.stat().st_size if out.exists() else "MISSING")


def wait(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"{url} not up")


def main():
    for html in sorted((ROOT / "results" / "maps").glob("*.html")):
        name = "map_eda_00000054.png" if html.name.startswith("eda_") else f"{html.stem}.png"
        shot(html.as_uri(), SHOTS / name)

    log_path = ROOT / "results" / "logs" / "api_demo.log"
    log = open(log_path, "w", encoding="utf-8")
    api = subprocess.Popen([str(API_PY), "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ui = None
    try:
        wait("http://127.0.0.1:8000/health")
        import json
        trips = json.load(urllib.request.urlopen("http://127.0.0.1:8000/trips"))["trips"]
        for t in trips:
            req = urllib.request.Request("http://127.0.0.1:8000/predict", data=json.dumps({"trip_id": t}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=600).read()
        ui = subprocess.Popen([str(UI_PY), "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501",
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait("http://127.0.0.1:8501")
        time.sleep(3)
        shot("http://127.0.0.1:8501/?trip=00000076", SHOTS / "ui_streamlit.png", 1400, 1500, realtime_ms=25000)
    finally:
        if ui:
            ui.terminate()
        api.terminate()
        log.close()

    lines = [ln for ln in open(log_path, encoding="utf-8").read().splitlines() if ln.strip()][-26:]
    fig = plt.figure(figsize=(13, 0.26 * len(lines) + 0.6))
    fig.patch.set_facecolor("#0c0c0c")
    fig.text(0.01, 0.98, "Laptop 1 - uvicorn api.main:app --host 0.0.0.0 --port 8000", color="#9cdcfe",
             family="monospace", fontsize=9, va="top")
    for i, ln in enumerate(lines):
        fig.text(0.01, 0.98 - (i + 1.5) / (len(lines) + 2), ln[:170], color="#e0e0e0", family="monospace",
                 fontsize=7.5, va="top")
    fig.savefig(SHOTS / "api_console.png", dpi=130, facecolor=fig.get_facecolor())
    print("api console ->", SHOTS / "api_console.png")


if __name__ == "__main__":
    sys.exit(main())
