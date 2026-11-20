"""PathSnap evaluator UI (Laptop 2).

    streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
"""
import os

import folium
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map matching & route reconstruction")
st.caption(f"API: {API_URL}")

try:
    trips = requests.get(f"{API_URL}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach API at {API_URL}: {e}")
    st.stop()

qp_trip = st.query_params.get("trip")  # optional deep link: ?trip=00000076 runs immediately
trip_id = st.selectbox("Evaluation trip", trips, index=trips.index(qp_trip) if qp_trip in trips else 0)
auto = qp_trip in trips and "result" not in st.session_state
if st.button("Run map matching", type="primary") or auto:
    with st.spinner("Calling API..."):
        r = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(r.text)
    else:
        st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']}  |  {res['n_gps_points']} GPS points  |  {res['processing_seconds']} s")
    if res["metrics"]:
        st.dataframe(pd.DataFrame(res["metrics"]).T.round(4))
    gps = res["gps_points"]
    m = folium.Map(location=gps[len(gps) // 2], zoom_start=15)
    fg_gps = folium.FeatureGroup(name="Raw GPS (blue)")
    for p in gps[::2]:
        folium.CircleMarker(p, radius=2, color="blue", fill=True).add_to(fg_gps)
    fg_gps.add_to(m)
    fg_pred = folium.FeatureGroup(name="Predicted route (red)")
    folium.PolyLine(res["predicted_route_latlon"], color="red", weight=4).add_to(fg_pred)
    fg_pred.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st_folium(m, height=600, use_container_width=True)
