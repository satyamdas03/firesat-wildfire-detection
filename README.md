<div align="center">

# FireSat — Real-Time Wildfire Detection via Edge AI on CubeSats

**Edge AI (MobileNetV3) · NASA FIRMS Live Data · CubeSat Simulation · Real-Time Global Dashboard**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![NASA FIRMS](https://img.shields.io/badge/Data-NASA%20FIRMS%20Live-blue?logo=nasa&logoColor=white)](https://firms.modaps.eosdas.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What Is FireSat?

FireSat is a full-stack simulation of a **real-time wildfire detection system powered by Edge AI running aboard a CubeSat in orbit**.

It demonstrates the complete pipeline from satellite to screen:

```
CubeSat Orbit  →  Edge AI Inference  →  Ground Station  →  Live Global Dashboard
(MobileNetV3)     (fire/no-fire)        (Express + WS)      (Leaflet.js map)
```

The system uses **real NASA FIRMS fire coordinates** from the last 24 hours, runs a fine-tuned **MobileNetV3-Small** model (designed for low-power CPUs — exactly what a CubeSat carries), and streams detection events to a real-time browser dashboard via **WebSocket**.

---

## Why CubeSats + Edge AI?

Traditional wildfire monitoring relies on ground sensors and periodic satellite passes with data beamed to high-powered ground stations for analysis. The problem: **latency**. By the time the image reaches the ground, the window for early intervention is often gone.

The Edge AI approach runs inference **onboard** — the satellite detects the fire and transmits only the alert (a tiny JSON packet), not raw imagery. This:
- Reduces downlink bandwidth by ~99%
- Cuts detection-to-alert latency from hours to seconds
- Works on CubeSat-class hardware (ARM Cortex, < 5W power budget)

MobileNetV3-Small was purpose-built for exactly this constraint: **2.5M parameters, ~56 GFLOPS, sub-50ms inference on CPU**.

---

## System Architecture

```
+------------------------------------------------------------------+
|                     CubeSat (Simulated)                           |
|                      satellite_ai.py                              |
|                                                                   |
|   NASA FIRMS API  -->  Real fire coordinates (lat/lon)           |
|   Local imagery   -->  MobileNetV3-Small inference               |
|                         (fire probability + confidence)           |
|                   -->  POST /api/alert  ---------------------->  |
+------------------------------------------------------------------+
                                                                 |
+----------------------------------------------------------------v-+
|                   Ground Station (server.js)                      |
|                                                                   |
|   Express.js + Socket.io                                          |
|   POST /api/alert  ->  buffer (last 500 alerts)                  |
|                     ->  io.emit("new_fire_alert")  ----------->  |
|   GET  /api/history ->  replay for fresh dashboard loads         |
|   GET  /api/stats   ->  regional breakdown (top 10 countries)    |
+------------------------------------------------------------------+
                                                                 |
+----------------------------------------------------------------v-+
|              FireSat Global Command Center (browser)              |
|                       public/index.html                           |
|                                                                   |
|   Leaflet.js world map with live fire markers                     |
|   Neon-styled mission control dashboard                           |
|   Real-time stats panel: total alerts, active regions            |
|   Alert feed: country, fire intensity (FRP/MW), confidence       |
+------------------------------------------------------------------+
```

---

## What Is 100% Real

| Component | Status | Details |
|---|---|---|
| MobileNetV3-Small model | ✅ Real | Actual PyTorch model designed for edge/low-power CPUs |
| Model inference latency | ✅ Real | Actual CPU time measured (< 50ms per frame) |
| NASA FIRMS coordinates | ✅ Real | Live fire lat/lon from the past 24 hours, globally |
| Fire Radiative Power (FRP) | ✅ Real | Real intensity measurements from NASA satellites |
| Network protocol | ✅ Real | Tiny JSON alert packets (mimics satellite downlink) |
| WebSocket streaming | ✅ Real | Socket.io real-time push to all connected browsers |
| Camera feed | Simulated | Local image files (real satellite = live sensor stream) |

---

## Key Components

### 1. Edge AI Module — `satellite_ai.py`

The satellite brain. Simulates the compute payload of a CubeSat:

- Loads the fine-tuned **MobileNetV3-Small** fire classifier (or falls back to ImageNet pretrained weights)
- Fetches **live NASA FIRMS hotspot data** — real fire coordinates detected by MODIS/VIIRS satellites in the last 24 hours
- Runs inference on local image files (stand-in for the onboard camera sensor)
- Performs reverse geocoding to identify the country of origin
- Transmits structured alert JSON to the ground station:

```json
{
  "latitude": -15.3,
  "longitude": 130.7,
  "country": "Australia",
  "fire_probability": 0.94,
  "confidence": 0.94,
  "fire_intensity": 312.5,
  "inference_latency_ms": 38.2,
  "timestamp": "2026-03-13T04:21:07Z"
}
```

### 2. Fire Classifier Training — `train_fire_model.py`

Fine-tunes MobileNetV3-Small on a binary fire/no-fire classification task:

- **Transfer learning**: ImageNet pretrained backbone, frozen during initial training
- **Head-only training first**: prevents overfitting on small datasets
- **Augmentation**: horizontal flip + color jitter for robustness to lighting and smoke conditions
- **80/20 train-val split** with PyTorch DataLoader
- Compatible with the [Kaggle Forest Fire Dataset](https://www.kaggle.com/datasets/kutaykutlu/forest-fire)
- Saves checkpoint to `models/fire_classifier.pth`

### 3. Ground Station Server — `server.js`

The mission control backend:

- **Express.js** REST API: receives satellite alerts, serves history and stats
- **Socket.io** WebSocket: broadcasts every alert to all connected browsers in real time
- **Alert buffer**: keeps last 500 events in memory for fresh page loads
- **Regional stats**: aggregates alerts by country, returns top-10 hotspot regions

### 4. Global Command Center Dashboard — `public/index.html`

A mission-control-style real-time browser UI:

- **Leaflet.js** world map with animated fire markers at real NASA coordinates
- **Live alert feed**: country, FRP (MW), confidence, timestamp
- **Stats panel**: total alerts received, active hotspot regions
- **Neon mission-control aesthetic**: dark background, cyan/red accent palette
- Zero-reload updates via Socket.io WebSocket

---

## Quickstart

### Prerequisites

- Python 3.10+
- Node.js 18+
- NASA FIRMS API key (free — get one at [firms.modaps.eosdas.nasa.gov](https://firms.modaps.eosdas.nasa.gov/api/area/))

### 1. Clone

```bash
git clone https://github.com/satyamdas03/firesat-wildfire-detection.git
cd firesat-wildfire-detection
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Create .env file
NASA_MAP_KEY=your_nasa_firms_api_key
GROUND_STATION_URL=http://localhost:3000/api/alert
```

### 4. (Optional) Train your own fire classifier

```bash
# Add images to:
#   dataset/fire/       (fire/smoke images)
#   dataset/no_fire/    (clear sky / vegetation images)

python train_fire_model.py
# Saves model to models/fire_classifier.pth
```

### 5. Start the Ground Station

```bash
npm install
node server.js
# Ground station live at http://localhost:3000
```

### 6. Launch the Edge AI Satellite Simulator

```bash
python satellite_ai.py
# Begins fetching NASA FIRMS data and running inference
# Alerts stream to ground station in real time
```

### 7. Open the Dashboard

Navigate to `http://localhost:3000` in your browser. Fire alerts appear on the global map in real time.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Edge AI model** | PyTorch, MobileNetV3-Small | 2.5M params, sub-50ms CPU inference — CubeSat-class hardware |
| **Transfer learning** | ImageNet pretrained weights | High accuracy with small fire datasets |
| **Live fire data** | NASA FIRMS API (MODIS/VIIRS) | Real global hotspot coordinates, 24hr refresh |
| **Geocoding** | reverse_geocoder | Country-level attribution of detections |
| **Ground station** | Node.js, Express.js | Lightweight, production-grade REST server |
| **Real-time push** | Socket.io (WebSocket) | Zero-latency alert streaming to dashboard |
| **Dashboard** | Leaflet.js, Vanilla JS | Interactive global map, no framework overhead |
| **Deployment** | Render.yaml included | One-click deploy to Render |

---

## Performance

| Metric | Value |
|---|---|
| Model parameters | 2.5M (MobileNetV3-Small) |
| Inference latency (CPU) | < 50ms per frame |
| Alert packet size | ~300 bytes JSON |
| Dashboard refresh | Real-time (WebSocket, 0ms delay) |
| NASA FIRMS coverage | Global, last 24 hours |
| Alert buffer | Last 500 events |

---

## Project Structure

```
firesat-wildfire-detection/
├── satellite_ai.py        # Edge AI satellite simulator (MobileNetV3 + NASA FIRMS)
├── train_fire_model.py    # Fine-tuning script for fire/no-fire classifier
├── server.js              # Ground station server (Express + Socket.io)
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── render.yaml            # Render deployment config
├── public/
│   └── index.html         # FireSat Global Command Center dashboard (Leaflet.js)
├── dataset/               # Training images (add fire/ and no_fire/ subdirs)
└── models/                # Saved model checkpoints
```

---

## Real-World Applicability

This architecture maps directly to real satellite systems:

| Simulation component | Real-world equivalent |
|---|---|
| `satellite_ai.py` loop | Onboard inference pipeline (ARM Cortex-M / Raspberry Pi CM4) |
| Local image files | Live camera sensor (e.g., OV5647 on CubeSat) |
| POST to ground station | Satellite downlink (UHF/S-band radio) |
| `server.js` | Ground station receiving software (GNU Radio + custom backend) |
| Leaflet dashboard | Ops team monitoring interface |

The model size (2.5M parameters, ~10MB) fits comfortably within CubeSat flash storage. Inference at < 50ms per frame on a standard CPU translates to real-time processing on modern embedded ARM processors.

---

## Author

**Satyam Das**
Master of Artificial Intelligence, University of Technology Sydney (Feb 2026 – Dec 2027)
B.Tech Computer Science, VIT Vellore (2025) · 3 published papers in AI/ML

GitHub: [github.com/satyamdas03](https://github.com/satyamdas03) · LinkedIn: [linkedin.com/in/satyam-das-36040a24b](https://linkedin.com/in/satyam-das-36040a24b/)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Detecting wildfires from orbit, one inference at a time.*

**MobileNetV3 · NASA FIRMS · Socket.io · Leaflet.js**

</div>
