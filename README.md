# CAD LiDAR Anomaly Detection

Simulation and evaluation of a Cross-Sensor Anomaly Detection (CAD) framework against LiDAR data fabrication attacks in Connected and Autonomous Vehicles (CAVs).

---

## Problem Statement

Autonomous vehicles rely heavily on LiDAR perception for occupancy mapping and object detection. However, LiDAR sensors are vulnerable to data fabrication attacks such as:

- **Spoof_RC** – injecting fake reflective clusters
- **Remove_RC** – removing legitimate reflective clusters

This project simulates adversarial manipulation of LiDAR data and evaluates a cross-sensor anomaly detection mechanism designed to detect such attacks.

---

## System Architecture

The implemented pipeline consists of:

1. **Perception Layer**
   - Occupancy grid mapping
   - LiDAR-based environment modeling

2. **Attack Simulation**
   - Reflective cluster spoofing
   - Reflective cluster removal

3. **Cross-Sensor Anomaly Detection (CAD)**
   - Consistency-based validation
   - Feature-based anomaly scoring

4. **Evaluation Framework**
   - ROC Curve analysis
   - Precision-Recall analysis
   - Threshold tuning

---

## Evaluation Metrics

The detection performance was evaluated using:

- **ROC-AUC**
- **PR-AUC**
- False Positive Rate (FPR)
- True Positive Rate (TPR)

The CAD framework demonstrated strong robustness against spoofing-based manipulations while highlighting sensitivity trade-offs under removal attacks.

---

## Project Structure
cad-lidar-anomaly-detection/
│
├── src/
│ ├── attacker_rc.py # LiDAR attack simulation (Spoof / Remove)
│ ├── cad_detector.py # Cross-sensor anomaly detection logic
│ ├── eval.py # Basic evaluation
│ ├── eval_advanced.py # Advanced performance analysis
│ ├── occupancy_map.py # Grid mapping implementation
│ ├── perception.py # LiDAR perception modeling
│ └── simulation.py # End-to-end simulation pipeline
│
├── demo_spyder.py # Example execution script
├── requirements.txt # Python dependencies
└── env.yml # Conda environment configuration


---

## Installation

Using pip:

```bash
pip install -r requirements.txt


