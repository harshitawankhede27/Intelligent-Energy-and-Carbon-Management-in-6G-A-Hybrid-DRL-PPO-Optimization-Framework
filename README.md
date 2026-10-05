<h1>Intelligent Energy and Carbon Management in 6G: A Hybrid DRL-PPO Framework</h1>
<h3><b>Manuscript ID: IEEE LATAM Submission ID: 10658 Authors:</b></h3>

-  Harshita Patil 
-  Dr. Kaushlendra Sharma 
-  Dr. Shishupal Kumar 
<h4>Overview</h4>
This repository contains the source code and simulation environment for an Intelligent Energy and Carbon Management System designed for 6G networks.

The project implements a Deep Reinforcement Learning (DRL) agent using Proximal Policy Optimization (PPO) to optimize the trade-off between energy consumption, carbon emissions, and Quality of Service (QoS). By leveraging real-time grid carbon intensity data and stochastic traffic modeling (MMPP), the agent dynamically switches base station modes (Sleep, Eco, Boost) to minimize environmental impact without violating URLLC constraints.

Key Features
Carbon-Aware Decision Making: Integrates real-time Grid Carbon Intensity (gCO2/kWh) into the control loop.
6G Power Scaling: Simulates Terahertz (THz) power consumption profiles using dynamic scaling factors verified against operational traces.
MMPP Traffic Modeling: Simulates bursty 6G traffic (e.g., Holographic MIMO, XR) using Markov Modulated Poisson Processes.
PPO Implementation: Uses Stable-Baselines3 for stable and efficient policy gradient training.
Custom Gym Environment: A fully OpenAI Gym/Gymnasium compatible environment representing a 6G Base Station.
Installation
Clone the repository:

Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Dependencies
python >= 3.8
numpy
pandas
matplotlib
gymnasium (or gym)
stable-baselines3
torch
Results

The proposed framework achieves:

84.5% reduction in carbon emissions compared to legacy baselines.

Zero QoS violations (0% packet drop rate) during high-traffic bursts.

Dynamic adaptation to "Green Windows" (periods of low grid carbon intensity).


# Intelligent Energy and Carbon Management in 6G: A Hybrid DRL-PPO Framework

**Manuscript ID:** IEEE LATAM Submission ID: 10658

## 👥 Authors

* **Harshita Patil**
* **Dr. Kaushlendra Sharma**
* **Dr. Shishupal Kumar**

---

## 📌 Overview

This repository contains the source code, datasets, and simulation environment developed for the research paper **"Intelligent Energy and Carbon Management in 6G: A Hybrid DRL-PPO Framework."**

The proposed framework uses **Deep Reinforcement Learning (DRL)** with **Proximal Policy Optimization (PPO)** to optimize the trade-off between:

* ⚡ Energy consumption
* 🌱 Carbon emissions
* 📡 Quality of Service (QoS)

The framework incorporates **grid carbon intensity**, **stochastic traffic modeling using Markov Modulated Poisson Processes (MMPP)**, and **6G power-consumption models** to dynamically control the operating mode of a 6G base station.

The agent dynamically selects among three operating modes:

* **Sleep** — Minimize energy consumption during low traffic.
* **Eco** — Balance energy consumption and QoS.
* **Boost** — Provide higher capacity during high-traffic conditions.

The objective is to reduce carbon emissions while maintaining QoS requirements for 6G services, including Ultra-Reliable Low-Latency Communications (URLLC).

---

## ✨ Key Features

### 🌱 Carbon-Aware Decision Making

Integrates grid carbon intensity measured in **gCO₂/kWh** into the reinforcement learning decision process, allowing the agent to adapt its operation according to the environmental conditions of the electricity grid.

### ⚡ 6G Power Scaling

Simulates power-consumption characteristics of future 6G networks using dynamic power-scaling factors associated with advanced wireless technologies, including **Terahertz (THz)** communication and high-capacity network scenarios.

### 📊 MMPP Traffic Modeling

Models bursty and time-varying 6G traffic using **Markov Modulated Poisson Processes (MMPP)** to represent demanding applications such as:

* Holographic communications
* Extended Reality (XR)
* Multimedia services
* High-data-rate 6G applications

### 🤖 PPO-Based Optimization

Implements **Proximal Policy Optimization (PPO)** using the **Stable-Baselines3** framework to learn an energy- and carbon-aware base-station control policy.

### 🧪 Custom 6G Environment

Provides a custom **Gymnasium/Gym-compatible reinforcement learning environment** representing the operating conditions and decision-making process of a 6G base station.

---

## 🏗️ System Framework

The overall framework consists of the following major components:

```text
                 6G Network Environment
                          │
          ┌───────────────┼────────────────┐
          │               │                │
     Traffic Data    Grid Carbon      Channel Data
          │           Intensity             │
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                 Custom 6G Environment
                          │
                          ▼
                    PPO Agent
                          │
             ┌────────────┼────────────┐
             │            │            │
           Sleep         Eco         Boost
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
              Energy & Carbon Optimization
                          │
                          ▼
                    QoS Evaluation
```

---

## 📂 Repository Contents

| **File**                 | **Type**      | **Description**                                                                  |
| ------------------------ | ------------- | -------------------------------------------------------------------------------- |
| `section1.py`            | Python        | Implements the first section/component of the simulation framework.              |
| `section2.py`            | Python        | Implements the second section/component of the simulation framework.             |
| `graph.py`               | Python        | Generates graphs and visualizations from simulation results.                     |
| `5g_energy_data.csv`     | Dataset       | Energy-related data used for the simulation and comparative analysis.            |
| `6g_multimedia_data.csv` | Dataset       | Multimedia/traffic data representing 6G application scenarios.                   |
| `6g_scaled_data.csv`     | Dataset       | Scaled 6G data used for simulation and power/traffic modeling.                   |
| `channel_trace.csv`      | Dataset       | Channel-trace data used to represent wireless channel conditions.                |
| `README.md`              | Documentation | Project description, installation instructions, and reproducibility information. |

---

## 📊 Datasets

The repository includes the following datasets:

### `5g_energy_data.csv`

Contains energy-related information used for comparison and energy-consumption modeling.

### `6g_multimedia_data.csv`

Contains multimedia traffic information used to model demanding 6G services and traffic conditions.

### `6g_scaled_data.csv`

Contains scaled data used by the 6G simulation framework for network and energy modeling.

### `channel_trace.csv`

Contains channel-trace information used to represent changing wireless communication conditions.

---

## 💻 Requirements

The framework requires:

* **Python 3.8 or later**
* NumPy
* Pandas
* Matplotlib
* Gymnasium or Gym
* Stable-Baselines3
* PyTorch

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/harshitawankhede27/Intelligent-Energy-and-Carbon-Management-in-6G-A-Hybrid-DRL-PPO-Optimization-Framework.git
```

Navigate to the repository:

```bash
cd Intelligent-Energy-and-Carbon-Management-in-6G-A-Hybrid-DRL-PPO-Optimization-Framework
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

If the implementation uses the older Gym API, install:

```bash
pip install gym
```

---

## ▶️ Running the Code

After installing the dependencies, the main simulation components can be executed using Python.

For example:

```bash
python section1.py
```

and:

```bash
python section2.py
```

To generate the graphs:

```bash
python graph.py
```

Ensure that the required `.csv` datasets are located in the repository directory before running the scripts.

---

## 📈 Results

The proposed hybrid DRL-PPO framework demonstrates:

* **84.5% reduction in carbon emissions** compared with the considered legacy baseline.
* **0% packet drop rate** under the evaluated high-traffic scenarios.
* Dynamic adaptation to **low-carbon ("Green Window") periods**.
* Energy-aware switching between **Sleep, Eco, and Boost** operating modes.
* Improved coordination between **energy consumption, carbon intensity, and QoS requirements**.

> **Note:** The reported values correspond to the experimental configuration and scenarios described in the associated manuscript.

---

## 🔬 Reproducibility

This repository provides the source code and datasets required to reproduce the simulation experiments reported in the manuscript.

For reproducibility:

1. Install the required Python dependencies.
2. Download/clone the repository.
3. Keep the provided `.csv` files in the repository directory.
4. Execute the simulation scripts.
5. Use `graph.py` to generate the corresponding visualizations.
6. Compare the resulting metrics with those reported in the manuscript.

---

## 📚 Research Context

The framework investigates the application of **Deep Reinforcement Learning** for sustainable 6G network management.

The optimization considers three primary objectives:

**Energy Efficiency + Carbon Reduction + QoS Preservation**

The approach combines:

* Deep Reinforcement Learning
* Proximal Policy Optimization (PPO)
* Markov Modulated Poisson Processes (MMPP)
* Carbon-aware network management
* 6G power-consumption modeling
* Dynamic base-station operation
* QoS-aware decision making

---

## 📄 Manuscript

**Title:** *Intelligent Energy and Carbon Management in 6G: A Hybrid DRL-PPO Framework*

**Manuscript ID:** IEEE LATAM Submission ID: 10658

**Authors:**

* Harshita Patil
* Dr. Kaushlendra Sharma
* Dr. Shishupal Kumar

---

## ✉️ Contact

For questions regarding the source code, datasets, simulation environment, or reproduction of the results:

**Harshita Patil**

📧 **[Contact Author](mailto:harshitapatil@example.com)**

---

## 📜 License

This repository is intended for **research and academic purposes**.

Please cite the associated manuscript if you use the code, datasets, or methodology in your research.
