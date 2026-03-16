Intelligent Energy and Carbon Management in 6G: A Hybrid DRL-PPO Framework
Overview
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
