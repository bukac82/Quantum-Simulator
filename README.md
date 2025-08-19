Quantum Simulator

A lightweight quantum computing simulator that allows users to design, simulate, and analyze quantum circuits without needing access to real quantum hardware. This project is built for learning, research, and prototyping purposes.

✨ Features

🧮 Quantum Circuit Simulation – Build and simulate circuits with common quantum gates (X, H, CNOT, etc.).

🔗 Entanglement & Superposition – Visualize fundamental quantum mechanics principles.

📊 State Vector & Probability Distribution – Inspect quantum states after each operation.

🔄 Measurement Simulation – Run multiple shots to see realistic outcome distributions.

🖥️ Extensible Design – Easily add new gates or algorithms.

📘 Educational Use – Great for learning quantum concepts without quantum hardware.

🚀 Getting Started

Clone the repo:

git clone https://github.com/your-username/quantum-simulator.git
cd quantum-simulator


Install dependencies:

pip install -r requirements.txt


Run a sample simulation:

python examples/bell_state.py

📚 Example

Creating a Bell state with H and CNOT gates:

|0> --H---■---
           |
|0> -------X---


Output (approximate distribution):

|00⟩ : 0.5  
|11⟩ : 0.5  

🔮 Roadmap

Support for quantum algorithms (Grover’s, Shor’s, QFT).

Bloch sphere visualization.

Integration with real quantum backends (Qiskit, Braket, etc.).
