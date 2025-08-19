"""
Circuit execution routes for the Quantum Computing API
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from datetime import datetime
from api_models import CircuitRequest, JobResponse
from api_utils import create_job_id, store_job_result, convert_numpy_types, initialize_job_status

try:
    from simulator import QuantumSimulator
except ImportError:
    QuantumSimulator = None

router = APIRouter(prefix="/api/circuit", tags=["circuit"])

@router.post("/execute", response_model=JobResponse)
async def execute_circuit(request: CircuitRequest, background_tasks: BackgroundTasks, 
                         job_storage: dict, active_simulators: dict):
    """Execute a custom quantum circuit"""
    if QuantumSimulator is None:
        raise HTTPException(status_code=503, detail="Quantum simulator not available")
    
    job_id = create_job_id()
    
    def run_circuit():
        try:
            initialize_job_status(job_storage, job_id)
            
            sim = QuantumSimulator(n_qubits=request.n_qubits)
            total_gates = len(request.gates)
            
            # Apply gates with progress tracking
            for i, gate_op in enumerate(request.gates):
                gate_name = gate_op.gate.upper()
                qubits = gate_op.qubits
                params = gate_op.params or {}
                
                # Progress update
                progress = 0.1 + (0.7 * (i + 1) / total_gates)
                job_storage[job_id]["progress"] = progress
                
                # Apply gate based on type
                if gate_name == "H" and len(qubits) == 1:
                    sim.h(qubits[0])
                elif gate_name == "X" and len(qubits) == 1:
                    sim.x(qubits[0])
                elif gate_name == "Y" and len(qubits) == 1:
                    sim.y(qubits[0])
                elif gate_name == "Z" and len(qubits) == 1:
                    sim.z(qubits[0])
                elif gate_name == "S" and len(qubits) == 1:
                    sim.s(qubits[0])
                elif gate_name == "T" and len(qubits) == 1:
                    sim.t(qubits[0])
                elif gate_name == "CNOT" and len(qubits) == 2:
                    sim.cnot(qubits[0], qubits[1])
                elif gate_name == "CZ" and len(qubits) == 2:
                    sim.cz(qubits[0], qubits[1])
                elif gate_name == "RX" and len(qubits) == 1:
                    angle = params.get("angle", 0)
                    sim.rx(angle, qubits[0])
                elif gate_name == "RY" and len(qubits) == 1:
                    angle = params.get("angle", 0)
                    sim.ry(angle, qubits[0])
                elif gate_name == "RZ" and len(qubits) == 1:
                    angle = params.get("angle", 0)
                    sim.rz(angle, qubits[0])
                else:
                    raise ValueError(f"Unsupported gate: {gate_name} with {len(qubits)} qubits")
            
            # Progress: measurements
            job_storage[job_id]["progress"] = 0.9
            
            # Get results
            measurements = sim.measure(shots=request.shots)
            probabilities = sim.get_probabilities().numpy().tolist()
            state_vector = sim.get_state().numpy()
            
            # Convert complex state vector for JSON
            state_vector_json = [{"real": float(x.real), "imag": float(x.imag)} for x in state_vector]
            
            result = {
                "measurements": measurements,
                "probabilities": probabilities,
                "state_vector": state_vector_json,
                "circuit_depth": sim.get_circuit_depth(),
                "gate_sequence": sim.gate_sequence,
                "n_qubits": request.n_qubits,
                "shots": request.shots,
                "memory_usage": sim.get_memory_usage()
            }
            
            # Store simulator for potential future use
            active_simulators[job_id] = sim
            
            store_job_result(job_storage, job_id, convert_numpy_types(result), progress=1.0)
            
        except Exception as e:
            store_job_result(job_storage, job_id, None, str(e))
    
    background_tasks.add_task(run_circuit)
    
    return JobResponse(
        job_id=job_id,
        status="running",
        created_at=datetime.now().isoformat(),
        progress=0.0
    )