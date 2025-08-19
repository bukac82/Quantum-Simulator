// ================= Algorithm Functions =================

async function runGrover() {
    try {
        showMessage('Running Grover search...', 'info');
        
        const searchArray = JSON.parse(document.getElementById('groverArray').value);
        const targetValue = document.getElementById('groverTarget').value;
        
        const request = {
            search_array: searchArray,
            target_value: targetValue
        };
        
        const response = await fetch('/api/grover/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        activeJobs[result.job_id] = 'grover';
        
        showMessage('Grover search job submitted', 'success');
        pollJobResults(result.job_id, 'algorithm');
        
    } catch (error) {
        showMessage(`Error running Grover: ${error.message}`, 'error');
    }
}

async function runShor() {
    try {
        showMessage('Running Shor factoring...', 'info');
        
        const request = {
            N: parseInt(document.getElementById('shorN').value),
            n_counting_qubits: parseInt(document.getElementById('shorCountingQubits').value)
        };
        
        const response = await fetch('/api/shor/factor', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        activeJobs[result.job_id] = 'shor';
        
        showMessage('Shor factoring job submitted', 'success');
        pollJobResults(result.job_id, 'algorithm');
        
    } catch (error) {
        showMessage(`Error running Shor: ${error.message}`, 'error');
    }
}

async function runQFT() {
    try {
        showMessage('Running Quantum Fourier Transform...', 'info');
        
        const request = {
            n_qubits: parseInt(document.getElementById('qftQubits').value)
        };
        
        const response = await fetch('/api/qft/transform', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayAlgorithmResults(result, 'QFT');
        showMessage('QFT completed successfully', 'success');
        
    } catch (error) {
        showMessage(`Error running QFT: ${error.message}`, 'error');
    }
}

function displayAlgorithmResults(result, algorithmType) {
    const resultsDiv = document.getElementById('algorithmResults');
    const contentDiv = document.getElementById('algorithmResultContent');
    
    let html = `<h5>${algorithmType} Results:</h5>`;
    
    if (algorithmType === 'Grover') {
        html += `
            <div class="result-metric">
                <span class="metric-label">Found Index:</span>
                <span class="metric-value">${result.found_index}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Found Value:</span>
                <span class="metric-value">${result.found_value}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Success:</span>
                <span class="metric-value">${result.success ? 'Yes' : 'No'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Search Space Size:</span>
                <span class="metric-value">${result.search_space_size}</span>
            </div>
        `;
    } else if (algorithmType === 'Shor') {
        html += `
            <div class="result-metric">
                <span class="metric-label">Number to Factor:</span>
                <span class="metric-value">${result.N}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Factors:</span>
                <span class="metric-value">${result.factors ? result.factors.join(' × ') : 'Not found'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Success:</span>
                <span class="metric-value">${result.success ? 'Yes' : 'No'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Verification:</span>
                <span class="metric-value">${result.verification ? 'Correct' : 'Failed'}</span>
            </div>
        `;
    } else if (algorithmType === 'QFT') {
        html += `
            <div class="result-metric">
                <span class="metric-label">Qubits:</span>
                <span class="metric-value">${result.n_qubits}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Circuit Depth:</span>
                <span class="metric-value">${result.circuit_depth}</span>
            </div>
        `;
        
        // Show sample measurements
        if (result.sample_measurements && result.sample_measurements.length > 0) {
            html += `
                <div class="solution-display">
                    <strong>Sample Measurements:</strong><br>
                    ${result.sample_measurements.slice(0, 5).join(', ')}
                </div>
            `;
        }
    }
    
    contentDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}