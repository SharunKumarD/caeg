let mainChart, donutChart;
let lastPredictionError = null;

const COLORS = {
    cyan: '#06b6d4',
    purple: '#a855f7',
    orange: '#f97316',
    teal: '#14b8a6',
    white: '#f3f4f6',
    gray: '#4b5563'
};

document.addEventListener('DOMContentLoaded', async () => {
    await loadScenarios();
    initCharts();
    
    document.getElementById('simulate-btn').addEventListener('click', runSimulationStep);
    document.getElementById('scenario-select').addEventListener('change', () => {
        lastPredictionError = null; // Reset closed-loop on new scenario
        runSimulationStep();
    });
    
    document.getElementById('dataset-select').addEventListener('change', () => {
        lastPredictionError = null; // Reset closed-loop on new dataset
        runSimulationStep();
    });

    // Checkbox listeners
    ['lstm', 'tcn', 'cnn'].forEach(expert => {
        document.getElementById(`toggle-${expert}`).addEventListener('change', (e) => {
            toggleChartSeries(expert, e.target.checked);
        });
    });

    // Initial run
    runSimulationStep();
});

async function loadScenarios() {
    try {
        const res = await fetch('/sample-scenarios');
        const data = await res.json();
        const select = document.getElementById('scenario-select');
        select.innerHTML = '';
        
        data.scenarios.forEach(sc => {
            const opt = document.createElement('option');
            opt.value = sc.id;
            opt.textContent = sc.name;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to load scenarios", err);
    }
}

async function runSimulationStep() {
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    btnText.textContent = "Simulating...";
    btnLoader.classList.remove('hidden');

    const scenarioId = document.getElementById('scenario-select').value;
    const datasetType = document.getElementById('dataset-select').value;
    const payload = { 
        scenario: scenarioId,
        dataset_type: datasetType
    };
    
    // Closed-loop: feed last error
    if (lastPredictionError) {
        payload.previous_prediction_error = lastPredictionError;
    }

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        // Update closed-loop state for NEXT step
        if (data.actual_next_24h && data.forecast_24h) {
            lastPredictionError = data.forecast_24h.map((f, i) => f - data.actual_next_24h[i]);
        }

        updateDashboard(data);
    } catch (err) {
        console.error("Simulation step failed", err);
    } finally {
        btnText.textContent = "Simulate Next Step (Loop)";
        btnLoader.classList.add('hidden');
    }
}

function updateDashboard(data) {
    updateContextCards(data.context_features);
    updateGatingUI(data.gating_weights, data.context_features);
    updateAblationTable(data);
    updateCharts(data);
}

function updateContextCards(ctx) {
    // Trend
    const trend = Math.abs(ctx.trend_strength);
    document.getElementById('ctx-trend-val').textContent = trend.toFixed(3);
    document.getElementById('ctx-trend-bar').style.width = `${Math.min(trend * 20, 100)}%`;

    // Volatility
    const vol = ctx.volatility_level;
    document.getElementById('ctx-vol-val').textContent = vol.toFixed(3);
    let volWidth = Math.min(vol * 50, 100);
    const volBar = document.getElementById('ctx-vol-bar');
    const volBadge = document.getElementById('ctx-vol-badge');
    volBar.style.width = `${volWidth}%`;
    
    if (vol < 0.5) {
        volBar.className = 'meter-fill bg-emerald-500';
        volBadge.className = 'text-xs px-2 py-0.5 rounded-full bg-emerald-900/50 text-emerald-400 border border-emerald-800';
        volBadge.textContent = 'Low';
    } else if (vol < 1.0) {
        volBar.className = 'meter-fill bg-amber-500';
        volBadge.className = 'text-xs px-2 py-0.5 rounded-full bg-amber-900/50 text-amber-400 border border-amber-800';
        volBadge.textContent = 'Med';
    } else {
        volBar.className = 'meter-fill bg-red-500';
        volBadge.className = 'text-xs px-2 py-0.5 rounded-full bg-red-900/50 text-red-400 border border-red-800';
        volBadge.textContent = 'High';
    }

    // Periodicity
    const period = Math.abs(ctx.periodicity_strength);
    document.getElementById('ctx-period-val').textContent = period.toFixed(3);
    document.getElementById('ctx-period-bar').style.width = `${period * 100}%`;

    // Error
    const err = ctx.recent_error;
    document.getElementById('ctx-err-val').textContent = err.toFixed(3);
    document.getElementById('ctx-err-bar').style.width = `${Math.min(err * 100, 100)}%`;
}

function updateGatingUI(weights, ctx) {
    const wLSTM = weights.LSTM * 100;
    const wTCN = weights.TCN * 100;
    const wCNN = weights.CNN * 100;

    document.getElementById('txt-w-lstm').textContent = wLSTM.toFixed(1) + '%';
    document.getElementById('txt-w-tcn').textContent = wTCN.toFixed(1) + '%';
    document.getElementById('txt-w-cnn').textContent = wCNN.toFixed(1) + '%';

    // Update Donut Chart
    donutChart.data.datasets[0].data = [weights.LSTM, weights.TCN, weights.CNN];
    donutChart.update();

    // Natural Language Explainability
    let explanation = "Standard periodic load observed. ";
    if (ctx.volatility_level > 1.0) {
        explanation = `Elevated volatility detected (${ctx.volatility_level.toFixed(2)}) -> Gate allocated combined ${(wCNN + wTCN).toFixed(0)}% weight to CNN/TCN for rapid spatial/temporal feature response while dampening LSTM.`;
    } else if (ctx.recent_error > 0.5) {
        explanation = `High closed-loop error detected (${ctx.recent_error.toFixed(2)}). Gating network shifting distribution to correct systemic bias dynamically.`;
    } else if (ctx.periodicity_strength > 0.8) {
        explanation = `Strong periodicity (${ctx.periodicity_strength.toFixed(2)}) detected. Gating heavily favors LSTM (${wLSTM.toFixed(0)}%) to leverage long-term historical dependencies.`;
    } else {
        explanation = `Mixed signals detected. Weights balanced optimally based on context encodings (Trend: ${ctx.trend_strength.toFixed(2)}, Err: ${ctx.recent_error.toFixed(2)}).`;
    }
    document.getElementById('explain-text').textContent = explanation;
}

function calculateMetrics(pred, actual) {
    if (!pred || !actual) return { MAE: '-', RMSE: '-', MAPE: '-' };
    let n = pred.length;
    let mae = 0, rmse = 0, mape = 0;
    
    for(let i=0; i<n; i++) {
        let err = Math.abs(pred[i] - actual[i]);
        mae += err;
        rmse += err*err;
        if(Math.abs(actual[i]) > 1e-8) {
            mape += Math.abs(err / actual[i]);
        }
    }
    
    return {
        MAE: (mae/n).toFixed(3),
        RMSE: Math.sqrt(rmse/n).toFixed(3),
        MAPE: ((mape/n)*100).toFixed(2) + '%'
    };
}

function updateAblationTable(data) {
    const tbody = document.getElementById('ablation-body');
    tbody.innerHTML = '';
    
    if (data.benchmark_table && data.benchmark_table.length > 0) {
        data.benchmark_table.forEach(m => {
            const tr = document.createElement('tr');
            
            // Check if it's the Proposed model to highlight it
            const isProposed = m.Model.includes('Proposed') || m.Model.includes('CAEG-Net (Proposed)');
            
            if (isProposed) {
                tr.className = 'caeg-row text-yellow-400 font-bold bg-gray-800/80';
            } else {
                tr.className = 'table-row-hover text-gray-300';
            }
            
            // Assuming metrics are returned as floats in the CSV/Dict
            const mse = m.MSE !== undefined ? Number(m.MSE).toFixed(4) : '-';
            const mae = m.MAE !== undefined ? Number(m.MAE).toFixed(4) : '-';
            const rmse = m.RMSE !== undefined ? Number(m.RMSE).toFixed(4) : '-';
            const r2 = m.R2 !== undefined ? Number(m.R2).toFixed(4) : '-';
            
            let modelName = m.Model;
            if (isProposed) {
                modelName = '★ ' + modelName;
            }
            
            tr.innerHTML = `
                <td>${modelName}</td>
                <td>${mse}</td>
                <td>${mae}</td>
                <td>${rmse}</td>
                <td>${r2}</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        // Fallback or empty state
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-gray-500">No benchmark data available.</td></tr>';
    }
}

function initCharts() {
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // Main Chart
    const ctxMain = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctxMain, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });

    // Donut Chart
    const ctxDonut = document.getElementById('donutChart').getContext('2d');
    donutChart = new Chart(ctxDonut, {
        type: 'doughnut',
        data: {
            labels: ['LSTM', 'TCN', 'CNN'],
            datasets: [{
                data: [33, 33, 34],
                backgroundColor: [COLORS.purple, COLORS.orange, COLORS.teal],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateCharts(data) {
    const labels = Array.from({length: 24}, (_, i) => `T+${i+1}`);
    
    const datasets = [];

    // Actual
    if (data.actual_next_24h) {
        datasets.push({
            label: 'Ground Truth',
            data: data.actual_next_24h,
            borderColor: 'rgba(255, 255, 255, 0.5)',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3
        });
    }

    // Experts (Index 1, 2, 3)
    const showLstm = document.getElementById('toggle-lstm').checked;
    const showTcn = document.getElementById('toggle-tcn').checked;
    const showCnn = document.getElementById('toggle-cnn').checked;

    datasets.push({
        label: 'LSTM Expert',
        data: data.expert_predictions.LSTM,
        borderColor: COLORS.purple,
        borderWidth: 1.5,
        tension: 0.3,
        hidden: !showLstm
    });
    datasets.push({
        label: 'TCN Expert',
        data: data.expert_predictions.TCN,
        borderColor: COLORS.orange,
        borderWidth: 1.5,
        tension: 0.3,
        hidden: !showTcn
    });
    datasets.push({
        label: 'CNN Expert',
        data: data.expert_predictions.CNN,
        borderColor: COLORS.teal,
        borderWidth: 1.5,
        tension: 0.3,
        hidden: !showCnn
    });

    // CAEG-Net
    datasets.push({
        label: 'CAEG-Net Forecast',
        data: data.forecast_24h,
        borderColor: COLORS.cyan,
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.3
    });

    mainChart.data.labels = labels;
    mainChart.data.datasets = datasets;
    mainChart.update();
}

function toggleChartSeries(expert, isVisible) {
    if (!mainChart) return;
    
    let datasetIndex = -1;
    if (expert === 'lstm') datasetIndex = 1;
    if (expert === 'tcn') datasetIndex = 2;
    if (expert === 'cnn') datasetIndex = 3;

    if (datasetIndex !== -1 && mainChart.data.datasets[datasetIndex]) {
        mainChart.data.datasets[datasetIndex].hidden = !isVisible;
        mainChart.update();
    }
}
