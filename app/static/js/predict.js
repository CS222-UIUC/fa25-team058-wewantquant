let predictionChart = null;

document.getElementById('predict-btn').addEventListener('click', function() {
    // Get form values
    const model = document.getElementById('model-select').value;
    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
    const days = document.getElementById('days-input').value;

    // Validation
    if (!model) {
        showError('Please select a model');
        return;
    }
    if (!ticker) {
        showError('Please enter a stock ticker');
        return;
    }

    // Hide error and results
    document.getElementById('error-message').classList.remove('show');
    document.getElementById('results-section').classList.remove('show');
    
    // Show loading
    document.getElementById('loading').classList.add('show');
    document.getElementById('predict-btn').disabled = true;

    // Send AJAX request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model,
            ticker: ticker,
            days: days
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading
        document.getElementById('loading').classList.remove('show');
        document.getElementById('predict-btn').disabled = false;

        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading').classList.remove('show');
        document.getElementById('predict-btn').disabled = false;
        showError('An error occurred while generating prediction');
    });
});

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
}

function displayResults(data) {
    // Update title
    document.getElementById('results-title').textContent = 
        `Prediction Results for ${data.ticker} - ${data.model_name}`;
    
    // Display summary stats
    displaySummaryStats(data);
    
    // Display chart
    displayChart(data);
    
    // Display table
    displayTable(data);
    
    // Show results section
    document.getElementById('results-section').classList.add('show');
}

function displaySummaryStats(data) {
    const summaryStats = document.getElementById('summary-stats');
    
    const finalPrediction = data.predictions[data.predictions.length - 1];
    const totalChange = ((finalPrediction.price - data.current_price) / data.current_price * 100).toFixed(2);
    const changeClass = totalChange >= 0 ? 'positive' : 'negative';
    const changeSymbol = totalChange >= 0 ? '▲' : '▼';
    
    summaryStats.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Current Price</div>
            <div class="stat-value">$${data.current_price.toFixed(2)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Final Prediction (Day ${data.predictions.length})</div>
            <div class="stat-value ${changeClass}">$${finalPrediction.price.toFixed(2)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Expected Change</div>
            <div class="stat-value ${changeClass}">
                ${changeSymbol} ${totalChange}%
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Model Confidence</div>
            <div class="stat-value">${data.confidence}%</div>
        </div>
    `;
}

function displayChart(data) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    // Prepare data for chart
    const labels = ['Current', ...data.predictions.map(p => p.date)];
    const prices = [data.current_price, ...data.predictions.map(p => p.price)];
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(54, 162, 235, 0.5)');
    gradient.addColorStop(1, 'rgba(54, 162, 235, 0.1)');
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Stock Price',
                data: prices,
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: gradient,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return 'Price: $' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function displayTable(data) {
    const tbody = document.getElementById('predictions-tbody');
    tbody.innerHTML = '';
    
    let previousPrice = data.current_price;
    
    data.predictions.forEach(prediction => {
        const change = prediction.price - previousPrice;
        const changePercent = (change / previousPrice * 100).toFixed(2);
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSymbol = change >= 0 ? '▲' : '▼';
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${prediction.date}</td>
            <td>$${prediction.price.toFixed(2)}</td>
            <td style="color: ${change >= 0 ? '#27ae60' : '#e74c3c'}">
                ${changeSymbol} $${Math.abs(change).toFixed(2)}
            </td>
            <td style="color: ${change >= 0 ? '#27ae60' : '#e74c3c'}">
                ${changeSymbol} ${Math.abs(changePercent)}%
            </td>
        `;
        tbody.appendChild(row);
        
        previousPrice = prediction.price;
    });
}