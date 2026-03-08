document.addEventListener('DOMContentLoaded', () => {

    const metricsData = JSON.parse(
        document.getElementById('metrics-data').textContent || "{}"
    );

    const accData = metricsData.accuracy || [];
    const f1Data = metricsData.f1_score || [];

    const epochs = accData.map((_, i) => i + 1);

    const commonOptions = {
        responsive: true,
        plugins: {
            legend: {
                labels: { color: '#ffffff' }
            }
        },
        scales: {
            x: {
                ticks: { color: '#ffffff' }
            },
            y: {
                min: 0.6,
                max: 1,
                ticks: { color: '#ffffff' }
            }
        }
    };

    // Accuracy Graph
    new Chart(document.getElementById('accuracyChart'), {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [{
                label: 'Training Accuracy',
                data: accData,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99,102,241,0.25)',
                fill: true,
                tension: 0.4
            }]
        },
        options: commonOptions
    });

    // F1 Graph
    new Chart(document.getElementById('f1Chart'), {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [{
                label: 'F1 Score',
                data: f1Data,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16,185,129,0.25)',
                fill: true,
                tension: 0.4
            }]
        },
        options: commonOptions
    });

});