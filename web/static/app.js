document.addEventListener('DOMContentLoaded', () => {

    function showMessage(text, isError = false) {
        const messageDiv = document.getElementById('message') || document.getElementById('contact-message') || document.getElementById('user-message') || document.getElementById('username-message');
        if (messageDiv) {
            messageDiv.textContent = text;
            messageDiv.style.color = isError ? 'red' : 'green';
            messageDiv.style.display = 'block';
            setTimeout(() => {
                messageDiv.textContent = '';
                messageDiv.style.display = 'none';
            }, 4000);
        }
    }

    document.body.addEventListener('htmx:afterRequest', (event) => {
        const messageHeader = event.detail.xhr.getResponseHeader('X-Message');
        if (messageHeader) {
            showMessage(messageHeader);
        }
        const errorMessageHeader = event.detail.xhr.getResponseHeader('X-Error-Message');
        if (errorMessageHeader) {
            showMessage(errorMessageHeader, true);
        }
    });

    console.log('app.js loaded and DOMContentLoaded fired.'); // New log

    if (document.getElementById('detection-analytics')) {
        console.log('Dashboard element found, executing dashboard-specific logic.'); // New log
        const detectionCountsChartCtx = document.getElementById('detectionCountsChart').getContext('2d');
        const detectionsOverTimeChartCtx = document.getElementById('detectionsOverTimeChart').getContext('2d');
        const galleryCarousel = document.getElementById('gallery-carousel');
        const recentDetectionsList = document.getElementById('recent-detections-list');

        let detectionCountsChart;
        let detectionsOverTimeChart;

        function getRandomColor() {
            const letters = '0123456789ABCDEF';
            let color = '#';
            for (let i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
            }
            return color;
        }

        async function fetchDetectionAnalytics() {
            try {
                const summaryResponse = await fetch('/api/analytics/summary');
                if (summaryResponse.ok) {
                    const summary = await summaryResponse.json();
                    document.getElementById('kpi-total-detections').textContent = summary.total ?? 0;
                    document.getElementById('kpi-today-detections').textContent = summary.today ?? 0;
                }

                const countsResponse = await fetch('/api/analytics/counts');
                if (!countsResponse.ok) throw new Error(`HTTP error! status: ${countsResponse.status}`);
                const countsData = await countsResponse.json();
                const classLabels = countsData.counts.map(item => item.class);
                const classCounts = countsData.counts.map(item => item.count);

                if (detectionCountsChart) detectionCountsChart.destroy();
                detectionCountsChart = new Chart(detectionCountsChartCtx, {
                    type: 'bar',
                    data: {
                        labels: classLabels,
                        datasets: [{
                            label: 'Detection Counts',
                            data: classCounts,
                            backgroundColor: 'rgba(34, 197, 94, 0.6)',
                            borderColor: 'rgba(22, 163, 74, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: { responsive: true, scales: { y: { beginAtZero: true } } }
                });

                const timelineResponse = await fetch('/api/analytics/timeline?interval=day');
                if (!timelineResponse.ok) throw new Error(`HTTP error! status: ${timelineResponse.status}`);
                const timelineData = await timelineResponse.json();
                const labels = [...new Set(timelineData.map(item => item.time_group))].sort();
                const datasets = {};
                timelineData.forEach(item => {
                    if (!datasets[item.class]) {
                        datasets[item.class] = new Array(labels.length).fill(0);
                    }
                    const index = labels.indexOf(item.time_group);
                    datasets[item.class][index] = item.count;
                });
                const chartDatasets = Object.keys(datasets).map(className => ({
                    label: className,
                    data: datasets[className],
                    fill: false,
                    borderColor: getRandomColor(),
                    tension: 0.1
                }));

                if (detectionsOverTimeChart) detectionsOverTimeChart.destroy();
                detectionsOverTimeChart = new Chart(detectionsOverTimeChartCtx, {
                    type: 'line',
                    data: { labels: labels, datasets: chartDatasets },
                    options: { responsive: true, scales: { y: { beginAtZero: true } } }
                });

            } catch (error) {
                console.error('Error fetching analytics:', error);
                showMessage('Failed to load analytics.', true);
            }
        }

        async function fetchDetections() {
            try {
                const response = await fetch('/api/detections?limit=10');
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const detections = await response.json();

                if (recentDetectionsList) {
                    recentDetectionsList.innerHTML = '';
                    if (detections.length === 0) {
                        recentDetectionsList.innerHTML = `
                        <tr class="bg-white border-t">
                            <td class="px-6 py-4 text-gray-500" colspan="3">No recent detections.</td>
                        </tr>`;
                    } else {
                        detections.forEach(detection => {
                            const row = document.createElement('tr');
                            row.className = 'bg-white border-t';
                            row.innerHTML = `
                                <td class="px-6 py-4 whitespace-nowrap">${new Date(detection.timestamp).toLocaleString()}</td>
                                <td class="px-6 py-4">${detection.camera_name}</td>
                                <td class="px-6 py-4">${(detection.confidence * 100).toFixed(0)}%</td>
                            `;
                            recentDetectionsList.appendChild(row);
                        });
                    }
                }

                if (galleryCarousel) {
                    galleryCarousel.innerHTML = '';
                    if (detections.length === 0) {
                        galleryCarousel.innerHTML = '<div class="col-span-full"><div class="p-6 text-center border-2 border-dashed rounded-lg text-gray-500">No detections yet.</div></div>';
                    } else {
                        detections.forEach(detection => {
                            if (detection.image_path) {
                                const imgElement = document.createElement('img');
                                imgElement.src = `/${detection.image_path}`;
                                imgElement.alt = `Detection: ${detection.class}`;
                                imgElement.className = 'w-full h-auto object-cover rounded-lg shadow-md';
                                galleryCarousel.appendChild(imgElement);
                            }
                        });
                    }
                }
            } catch (error) {
                console.error('Error fetching detections:', error);
                if (galleryCarousel) showMessage('Failed to load detections.', true);
            }
        }

        fetchDetectionAnalytics();
        fetchDetections();

        setInterval(fetchDetectionAnalytics, 10000);
        setInterval(fetchDetections, 5000);
    } else {
        console.log('Dashboard element not found, skipping dashboard-specific logic.'); // New log
    }
});