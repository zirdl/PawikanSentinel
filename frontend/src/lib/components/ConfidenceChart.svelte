<script>
  import { onMount } from 'svelte';
  import { Line } from 'svelte-chartjs';
  import {
    Chart as ChartJS,
    Title,
    Tooltip,
    Legend,
    LineElement,
    LinearScale,
    PointElement,
    CategoryScale,
    Filler
  } from 'chart.js';

  ChartJS.register(
    Title,
    Tooltip,
    Legend,
    LineElement,
    LinearScale,
    PointElement,
    CategoryScale,
    Filler
  );

  export let period = 'month';

  let chartData = {
    labels: [],
    datasets: [
      {
        label: 'Confidence (%)',
        data: [],
        fill: true,
        borderColor: '#134231', // primary
        backgroundColor: 'rgba(188, 237, 212, 0.4)', // primary-fixed (with opacity)
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#134231'
      }
    ]
  };

  let chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#191c1c', // on-surface
        titleFont: { family: 'Manrope' },
        bodyFont: { family: 'Inter' }
      }
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        grid: { display: false },
        ticks: { 
          color: '#414944',
          callback: function(value) { return value + '%'; }
        } // on-surface-variant
      },
      x: {
        grid: { display: false },
        ticks: { color: '#414944' }
      }
    }
  };

  async function fetchChartData() {
    try {
      const res = await fetch(`/api/detections/chart?period=${period}`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        
        // Map data based on labels if it's months or hours
        let labels = data.map(d => d.date);
        if (period === 'year') {
          const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
          labels = data.map(d => monthNames[parseInt(d.date) - 1]);
        }

        chartData = {
          labels: labels,
          datasets: [{
            ...chartData.datasets[0],
            data: data.map(d => d.confidence)
          }]
        };
      }
    } catch (error) {
      console.error('Error fetching chart data:', error);
    }
  }

  $: period && fetchChartData();

  onMount(fetchChartData);
</script>

<div class="h-64 lg:h-80 w-full relative">
  {#if chartData.labels.length > 0}
    <Line data={chartData} options={chartOptions} />
  {:else}
    <div class="flex items-center justify-center h-full text-on-surface-variant italic">
      Loading chart data...
    </div>
  {/if}
</div>
