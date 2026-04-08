<script>
  import { onMount } from 'svelte';
  import { stats, fetchInitialData, connectWebSocket } from '../stores/detections';
  import StatCard from '../components/StatCard.svelte';
  import ConfidenceChart from '../components/ConfidenceChart.svelte';
  import DetectionGallery from '../components/DetectionGallery.svelte';
  import DetectionTable from '../components/DetectionTable.svelte';

  onMount(() => {
    fetchInitialData();
    connectWebSocket();
  });
</script>

<div class="page-content">
  <!-- Header -->
  <div class="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-2">
    <div>
      <span class="text-secondary font-label text-xs font-semibold uppercase tracking-widest mb-2 block">
        Conservation Sentinel
      </span>
      <h1 class="text-4xl font-extrabold text-on-surface tracking-tight font-headline">
        Monitoring Dashboard
      </h1>
      <p class="text-on-surface-variant mt-2 max-w-xl text-sm">
        Real-time telemetry and species detection analytics from the Palawan surveillance network.
      </p>
    </div>
    <div class="flex items-center gap-3">
      <span class="inline-flex items-center gap-2 px-3 py-1.5 bg-primary-fixed text-primary rounded-full text-[10px] font-bold uppercase tracking-tighter">
        <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
        System Active
      </span>
    </div>
  </div>

  <!-- Stats Grid -->
  <section class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <StatCard 
      label="Total Detections" 
      value={$stats.total.toLocaleString()} 
      icon="analytics" 
      trend="+12% vs last week"
      variant="primary"
    />
    <StatCard 
      label="Detections Today" 
      value={$stats.today.toString()} 
      icon="today" 
      trend="Peak: 14:00"
      variant="secondary"
    />
    <StatCard 
      label="This Month" 
      value={$stats.this_month.toLocaleString()} 
      icon="calendar_month" 
      trend="Sept 2024"
      variant="tertiary"
    />
  </section>

  <!-- Middle Section: Chart & Gallery -->
  <section class="grid grid-cols-1 lg:grid-cols-3 gap-8">
    <div class="lg:col-span-2 bg-surface-container-lowest p-8 rounded-xl shadow-sm flex flex-col">
      <div class="flex items-center justify-between mb-8">
        <div>
          <h2 class="text-xl font-bold text-primary font-headline">Confidence History</h2>
          <p class="text-on-surface-variant text-sm">AI prediction accuracy over the last 24 hours</p>
        </div>
        <div class="flex gap-2">
          <button class="px-4 py-1.5 bg-primary text-on-primary rounded-full text-[10px] font-bold uppercase">24h</button>
          <button class="px-4 py-1.5 bg-surface-container-high text-on-surface-variant rounded-full text-[10px] font-bold uppercase">7d</button>
        </div>
      </div>
      
      <ConfidenceChart />
      
      <div class="flex justify-between mt-4 text-[10px] text-on-surface-variant font-label uppercase tracking-widest font-bold">
        <span>00:00</span>
        <span>06:00</span>
        <span>12:00</span>
        <span>18:00</span>
        <span>Now</span>
      </div>
    </div>

    <!-- Gallery Column -->
    <DetectionGallery />
  </section>

  <!-- Recent Detections Table -->
  <section>
    <DetectionTable />
  </section>
</div>
