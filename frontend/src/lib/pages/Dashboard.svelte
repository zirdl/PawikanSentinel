<script>
  import { onMount } from "svelte";
  import {
    stats,
    fetchInitialData,
    connectWebSocket,
  } from "../stores/detections";
  import StatCard from "../components/StatCard.svelte";
  import ConfidenceChart from "../components/ConfidenceChart.svelte";
  import DetectionGallery from "../components/DetectionGallery.svelte";
  import DetectionTable from "../components/DetectionTable.svelte";

  let selectedPeriod = "day";

  onMount(() => {
    fetchInitialData();
    connectWebSocket();
  });
</script>

<!-- Header -->
<div class="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-2">
  <div>
    <span
      class="text-secondary font-label text-xs font-semibold uppercase tracking-widest mb-2 block"
    >
      CURMA
    </span>
    <h1
      class="text-4xl font-extrabold text-on-surface tracking-tight font-headline"
    >
      Monitoring Dashboard
    </h1>
    <p class="text-on-surface-variant mt-2 max-w-xl text-sm">
      Real-time telemetry and detection analytics for Coastal Underwater
      Resource Management Actions (CURMA).
    </p>
  </div>
  <div class="flex items-center gap-3">
    <span
      class="inline-flex items-center gap-2 px-3 py-1.5 bg-primary-fixed text-primary rounded-full text-[10px] font-bold uppercase tracking-tighter"
    >
      <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
      System Active
    </span>
  </div>
</div>

<!-- Stats Grid -->
<section class="grid grid-cols-1 md:grid-cols-3 gap-6 my-10">
  <StatCard
    label="Total Detections"
    value={$stats.total.toLocaleString()}
    icon="analytics"
    trend={$stats.week_increase >= 0
      ? `+${$stats.week_increase}% vs last week`
      : `${$stats.week_increase}% vs last week`}
    variant="primary"
  />
  <StatCard
    label="Detections Today"
    value={$stats.today.toString()}
    icon="today"
    trend={`Peak: ${$stats.peak_time}`}
    variant="secondary"
  />
  <StatCard
    label="This Month"
    value={$stats.this_month.toLocaleString()}
    icon="calendar_month"
    trend={new Date().toLocaleDateString("default", {
      month: "short",
      year: "numeric",
    })}
    variant="tertiary"
  />
</section>

<!-- Middle Section: Chart & Gallery -->
<section class="grid grid-cols-1 lg:grid-cols-3 gap-8 my-10">
  <div
    class="lg:col-span-2 bg-surface-container-lowest p-8 rounded-xl shadow-md flex flex-col"
  >
    <div class="flex items-center justify-between mb-8">
      <div>
        <h2 class="text-xl font-bold text-primary font-headline">
          Confidence History
        </h2>
        <p class="text-on-surface-variant text-sm">
          AI prediction accuracy averages
        </p>
      </div>
      <div class="flex bg-surface-container-high p-1 rounded-full gap-1">
        {#each ["day", "month", "year", "all"] as p}
          <button
            class="px-4 py-1.5 rounded-full text-[10px] font-bold uppercase transition-all {selectedPeriod ===
            p
              ? 'bg-primary text-on-primary shadow-sm'
              : 'text-on-surface-variant hover:bg-surface-container-lowest'}"
            on:click={() => (selectedPeriod = p)}
          >
            {p}
          </button>
        {/each}
      </div>
    </div>

    <ConfidenceChart period={selectedPeriod} />
  </div>

  <!-- Gallery Column -->
  <DetectionGallery />
</section>

<!-- Recent Detections Table -->
<section class="mt-4 mb-20">
  <DetectionTable />
</section>
