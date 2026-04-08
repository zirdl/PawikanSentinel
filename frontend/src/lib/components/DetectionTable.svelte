<script>
  import { recentDetections } from '../stores/detections';
  import Badge from './Badge.svelte';

  function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    // Handle "2024-05-24 14:22:05" format or ISO
    const date = new Date(timestamp.replace(' ', 'T'));
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
</script>

<div class="bg-surface-container-lowest rounded-xl shadow-sm overflow-hidden">
  <div class="p-6 border-b border-surface-container-high flex items-center justify-between">
    <h2 class="text-xl font-bold text-primary font-headline">Recent Detections</h2>
    <button class="flex items-center gap-2 text-sm font-medium text-on-surface-variant hover:text-primary transition-colors">
      <span class="material-symbols-outlined text-lg">filter_list</span>
      Filter Feed
    </button>
  </div>
  <div class="overflow-x-auto">
    <table class="w-full text-left border-collapse">
      <thead>
        <tr class="bg-surface-container-low">
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Time</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Camera</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Species</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Confidence</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Status</th>
          <th class="px-6 py-4"></th>
        </tr>
      </thead>
      <tbody class="divide-y divide-surface-container">
        {#each $recentDetections as detection (detection.id)}
          <tr class="hover:bg-surface-container-low/50 transition-colors cursor-pointer group">
            <td class="px-6 py-4 text-sm font-medium">{formatTime(detection.timestamp)}</td>
            <td class="px-6 py-4 text-sm font-label">{detection.camera_name || 'Sentinel-01'}</td>
            <td class="px-6 py-4 text-sm font-semibold">{detection.class || detection._class}</td>
            <td class="px-6 py-4">
              <div class="flex items-center gap-2">
                <div class="w-16 h-1.5 bg-surface-container rounded-full overflow-hidden">
                  <div class="h-full bg-primary" style="width: {detection.confidence * 100}%"></div>
                </div>
                <span class="text-xs font-bold text-primary">{Math.round(detection.confidence * 100)}%</span>
              </div>
            </td>
            <td class="px-6 py-4">
              <Badge text="Verified" variant="primary" />
            </td>
            <td class="px-6 py-4 text-right">
              <span class="material-symbols-outlined text-outline group-hover:text-primary transition-colors">chevron_right</span>
            </td>
          </tr>
        {:else}
          <tr>
            <td colspan="6" class="px-6 py-12 text-center text-on-surface-variant italic">
              No recent detections found.
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
