<script>
  import { recentDetections, lightboxStore } from '../stores/detections';
  import Badge from './Badge.svelte';

  let currentPage = 1;
  let isExpanded = false;
  $: itemsPerPage = isExpanded ? 10 : 3;
  let sortOrder = 'desc';

  function toggleExpand() {
    isExpanded = !isExpanded;
    currentPage = 1; // Reset to page 1 to maintain view consistency
  }

  function openInLightbox(detection) {
    const filename = detection.image_path ? detection.image_path.split('/').pop() : '';
    const imageObj = { 
      path: detection.image_path ? `/detections/${filename}` : '/placeholder.jpg',
      camera_name: detection.camera_name || 'Sentinel-01',
      modified: new Date(detection.timestamp.replace(' ', 'T')).getTime() / 1000
    };
    
    lightboxStore.set({ 
      isOpen: true, 
      image: imageObj,
      index: 0,
      images: [imageObj]
    });
    document.body.classList.add('lightbox-open');
  }

  $: sortedDetections = [...$recentDetections].sort((a, b) => {
    const timeA = new Date(a.timestamp.replace(' ', 'T')).getTime();
    const timeB = new Date(b.timestamp.replace(' ', 'T')).getTime();
    return sortOrder === 'desc' ? timeB - timeA : timeA - timeB;
  });

  $: totalPages = Math.ceil(sortedDetections.length / itemsPerPage);
  $: paginatedDetections = sortedDetections.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  function toggleSort() {
    sortOrder = sortOrder === 'desc' ? 'asc' : 'desc';
    currentPage = 1;
  }

  function prevPage() {
    if (currentPage > 1) currentPage--;
  }

  function nextPage() {
    if (currentPage < totalPages) currentPage++;
  }

  function goToPage(p) {
    if (p >= 1 && p <= totalPages) currentPage = p;
  }

  function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    // Handle "2024-05-24 14:22:05" format or ISO
    const date = new Date(timestamp.replace(' ', 'T'));
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  $: visiblePages = (() => {
    let pages = [];
    const maxVisible = 3;
    if (totalPages <= 5) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      if (currentPage <= 3) {
        pages = [1, 2, 3, '...', totalPages];
      } else if (currentPage >= totalPages - 2) {
        pages = [1, '...', totalPages - 2, totalPages - 1, totalPages];
      } else {
        pages = [1, '...', currentPage, '...', totalPages];
      }
    }
    return pages;
  })();
</script>

<div class="bg-surface-container-lowest rounded-xl shadow-md overflow-hidden flex flex-col">
  <div class="p-6 border-b border-surface-container-high flex items-center justify-between">
    <h2 class="text-xl font-bold text-primary font-headline">Recent Detections</h2>
    <div class="flex items-center gap-4">
      <button 
        class="flex items-center gap-2 text-sm font-medium text-on-surface-variant hover:text-primary transition-colors"
        on:click={toggleExpand}
      >
        <span class="material-symbols-outlined text-lg">
          {isExpanded ? 'unfold_less' : 'unfold_more'}
        </span>
        {isExpanded ? 'Collapse' : 'Expand'}
      </button>

      <div class="w-px h-6 bg-outline-variant/20"></div>

      <button 
        class="flex items-center gap-2 text-sm font-medium text-on-surface-variant hover:text-primary transition-colors"
        on:click={toggleSort}
      >
        <span class="material-symbols-outlined text-lg">
          {sortOrder === 'desc' ? 'arrow_downward' : 'arrow_upward'}
        </span>
        {sortOrder === 'desc' ? 'Newest First' : 'Oldest First'}
      </button>
    </div>
  </div>
  <div class="overflow-x-auto flex-1">
    <table class="w-full text-left border-collapse table-fixed">
      <thead>
        <tr class="bg-surface-container-low">
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant text-center">Time</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant text-center">Camera</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant text-center">Confidence</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant text-center">Action</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-surface-container">
        {#each paginatedDetections as detection (detection.id || detection.timestamp)}
          <tr class="hover:bg-surface-container-low/50 transition-colors cursor-pointer group">
            <td class="px-6 py-4 text-sm font-medium text-center">{formatTime(detection.timestamp)}</td>
            <td class="px-6 py-4 text-sm font-label text-center">{detection.camera_name || 'Sentinel-01'}</td>
            <td class="px-6 py-4">
              <div class="flex items-center justify-center gap-2">
                <div class="w-16 h-1.5 bg-surface-container rounded-full overflow-hidden">
                  <div class="h-full bg-primary" style="width: {detection.confidence * 100}%"></div>
                </div>
                <span class="text-xs font-bold text-primary">{Math.round(detection.confidence * 100)}%</span>
              </div>
            </td>
            <td class="px-6 py-4 text-center">
              <button 
                class="p-2 rounded-lg bg-primary-fixed/30 text-primary hover:bg-primary-fixed hover:text-primary transition-all active:scale-90 flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-tight mx-auto"
                on:click|stopPropagation={() => openInLightbox(detection)} 
                title="View Capture"
              >
                <span class="material-symbols-outlined text-lg">image</span>
                <span>View</span>
              </button>
            </td>
          </tr>
        {:else}
          <tr>
            <td colspan="4" class="px-6 py-12 text-center text-on-surface-variant italic">
              No recent detections found.
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  
  {#if totalPages > 1}
    <div class="p-6 border-t border-surface-container-high flex flex-col md:flex-row items-center justify-between gap-6 bg-surface-container-lowest/50">
      <p class="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest">
        Showing Page {currentPage} of {totalPages}
      </p>
      
      <div class="flex items-center gap-1.5">
        <!-- First Page -->
        <button 
          on:click={() => goToPage(1)} 
          disabled={currentPage === 1}
          class="w-10 h-10 flex items-center justify-center rounded-xl bg-surface-container-low text-on-surface-variant hover:bg-primary-fixed hover:text-primary disabled:opacity-30 disabled:hover:bg-surface-container-low transition-all"
          title="First Page"
        >
          <span class="material-symbols-outlined text-lg">first_page</span>
        </button>

        <!-- Prev Page -->
        <button 
          on:click={prevPage} 
          disabled={currentPage === 1}
          class="w-10 h-10 flex items-center justify-center rounded-xl bg-surface-container-low text-on-surface-variant hover:bg-primary-fixed hover:text-primary disabled:opacity-30 disabled:hover:bg-surface-container-low transition-all mr-2"
          title="Previous Page"
        >
          <span class="material-symbols-outlined text-lg">chevron_left</span>
        </button>

        <!-- Page Numbers -->
        {#each visiblePages as p}
          {#if p === '...'}
            <span class="px-2 text-on-surface-variant/40">...</span>
          {:else}
            <button 
              on:click={() => goToPage(p)}
              class="w-10 h-10 flex items-center justify-center rounded-xl text-xs font-bold transition-all
              {currentPage === p 
                ? 'bg-primary text-on-primary shadow-lg shadow-primary/20 scale-110' 
                : 'bg-surface-container-low text-on-surface-variant hover:bg-surface-container'}"
            >
              {p}
            </button>
          {/if}
        {/each}

        <!-- Next Page -->
        <button 
          on:click={nextPage} 
          disabled={currentPage === totalPages}
          class="w-10 h-10 flex items-center justify-center rounded-xl bg-surface-container-low text-on-surface-variant hover:bg-primary-fixed hover:text-primary disabled:opacity-30 disabled:hover:bg-surface-container-low transition-all ml-2"
          title="Next Page"
        >
          <span class="material-symbols-outlined text-lg">chevron_right</span>
        </button>

        <!-- Last Page -->
        <button 
          on:click={() => goToPage(totalPages)} 
          disabled={currentPage === totalPages}
          class="w-10 h-10 flex items-center justify-center rounded-xl bg-surface-container-low text-on-surface-variant hover:bg-primary-fixed hover:text-primary disabled:opacity-30 disabled:hover:bg-surface-container-low transition-all"
          title="Last Page"
        >
          <span class="material-symbols-outlined text-lg">last_page</span>
        </button>
      </div>
    </div>
  {/if}
</div>
