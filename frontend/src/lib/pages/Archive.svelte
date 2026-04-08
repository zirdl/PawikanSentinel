<script>
  import { onMount } from 'svelte';
  import { galleryImages, fetchInitialData } from '../stores/detections';
  import Lightbox from '../components/Lightbox.svelte';

  let selectedIndex = -1;
  let isLightboxOpen = false;
  
  let currentPage = 1;
  const itemsPerPage = 20;
  let sortOrder = 'desc'; // desc = newest first

  $: sortedGallery = [...$galleryImages].sort((a, b) => {
    return sortOrder === 'desc' ? b.modified - a.modified : a.modified - b.modified;
  });
  
  $: totalPages = Math.max(1, Math.ceil(sortedGallery.length / itemsPerPage));
  $: paginatedGallery = sortedGallery.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  function openLightbox(index) {
    selectedIndex = index;
    isLightboxOpen = true;
    document.body.classList.add('lightbox-open');
  }

  function closeLightbox() {
    isLightboxOpen = false;
    selectedIndex = -1;
    document.body.classList.remove('lightbox-open');
  }

  function handlePrev() {
    if (selectedIndex > 0) selectedIndex--;
  }

  function handleNext() {
    if (selectedIndex < sortedGallery.length - 1) selectedIndex++;
  }

  function toggleSort() {
    sortOrder = sortOrder === 'desc' ? 'asc' : 'desc';
    currentPage = 1; // reset to first page on sort
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

  $: visiblePages = (() => {
    let pages = [];
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

  onMount(() => {
    fetchInitialData();
  });
</script>

<header class="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
  <div>
    <span class="text-secondary font-label text-xs font-semibold uppercase tracking-widest mb-2 block">Detection Repository</span>
    <h1 class="text-4xl font-extrabold text-on-surface tracking-tight font-headline">Surveillance Archive</h1>
    <p class="text-on-surface-variant mt-2 max-w-xl text-sm">Full chronological record of AI detections with high-resolution captures for conservation research.</p>
  </div>
  <button 
    class="flex items-center gap-3 bg-surface-container-low px-4 py-2 rounded-xl border border-outline-variant/15 hover:bg-surface-container transition-colors shadow-sm"
    on:click={toggleSort}
  >
    <span class="material-symbols-outlined text-primary">
      {sortOrder === 'desc' ? 'arrow_downward' : 'arrow_upward'}
    </span>
    <span class="text-xs font-bold text-on-surface-variant uppercase tracking-widest">
      {sortOrder === 'desc' ? 'Date Descending' : 'Date Ascending'}
    </span>
  </button>
</header>

{#if sortedGallery.length > 0}
  <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
    {#each paginatedGallery as img, i}
      <button 
        class="aspect-square bg-surface-container-high rounded-xl overflow-hidden relative group transition-all hover:shadow-xl active:scale-95 border border-outline-variant/5 shadow-sm"
        on:click={() => openLightbox((currentPage - 1) * itemsPerPage + i)}
      >
        <img src={img.path} alt={img.camera_name} class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110" />
        <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-4">
          <p class="text-[10px] text-white font-bold uppercase tracking-widest mb-1">{img.camera_name}</p>
          <p class="text-[8px] text-white/60 font-medium">{new Date(img.modified * 1000).toLocaleString()}</p>
        </div>
        <div class="absolute top-2 right-2 flex gap-1">
           <span class="material-symbols-outlined text-white text-lg opacity-0 group-hover:opacity-100 transition-all drop-shadow-md">zoom_in</span>
        </div>
      </button>
    {/each}
  </div>

  {#if totalPages > 1}
    <div class="mt-12 flex flex-col md:flex-row items-center justify-between bg-surface-container-lowest p-6 rounded-2xl border border-outline-variant/10 shadow-md gap-6">
      <p class="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest">Showing Page {currentPage} of {totalPages}</p>
      
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
{:else}
  <div class="flex flex-col items-center justify-center h-96 text-on-surface-variant">
    <span class="material-symbols-outlined text-6xl mb-4 opacity-20">inventory_2</span>
    <p class="font-bold uppercase tracking-widest text-sm italic">Archive is empty</p>
    <p class="text-xs mt-1">Detections will appear here as they are processed by the sentinels.</p>
  </div>
{/if}

<Lightbox 
  isOpen={isLightboxOpen} 
  image={sortedGallery[selectedIndex]} 
  hasPrev={selectedIndex > 0}
  hasNext={selectedIndex < sortedGallery.length - 1}
  on:close={closeLightbox}
  on:prev={handlePrev}
  on:next={handleNext}
/>

