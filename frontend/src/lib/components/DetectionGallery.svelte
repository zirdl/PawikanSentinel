<script>
  import { onMount } from 'svelte';
  import Modal from './Modal.svelte';

  let images = [];
  let loading = true;
  let selectedImage = null;
  let isModalOpen = false;

  async function fetchGallery() {
    try {
      const res = await fetch('/api/detections/gallery', { credentials: 'include' });
      if (res.ok) {
        images = await res.json();
      }
    } catch (error) {
      console.error('Error fetching gallery:', error);
    } finally {
      loading = false;
    }
  }

  function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function openModal(img) {
    selectedImage = img;
    isModalOpen = true;
  }

  function closeModal() {
    isModalOpen = false;
    selectedImage = null;
  }

  onMount(fetchGallery);
</script>

<div class="bg-surface-container-high/50 p-6 rounded-xl border border-outline-variant/10">
  <h2 class="text-xl font-bold text-primary font-headline mb-4">Detection Gallery</h2>
  
  {#if loading}
    <div class="flex items-center justify-center h-48 italic text-on-surface-variant">
      Loading gallery...
    </div>
  {:else if images.length > 0}
    <div class="grid grid-cols-2 gap-3">
      {#each images.slice(0, 4) as img}
        <div 
          class="aspect-square rounded-lg overflow-hidden relative group cursor-pointer shadow-sm" 
          on:click={() => openModal(img)}
          on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && openModal(img)}
          role="button"
          tabindex="0"
          aria-label="View detection detail for {img.camera_name}"
        >
          <img class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
               src={img.path} 
               alt={img.camera_name} />
          <div class="absolute inset-0 bg-primary/20 opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <div class="absolute bottom-1 right-1 bg-white/90 px-1.5 py-0.5 rounded text-[8px] font-bold shadow-sm">
            {formatTime(img.modified)}
          </div>
          <div class="absolute top-1 left-1 bg-primary/80 backdrop-blur-sm text-white px-1.5 py-0.5 rounded text-[8px] font-medium opacity-0 group-hover:opacity-100 transition-opacity">
            {img.camera_name}
          </div>
        </div>
      {/each}
    </div>
    <button class="w-full mt-6 py-3 border border-primary text-primary rounded-xl font-semibold text-sm hover:bg-primary hover:text-on-primary transition-all active:scale-95 shadow-sm">
      View Full Archive
    </button>
  {:else}
    <div class="flex flex-col items-center justify-center h-48 text-on-surface-variant/40">
      <span class="material-symbols-outlined text-4xl mb-2">image_not_supported</span>
      <p class="text-xs uppercase tracking-widest font-bold">No captures yet</p>
    </div>
  {/if}
</div>

{#if selectedImage}
  <Modal title="Detection Detail" isOpen={isModalOpen} on:close={closeModal}>
    <div class="flex flex-col gap-4">
      <div class="rounded-xl overflow-hidden bg-black flex items-center justify-center min-h-[300px]">
        <img src={selectedImage.path} alt={selectedImage.camera_name} class="max-w-full max-h-[60vh] object-contain" />
      </div>
      <div class="flex justify-between items-center py-2 border-t border-outline-variant/10">
        <div class="flex flex-col">
          <span class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Source Camera</span>
          <span class="font-bold text-primary">{selectedImage.camera_name}</span>
        </div>
        <div class="flex flex-col items-end">
          <span class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Captured At</span>
          <span class="text-on-surface font-medium">{new Date(selectedImage.modified * 1000).toLocaleString()}</span>
        </div>
      </div>
    </div>
    <div slot="footer" class="flex gap-3">
      <button on:click={closeModal} class="btn-secondary text-xs px-6">Close</button>
      <a href={selectedImage.path} download class="btn-gradient-primary text-xs px-6 py-2.5 flex items-center gap-2 rounded-xl transition-all hover:shadow-lg">
        <span class="material-symbols-outlined text-sm">download</span>
        Download
      </a>
    </div>
  </Modal>
{/if}
