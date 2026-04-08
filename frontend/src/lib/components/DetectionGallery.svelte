<script>
  import { onMount } from 'svelte';
  import { navigate } from 'svelte-routing';
  import { galleryImages, lightboxStore } from '../stores/detections';
  import Lightbox from './Lightbox.svelte';

  function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function openLightbox(index) {
    lightboxStore.set({
      isOpen: true,
      index: index,
      image: $galleryImages[index],
      images: $galleryImages
    });
    document.body.classList.add('lightbox-open');
  }

  function closeLightbox() {
    lightboxStore.update(s => ({ ...s, isOpen: false }));
    document.body.classList.remove('lightbox-open');
  }

  function handlePrev() {
    if ($lightboxStore.index > 0) {
      const newIndex = $lightboxStore.index - 1;
      lightboxStore.update(s => ({
        ...s,
        index: newIndex,
        image: s.images[newIndex]
      }));
    }
  }

  function handleNext() {
    if ($lightboxStore.index < $lightboxStore.images.length - 1) {
      const newIndex = $lightboxStore.index + 1;
      lightboxStore.update(s => ({
        ...s,
        index: newIndex,
        image: s.images[newIndex]
      }));
    }
  }
</script>

<div class="bg-surface-container-high/50 p-6 rounded-xl border border-outline-variant/10 shadow-md">
  <div class="flex items-center justify-between mb-4">
    <h2 class="text-xl font-bold text-primary font-headline">Recent Captures</h2>
    <span class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest bg-surface-container px-2 py-1 rounded">Live Feed</span>
  </div>
  
  {#if $galleryImages.length === 0}
    <div class="flex items-center justify-center h-48 italic text-on-surface-variant">
      Waiting for detections...
    </div>
  {:else}
    <div class="grid grid-cols-2 gap-3">
      {#each $galleryImages.slice(0, 4) as img, i}
        <button 
          class="aspect-square rounded-lg overflow-hidden relative group cursor-pointer shadow-sm border border-outline-variant/5" 
          on:click={() => openLightbox(i)}
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
        </button>
      {/each}
    </div>
    
    <button 
      class="w-full mt-6 py-3 border border-primary/30 text-primary rounded-xl font-bold text-xs uppercase tracking-widest hover:bg-primary hover:text-on-primary transition-all active:scale-95 shadow-sm"
      on:click={() => navigate('/archive')}
    >
      View Full Archive
    </button>
    <Lightbox 
      isOpen={$lightboxStore.isOpen} 
      image={$lightboxStore.image} 
      hasPrev={$lightboxStore.index > 0}
      hasNext={$lightboxStore.index < $lightboxStore.images.length - 1}
      on:close={closeLightbox}
      on:prev={handlePrev}
      on:next={handleNext}
    />
  {/if}
</div>
