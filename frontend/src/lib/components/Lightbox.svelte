<script>
  import { createEventDispatcher } from 'svelte';
  import { fade, scale } from 'svelte/transition';

  export let isOpen = false;
  export let hasPrev = false;
  export let hasNext = false;
  export let image = { path: '', camera_name: '', modified: 0 };

  const dispatch = createEventDispatcher();

  let zoom = 1;
  let isDragging = false;
  let offset = { x: 0, y: 0 };
  let startPos = { x: 0, y: 0 };

  function handleClose() {
    zoom = 1;
    offset = { x: 0, y: 0 };
    dispatch('close');
  }

  function handleZoomIn() {
    zoom = Math.min(zoom + 0.5, 4);
  }

  function handleZoomOut() {
    zoom = Math.max(zoom - 0.5, 1);
    if (zoom === 1) offset = { x: 0, y: 0 };
  }

  function handleReset() {
    zoom = 1;
    offset = { x: 0, y: 0 };
  }

  function handleMouseDown(e) {
    if (zoom > 1) {
      isDragging = true;
      startPos = { x: e.clientX - offset.x, y: e.clientY - offset.y };
    }
  }

  function handleMouseMove(e) {
    if (isDragging) {
      offset = { x: e.clientX - startPos.x, y: e.clientY - startPos.y };
    }
  }

  function handleMouseUp() {
    isDragging = false;
  }

  function handlePrev(e) {
    e.stopPropagation();
    handleReset();
    dispatch('prev');
  }

  function handleNext(e) {
    e.stopPropagation();
    handleReset();
    dispatch('next');
  }

  function handleKeydown(e) {
    if (e.key === 'Escape') handleClose();
    if (e.key === 'ArrowLeft' && hasPrev) handlePrev(e);
    if (e.key === 'ArrowRight' && hasNext) handleNext(e);
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen && image}
  <div 
    class="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-black/95 backdrop-blur-md"
    on:mousemove={handleMouseMove}
    on:mouseup={handleMouseUp}
    on:mouseleave={handleMouseUp}
    role="dialog"
    aria-modal="true"
    tabindex="-1"
    transition:fade={{ duration: 200 }}
  >
    <!-- Top Bar -->
    <div class="absolute top-0 left-0 right-0 p-6 flex items-center justify-between z-10 bg-gradient-to-b from-black/60 to-transparent">
      <div class="text-white">
        <h3 class="text-lg font-bold font-headline">{image?.camera_name || 'Camera Capture'}</h3>
        <p class="text-[10px] text-white/60 uppercase tracking-widest font-bold">
          {image ? new Date(image.modified * 1000).toLocaleString() : ''}
        </p>
      </div>
      <div class="flex items-center gap-4">
        <div class="flex items-center bg-white/10 rounded-full px-2 py-1 gap-1 border border-white/10">
          <button class="w-9 h-9 flex items-center justify-center text-white hover:bg-white/10 rounded-full transition-colors" on:click={handleZoomOut}>
            <span class="material-symbols-outlined text-sm">zoom_out</span>
          </button>
          <span class="text-[10px] font-mono font-bold w-12 text-center text-white/80">{Math.round(zoom * 100)}%</span>
          <button class="w-9 h-9 flex items-center justify-center text-white hover:bg-white/10 rounded-full transition-colors" on:click={handleZoomIn}>
            <span class="material-symbols-outlined text-sm">zoom_in</span>
          </button>
          <button class="w-9 h-9 flex items-center justify-center text-white/40 hover:text-white transition-colors" on:click={handleReset}>
            <span class="material-symbols-outlined text-sm">restart_alt</span>
          </button>
        </div>
        <button class="w-11 h-11 flex items-center justify-center text-white hover:bg-white/20 rounded-full transition-all active:scale-90" on:click={handleClose}>
          <span class="material-symbols-outlined">close</span>
        </button>
      </div>
    </div>

    <!-- Navigation Arrows -->
    {#if hasPrev}
      <button 
        class="absolute left-6 top-1/2 -translate-y-1/2 p-4 text-white hover:bg-white/10 rounded-full transition-all active:scale-90 z-20 group" 
        on:click={handlePrev}
      >
        <span class="material-symbols-outlined text-4xl opacity-40 group-hover:opacity-100">chevron_left</span>
      </button>
    {/if}

    {#if hasNext}
      <button 
        class="absolute right-6 top-1/2 -translate-y-1/2 p-4 text-white hover:bg-white/10 rounded-full transition-all active:scale-90 z-20 group" 
        on:click={handleNext}
      >
        <span class="material-symbols-outlined text-4xl opacity-40 group-hover:opacity-100">chevron_right</span>
      </button>
    {/if}

    <!-- Image Container -->
    <div 
      class="w-full h-full flex items-center justify-center overflow-hidden cursor-move"
      on:mousedown={handleMouseDown}
      on:click|self={handleClose}
      on:keydown|self={(e) => e.key === 'Enter' && handleClose()}
      role="button"
      tabindex="0"
    >
      <div 
        class="transition-transform duration-200 ease-out flex items-center justify-center translate-x-0"
        style="transform: translate({offset.x}px, {offset.y}px) scale({zoom})"
      >
        <img 
          src={image?.path} 
          alt={image?.camera_name} 
          class="max-w-full max-h-[85vh] object-contain shadow-2xl select-none"
          draggable="false"
          transition:scale={{ duration: 300, start: 0.95 }}
        />
      </div>
    </div>

    <!-- Bottom Info (Optional/Overlay) -->
    <div class="absolute bottom-6 left-1/2 -translate-x-1/2 bg-white/10 backdrop-blur-xl px-6 py-3 rounded-full border border-white/10 text-white/80 text-xs font-medium flex items-center gap-6">
       <span class="flex items-center gap-2">
         <span class="material-symbols-outlined text-sm">visibility</span>
         Full Detail View
       </span>
       {#if image}
         <a href={image.path} download class="text-primary font-bold hover:underline underline-offset-4 flex items-center gap-1">
           <span class="material-symbols-outlined text-sm">download</span>
           Download High-Res
         </a>
       {/if}
    </div>
  </div>
{/if}

<style>
  :global(body.lightbox-open) {
    overflow: hidden;
  }
</style>
