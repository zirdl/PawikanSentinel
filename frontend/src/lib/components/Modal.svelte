<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, scale } from 'svelte/transition';

  export let title = '';
  export let isOpen = false;

  const dispatch = createEventDispatcher();

  function close() {
    dispatch('close');
  }

  function handleKeydown(e) {
    if (e.key === 'Escape') close();
  }
</script>

{#if isOpen}
  <div 
    class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
    transition:fade={{ duration: 200 }}
    on:click|self={close}
    on:keydown={handleKeydown}
    role="dialog"
    aria-modal="true"
    tabindex="-1"
  >
    <div 
      class="bg-surface-container-lowest border border-outline-variant/30 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden flex flex-col"
      transition:scale={{ duration: 250, start: 0.95 }}
    >
      <header class="px-6 py-4 border-b border-outline-variant/15 flex items-center justify-between bg-surface-container-low/50">
        <h3 class="text-xl font-bold font-headline text-on-surface">{title}</h3>
        <button 
          on:click={close}
          class="p-2 hover:bg-surface-container-high rounded-full text-on-surface-variant transition-colors"
        >
          <span class="material-symbols-outlined text-xl">close</span>
        </button>
      </header>

      <div class="p-6 overflow-y-auto max-h-[70vh]">
        <slot />
      </div>

      <footer class="px-6 py-4 bg-surface-container-low/50 border-t border-outline-variant/15 flex justify-end gap-3">
        <slot name="footer">
          <button on:click={close} class="btn-secondary text-sm">Cancel</button>
        </slot>
      </footer>
    </div>
  </div>
{/if}
