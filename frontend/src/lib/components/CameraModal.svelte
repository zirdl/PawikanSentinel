<script>
  import Modal from './Modal.svelte';
  import { addCamera, updateCamera } from '../stores/cameras';

  export let isOpen = false;
  export let camera = { name: '', rtsp_url: '', active: true };
  export let cameraId = null;
  export let onSave = () => {};

  let isSaving = false;

  async function handleSave() {
    isSaving = true;
    try {
      let success;
      if (cameraId) {
        success = await updateCamera(cameraId, camera);
      } else {
        success = await addCamera(camera);
      }
      
      if (success) {
        isOpen = false;
        onSave();
      }
    } finally {
      isSaving = false;
    }
  }
</script>

<Modal title={cameraId ? "Edit Stream" : "Add Stream"} {isOpen} on:close={() => isOpen = false}>
  <div class="space-y-4">
    <div class="space-y-1">
      <label for="cam-name" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Target Name</label>
      <input id="cam-name" type="text" class="input-surface w-full" placeholder="Beach Alpha Cam" bind:value={camera.name} />
    </div>
    <div class="space-y-1">
      <label for="cam-rtsp" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">RTSP Endpoint</label>
      <input id="cam-rtsp" type="text" class="input-surface w-full font-mono text-xs" placeholder="rtsp://admin:pass@192.168.1.100:554/ch1" bind:value={camera.rtsp_url} />
    </div>
    <div class="flex items-center gap-2 py-2">
      <input type="checkbox" id="cam-active" class="w-4 h-4 rounded accent-primary" bind:checked={camera.active} />
      <label for="cam-active" class="text-sm font-medium">Currently Active</label>
    </div>
  </div>
  <div slot="footer" class="flex gap-3">
    <button class="btn-secondary text-xs" on:click={() => isOpen = false} disabled={isSaving}>Discard</button>
    <button class="btn-gradient-primary text-xs px-6 flex items-center gap-2" on:click={handleSave} disabled={isSaving}>
      {#if isSaving}
        <span class="material-symbols-outlined text-sm animate-spin">progress_activity</span>
        Saving...
      {:else}
        {cameraId ? "Update Feed" : "Initialize Feed"}
      {/if}
    </button>
  </div>
</Modal>
