<script>
  import { onMount } from "svelte";
  import {
    cameras,
    loading,
    fetchCameras,
    deleteCamera,
  } from "../stores/cameras";
  import VideoFeedCard from "../components/VideoFeedCard.svelte";
  import CameraModal from "../components/CameraModal.svelte";

  let isModalOpen = false;
  let cameraForm = { name: "", rtsp_url: "", active: true };
  let editingCameraId = null;

  function openModal(cam = null) {
    if (cam && cam.id) {
      cameraForm = { ...cam };
      editingCameraId = cam.id;
    } else {
      cameraForm = { name: "", rtsp_url: "", active: true };
      editingCameraId = null;
    }
    isModalOpen = true;
  }

  async function removeCamera(id) {
    if (confirm("Are you sure you want to delete this camera feed?")) {
      await deleteCamera(id);
    }
  }

  onMount(fetchCameras);
</script>

<header
  class="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6"
>
  <div>
    <span
      class="text-secondary font-label text-xs font-semibold uppercase tracking-widest mb-2 block"
      >Surveillance Network</span
    >
    <h1
      class="text-4xl font-extrabold text-on-surface tracking-tight font-headline"
    >
      Active Live Streams
    </h1>
    <p class="text-on-surface-variant mt-2 max-w-xl text-sm">
      Real-time monitoring of critical nesting sites across San Juan and
      Bacnotan, La Union.
    </p>
  </div>
  <button
    on:click={() => openModal()}
    class="btn-gradient-primary flex items-center gap-2"
  >
    <span class="material-symbols-outlined">add_a_photo</span>
    Configure Stream
  </button>
</header>

{#if $loading && $cameras.length === 0}
  <div
    class="flex items-center justify-center h-64 italic text-on-surface-variant"
  >
    Connecting to sentinel network...
  </div>
{:else}
  <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
    {#each $cameras as camera}
      <VideoFeedCard {camera} onEdit={openModal} onDelete={removeCamera} />
    {/each}

    <!-- Add New Feed Card -->
    <button
      on:click={() => openModal()}
      class="border-2 border-dashed border-outline-variant rounded-xl flex flex-col items-center justify-center p-8 group hover:border-primary transition-all min-h-[340px] bg-surface-container-low/30"
    >
      <div
        class="w-16 h-16 rounded-full bg-surface-container flex items-center justify-center text-on-surface-variant group-hover:bg-primary-fixed group-hover:text-primary transition-all mb-4"
      >
        <span class="material-symbols-outlined text-3xl">add</span>
      </div>
      <span class="font-headline font-bold text-lg text-on-surface"
        >Add Camera Feed</span
      >
      <span
        class="font-body text-[10px] text-on-surface-variant mt-1 uppercase tracking-widest font-bold"
        >Connect new RTSP/ONVIF stream</span
      >
    </button>
  </div>
{/if}

<CameraModal
  bind:isOpen={isModalOpen}
  bind:camera={cameraForm}
  bind:cameraId={editingCameraId}
  onSave={fetchCameras}
/>
