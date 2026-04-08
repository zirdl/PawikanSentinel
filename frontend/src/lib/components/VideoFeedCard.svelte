<script>
  import Badge from "./Badge.svelte";
  export let camera = {};
  export let onEdit = (cam) => {};
  export let onDelete = (id) => {};

  // Mock data for visual fidelity with Stitch design
  const placeholderImages = [
    "https://lh3.googleusercontent.com/aida-public/AB6AXuDjPNRPR4MCdwJYFvQRUvYuHJ0kRNKjBdyrsaZhdmEMNpCtxwQWkiYpb4odTnNWlYfp82hd31qPiOl1H4ix5nvsat4Q-28rE2vHB28WDY1UthAmgCk7cfHtwM8dsu44I96g16-TiO3fhhuxpMVdSGpULB7RS5aQxce9ToxJ0QVdxtQ3MeG0StUSIpFxGR6_P3P-tHSQHO5TDwDTtvN4EJ9_--g0X9HcvPN7kgHiKn5eg3Hnz-AjqD1ZHq0bRG7zEM9ELhWVMt6TsEjV",
    "https://lh3.googleusercontent.com/aida-public/AB6AXuCaS5FPIOeEryk7e62itPMpFKAUrK_5xWKz2EbeDyP9Xf00troJqY1R9bTtd_v2U7DHMKDd6hk6QwFUnT51cx3yc-ljWu-cnjF5aHB9dZYOSUt9789ew_mylVigsHk9SnEH9Gzxp4VFggyaxMLpn5TzNhG9M6kwOjeMITFgLMT7EwlzSW_BxeX9uISjL9h-jippsQA6sHASb6HzmN9Z5UAQ4JfgHRbSZeMm1G40_F1cH1yfL66hD2VpLfBGGP_0RsPy4yGF69VvEqdG",
  ];

  const img =
    placeholderImages[camera.id % placeholderImages.length] ||
    placeholderImages[0];
</script>

<div
  class="bg-surface-container-lowest rounded-xl overflow-hidden shadow-sm transition-transform hover:-translate-y-1 group"
>
  <!-- Video Container -->
  <div class="relative aspect-video bg-black">
    {#if camera.active}
      <img
        class="w-full h-full object-cover opacity-80"
        src={img}
        alt={camera.name}
      />
      <div
        class="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"
      ></div>
      <div class="absolute top-4 left-4 flex gap-2">
        <Badge text="Active" variant="success" pulse={true} />
        <span
          class="px-3 py-1 bg-black/40 backdrop-blur-md text-white text-[10px] font-bold uppercase rounded-full"
          >4K RTSP</span
        >
      </div>
      <div
        class="absolute bottom-4 right-4 text-white/70 font-label text-[10px] uppercase tracking-widest"
      >
        {camera.rtsp_url.split("@").pop()}
      </div>
      <div
        class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <button
          class="w-12 h-12 rounded-full bg-white/20 backdrop-blur-xl flex items-center justify-center text-white border border-white/30 active:scale-90 transition-transform"
        >
          <span class="material-symbols-outlined filled">play_arrow</span>
        </button>
      </div>
    {:else}
      <div
        class="absolute inset-0 flex flex-col items-center justify-center text-on-surface-variant/40 bg-surface-dim"
      >
        <span class="material-symbols-outlined text-5xl mb-2">videocam_off</span
        >
        <span class="font-label text-[10px] font-bold uppercase tracking-widest"
          >Signal Lost / Inactive</span
        >
      </div>
      <div class="absolute top-4 left-4">
        <Badge text="Inactive" variant="warning" />
      </div>
    {/if}
  </div>

  <!-- Content -->
  <div class="p-6">
    <div class="flex justify-between items-start mb-4">
      <div>
        <h3 class="font-headline font-bold text-lg text-on-surface">
          {camera.name}
        </h3>
      </div>
      {#if camera.active}
        <div
          class="bg-tertiary-fixed text-on-tertiary-fixed px-2 py-1 rounded-lg text-xs font-bold"
        >
          98% Confidence
        </div>
      {/if}
    </div>

    <div class="grid grid-cols-2 gap-3 mb-3">
      <button
        class="flex items-center justify-center gap-2 bg-surface-container-highest text-on-surface font-semibold py-2.5 rounded-lg hover:bg-surface-dim transition-colors text-sm active:scale-95"
      >
        <span class="material-symbols-outlined text-lg"
          >settings_input_component</span
        >
        VLC
      </button>
      <button
        class="flex items-center justify-center gap-2 bg-primary text-on-primary font-bold py-2.5 rounded-lg hover:opacity-90 transition-all text-sm active:scale-95"
      >
        <span class="material-symbols-outlined text-lg">visibility</span>
        View
      </button>
    </div>

    <!-- Admin Controls -->
    <div
      class="flex items-center justify-between pt-3 border-t border-surface-container-highest mt-3 transition-opacity"
    >
      <span
        class="text-[10px] text-on-surface-variant italic font-medium tracking-widest uppercase"
        >Configure</span
      >
      <div class="flex gap-1">
        <button
          class="p-1.5 text-on-surface-variant hover:text-primary rounded-lg hover:bg-surface-container transition-all"
          on:click={() => onEdit(camera)}
        >
          <span class="material-symbols-outlined text-lg">edit</span>
        </button>
        <button
          class="p-1.5 text-on-surface-variant hover:text-error rounded-lg hover:bg-surface-container transition-all"
          on:click={() => onDelete(camera.id)}
        >
          <span class="material-symbols-outlined text-lg">delete</span>
        </button>
      </div>
    </div>
  </div>
</div>
