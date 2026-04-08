<script>
  import { onMount } from "svelte";
  import { logout } from "../stores/auth";
  export let username = "Project Manager";
  export let sector = "CURMA, San Juan";

  async function fetchUser() {
    try {
      const res = await fetch("/api/auth/me", { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        username = data.username;
      }
    } catch (e) {
      console.error("Failed to fetch user:", e);
    }
  }

  onMount(fetchUser);
</script>

<header
  class="fixed top-0 left-0 right-0 z-40 bg-primary shadow-sm"
>
  <div class="flex items-center justify-between h-16 px-4 lg:px-8">
    <!-- Left: logo -->
    <div class="flex items-center gap-3">
      <a href="/" class="flex items-center gap-2.5">
        <div
          class="w-10 h-10 rounded-xl bg-white flex items-center justify-center shadow-md overflow-hidden"
        >
          <img 
            src="/logo.svg" 
            alt="Pawikan Sentinel" 
            class="w-8 h-8 drop-shadow-[0_2px_2px_rgba(0,0,0,0.15)]"
          />
        </div>
        <span
          class="text-xl font-extrabold text-on-primary tracking-tight font-headline"
          >Pawikan Sentinel</span
        >
      </a>
    </div>

    <!-- Right: user menu -->
    <div class="flex items-center gap-4">
      <div
        class="flex items-center gap-2 p-1.5 rounded-xl hover:bg-white/10 transition-colors cursor-pointer"
      >
        <div
          class="w-9 h-9 rounded-full bg-white/20 flex items-center justify-center"
        >
          <span class="material-symbols-outlined text-white text-sm"
            >person</span
          >
        </div>
        <div class="hidden sm:block text-left">
          <p
            class="text-xs font-semibold text-on-primary font-headline leading-tight capitalize"
          >
            {username}
          </p>
          <p
            class="text-[10px] text-on-primary/70 uppercase tracking-widest"
          >
            {sector}
          </p>
        </div>
      </div>

      <div
        class="flex items-center gap-2 border-l border-white/20 pl-4"
      >
        <button
          on:click={logout}
          class="w-10 h-10 flex items-center justify-center rounded-full hover:bg-error/20 transition-colors text-white hover:text-error-container"
        >
          <span class="material-symbols-outlined">logout</span>
        </button>
      </div>
    </div>
  </div>
</header>
