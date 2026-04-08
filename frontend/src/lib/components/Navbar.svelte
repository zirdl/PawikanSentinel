<script>
  import { onMount } from 'svelte';
  import { logout } from '../stores/auth';
  export let username = "Project Manager";
  export let sector = "Palawan Sector";

  async function fetchUser() {
    try {
      const res = await fetch('/api/auth/me', { credentials: 'include' });
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

<header class="fixed top-0 left-0 right-0 z-40 bg-white/80 backdrop-blur-lg border-b border-outline-variant/15 shadow-sm">
  <div class="flex items-center justify-between h-16 px-4 lg:px-8">
    <!-- Left: logo -->
    <div class="flex items-center gap-3">
      <a href="/" class="flex items-center gap-2.5">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-sm">
          <span class="material-symbols-outlined text-on-primary text-base">eco</span>
        </div>
        <span class="text-xl font-extrabold text-on-surface tracking-tight font-headline">Pawikan Sentinel</span>
      </a>
    </div>

    <!-- Right: user menu -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 p-1.5 rounded-xl hover:bg-surface-container transition-colors cursor-pointer">
        <div class="w-9 h-9 rounded-full bg-primary-fixed flex items-center justify-center">
          <span class="material-symbols-outlined text-primary text-sm">person</span>
        </div>
        <div class="hidden sm:block text-left">
          <p class="text-xs font-semibold text-on-surface font-headline leading-tight capitalize">{username}</p>
          <p class="text-[10px] text-on-surface-variant uppercase tracking-widest">{sector}</p>
        </div>
      </div>
      
      <div class="flex items-center gap-2 border-l border-outline-variant/20 pl-4">
        <button class="p-2 rounded-full hover:bg-surface-container transition-colors relative group">
          <span class="material-symbols-outlined text-on-surface-variant">notifications</span>
          <span class="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full border-2 border-white"></span>
        </button>
        <a href="/settings" class="p-2 rounded-full hover:bg-surface-container transition-colors">
          <span class="material-symbols-outlined text-on-surface-variant">settings</span>
        </a>
        <button on:click={logout} class="p-2 rounded-full hover:bg-error/10 transition-colors text-on-surface-variant hover:text-error">
          <span class="material-symbols-outlined">logout</span>
        </button>
      </div>
    </div>
  </div>
</header>
