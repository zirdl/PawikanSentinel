<script>
  import { onMount, createEventDispatcher } from "svelte";
  import { Link } from "svelte-routing";
  import { logout } from "../stores/auth";

  export let currentPath = "/";
  export let isCollapsed = false;

  const navItems = [
    { path: "/", icon: "dashboard", label: "Dashboard" },
    { path: "/cameras", icon: "videocam", label: "Cameras" },
    { path: "/archive", icon: "history", label: "Archive" },
    { path: "/settings", icon: "settings", label: "Settings" },
  ];

  let username = "Project Manager";
  let sector = "CURMA, San Juan";

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

<aside
  class="fixed top-0 left-0 bottom-0 z-30 {isCollapsed ? 'w-[88px]' : 'w-64'} transition-all duration-300 ease-in-out bg-gradient-to-b from-primary to-primary-container text-on-primary rounded-r-[1.3rem] shadow-[8px_0_15px_-3px_rgba(0,0,0,0.1)] overflow-y-auto hidden lg:flex flex-col py-6"
>
  <!-- Header / Brand & Toggle -->
  <div class="{isCollapsed ? 'justify-center flex-col gap-4' : 'px-6 justify-between'} mb-8 flex items-center transition-all duration-300">
    <div class="flex items-center gap-2 overflow-hidden">
      <button 
        on:click={() => isCollapsed = !isCollapsed}
        class="w-10 h-10 rounded-xl {isCollapsed ? 'bg-white/10 hover:bg-white/20' : 'bg-white shadow-md'} flex items-center justify-center shrink-0 transition-colors"
        title="Toggle Sidebar"
      >
        {#if isCollapsed}
          <span class="material-symbols-outlined text-white">menu</span>
        {:else}
          <img 
            src="/logo.svg" 
            alt="Pawikan Sentinel" 
            class="w-8 h-8 drop-shadow-[0_2px_2px_rgba(0,0,0,0.15)]"
          />
        {/if}
      </button>
      
      {#if !isCollapsed}
        <span
          class="text-xl leading-none font-extrabold text-on-primary tracking-tight font-headline opacity-100 transition-opacity duration-300 delay-100"
        >
          Pawikan Sentinel
        </span>
      {/if}
    </div>
    
    {#if !isCollapsed}
      <button 
        on:click={() => isCollapsed = !isCollapsed}
        class="text-white/50 hover:text-white transition-colors"
      >
        <span class="material-symbols-outlined text-sm">keyboard_double_arrow_left</span>
      </button>
    {/if}
  </div>

  <nav class="{isCollapsed ? 'px-4' : 'px-3'} flex-1 space-y-2">
    {#each navItems as item}
      <Link
        to={item.path}
        title={isCollapsed ? item.label : ""}
        class="flex items-center {isCollapsed ? 'justify-center w-12 h-12 rounded-xl mx-auto' : 'gap-3 px-4 py-3 rounded-xl'} text-sm font-medium transition-all duration-200 
               {currentPath === item.path
          ? 'bg-white/20 text-white shadow-sm'
          : 'text-white/70 hover:bg-white/10 hover:text-white'}"
      >
        <span
          class="material-symbols-outlined text-lg {currentPath === item.path
            ? 'filled'
            : ''}"
        >
          {item.icon}
        </span>
        {#if !isCollapsed}
          <span class="whitespace-nowrap">{item.label}</span>
        {/if}
      </Link>
    {/each}
  </nav>

  <!-- Sidebar footer: User Profile & Logout -->
  <div class="px-4 mt-8">
    <div class="{isCollapsed ? 'p-2 flex-col gap-4' : 'p-3 flex-row justify-between'} flex items-center rounded-2xl bg-white/10 border border-white/5 transition-all duration-300">
      
      {#if isCollapsed}
        <div class="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center shrink-0">
          <span class="material-symbols-outlined text-white text-sm">person</span>
        </div>
        <button
          on:click={logout}
          title="Sign out"
          class="w-10 h-10 flex items-center justify-center rounded-full hover:bg-error/20 transition-colors text-white/70 hover:text-error-container shrink-0"
        >
          <span class="material-symbols-outlined text-sm rotate-180">logout</span>
        </button>
      {:else}
        <div class="flex items-center gap-3 overflow-hidden">
          <div
            class="w-9 h-9 rounded-full bg-white/20 flex items-center justify-center shrink-0"
          >
            <span class="material-symbols-outlined text-white text-sm"
              >person</span
            >
          </div>
          <div class="truncate">
            <p
              class="text-xs font-semibold text-white font-headline leading-tight capitalize truncate"
            >
              {username}
            </p>
            <p
              class="text-[9px] text-white/70 uppercase tracking-widest truncate"
            >
              {sector}
            </p>
          </div>
        </div>

        <button
          on:click={logout}
          title="Sign out"
          class="w-9 h-9 flex items-center justify-center rounded-full hover:bg-error/20 transition-colors text-white/70 hover:text-error-container shrink-0"
        >
          <span class="material-symbols-outlined text-sm rotate-180">logout</span>
        </button>
      {/if}
    </div>
  </div>
</aside>
