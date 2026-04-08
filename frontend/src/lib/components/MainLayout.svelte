<script>
  import { Router, Route, useLocation } from "svelte-routing";
  import Navbar from "./Navbar.svelte";
  import Sidebar from "./Sidebar.svelte";
  import { logout } from "../stores/auth";

  const location = useLocation();
</script>

<div class="min-h-screen bg-background text-on-surface antialiased">
  <Navbar />
  
  <div class="flex">
    <Sidebar currentPath={$location.pathname} />
    
    <main class="flex-1 lg:pl-64 pt-16 min-h-screen">
      <div class="max-w-7xl mx-auto">
        {#key $location.pathname}
          <slot />
        {/key}
      </div>
    </main>
  </div>

  <footer class="lg:pl-64 p-6 border-t border-outline-variant/10 bg-surface-container-lowest/50">
    <div class="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-[10px] text-on-surface-variant/60 uppercase tracking-widest font-bold">
      <span>&copy; 2026 Pawikan Sentinel</span>
      <div class="flex gap-6">
        <span>Real-time Conservation Technology</span>
        <button on:click={logout} class="hover:text-error transition-colors">Sign Out</button>
      </div>
    </div>
  </footer>
</div>
