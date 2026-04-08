<script>
  import { Router, Route, useLocation } from "svelte-routing";
  import Sidebar from "./Sidebar.svelte";
  import { logout } from "../stores/auth";

  const location = useLocation();
  let isCollapsed = false;
</script>

<div class="min-h-screen bg-background text-on-surface antialiased">
  <div class="flex">
    <Sidebar currentPath={$location.pathname} bind:isCollapsed />

    <main
      class="flex-1 {isCollapsed
        ? 'lg:pl-[88px]'
        : 'lg:pl-64'} min-h-screen transition-all duration-300 ease-in-out"
    >
      <div class="max-w-7xl mx-auto px-4 sm:px-8 lg:px-12 py-12">
        {#key $location.pathname}
          <div class="page-content">
            <slot />
          </div>
        {/key}
      </div>
    </main>
  </div>

  <footer
    class="{isCollapsed
      ? 'lg:pl-[88px]'
      : 'lg:pl-64'} border-t border-outline-variant/10 bg-surface-container-lowest/50 transition-all duration-300 ease-in-out"
  >
    <div
      class="max-w-7xl mx-auto px-4 sm:px-8 lg:px-12 py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-[10px] text-on-surface-variant/60 uppercase tracking-widest font-bold"
    >
      <span>&copy; 2026 Pawikan Sentinel</span>
      <div class="flex gap-6">
        <span>Real-time Conservation Technology</span>
      </div>
    </div>
  </footer>
</div>
