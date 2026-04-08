<script>
  import { Link } from "svelte-routing";
  export let currentPath = "/";

  const navItems = [
    { path: "/", icon: "dashboard", label: "Dashboard" },
    { path: "/cameras", icon: "videocam", label: "Cameras" },
    { path: "/settings", icon: "settings", label: "Settings" },
  ];
</script>

<aside class="fixed top-16 left-0 bottom-0 z-30 w-64 bg-surface-container-high/50 border-r border-outline-variant/10 overflow-y-auto hidden lg:flex flex-col">
  <nav class="p-3 flex-1 space-y-1">
    {#each navItems as item}
      <Link
        to={item.path}
        class="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 
               {currentPath === item.path 
                 ? 'bg-primary-fixed text-primary' 
                 : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'}"
      >
        <span class="material-symbols-outlined text-lg {currentPath === item.path ? 'filled' : ''}">
          {item.icon}
        </span>
        {item.label}
      </Link>
    {/each}
  </nav>

  <!-- Sidebar footer -->
  <div class="p-4 border-t border-outline-variant/10">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
        <span class="material-symbols-outlined text-on-primary text-sm">eco</span>
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm font-bold text-on-surface truncate font-headline">Pawikan Sentinel</p>
        <p class="text-[10px] text-on-surface-variant uppercase tracking-widest">v1.0.2</p>
      </div>
    </div>
  </div>
</aside>
