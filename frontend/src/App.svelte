<script>
  import { onMount } from "svelte";
  import { Router } from "svelte-routing";
  import { isAuthenticated, loading, checkAuth } from "./lib/stores/auth";
  
  import MainLayout from "./lib/components/MainLayout.svelte";
  import PageRenderer from "./lib/components/PageRenderer.svelte";
  import Login from "./lib/pages/Login.svelte";

  export let url = "";

  onMount(() => {
    checkAuth();
  });
</script>

<Router {url}>
  <div class="pawikan-app-root">
    {#if $loading}
      <div class="min-h-screen bg-surface flex items-center justify-center">
        <div class="flex flex-col items-center gap-4">
          <div class="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
          <p class="text-[10px] font-bold text-primary uppercase tracking-widest">Waking Sentinel...</p>
        </div>
      </div>
    {:else if !$isAuthenticated}
      <Login />
    {:else}
      <MainLayout>
        <PageRenderer />
      </MainLayout>
    {/if}
  </div>
</Router>
