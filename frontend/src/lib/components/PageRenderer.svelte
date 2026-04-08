<script>
  import { useLocation } from "svelte-routing";
  import Dashboard from "../pages/Dashboard.svelte";
  import Cameras from "../pages/Cameras.svelte";
  import Settings from "../pages/Settings.svelte";
  import Archive from "../pages/Archive.svelte";

  const location = useLocation();
</script>

<div class="page-transition-wrapper">
  {#if $location.pathname === "/cameras"}
    <Cameras />
  {:else if $location.pathname === "/settings"}
    <Settings />
  {:else if $location.pathname === "/archive"}
    <Archive />
  {:else if $location.pathname === "/login"}
    <Dashboard />
  {:else}
    <Dashboard />
  {/if}
</div>
