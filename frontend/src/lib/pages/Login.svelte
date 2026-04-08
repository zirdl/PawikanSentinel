<script>
  import { login } from "../stores/auth";
  let username = "";
  let password = "";
  let error = "";
  let showPassword = false;
  let loading = false;

  async function handleSubmit() {
    loading = true;
    error = "";
    const res = await login(username, password);
    if (!res.success) {
      error = res.error;
    }
    loading = false;
  }
</script>

<div
  class="min-h-screen relative flex items-center justify-center bg-surface p-4 overflow-hidden"
>
  <!-- Background Blurred Logo Layer -->
  <div
    class="absolute inset-0 flex items-center justify-center pointer-events-none z-0"
  >
    <img
      src="/logo.svg"
      alt=""
      class="w-[60vw] max-w-[800px] opacity-15 blur-[8px] transform scale-125"
      aria-hidden="true"
    />
  </div>

  <!-- Login Modal -->
  <div
    class="relative z-10 max-w-md w-full bg-surface-container-lowest/80 backdrop-blur-xl rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.12)] overflow-hidden border border-outline-variant/20"
  >
    <div
      class="p-8 bg-gradient-to-br from-primary to-primary-container text-on-primary text-center"
    >
      <!-- <div class="w-16 h-16 rounded-xl bg-white/20 backdrop-blur-md flex items-center justify-center mb-6 shadow-sm mx-auto overflow-hidden">
        <img src="/logo.svg" alt="" class="w-12 h-12 drop-shadow-sm" /> -->
      <!-- </div> -->
      <h1 class="text-3xl font-extrabold font-headline tracking-tight">
        Pawikan Sentinel
      </h1>
      <p
        class="text-on-primary/80 text-sm mt-1 uppercase tracking-widest font-bold"
      >
        Conservation Surveillance
      </p>
    </div>

    <div class="p-8">
      <h2 class="text-xl font-bold text-on-surface font-headline mb-6">
        Sign in to Dashboard
      </h2>

      <form on:submit|preventDefault={handleSubmit} class="space-y-5">
        {#if error}
          <div
            class="p-3 bg-error-container text-on-error-container rounded-xl text-xs font-bold uppercase tracking-tight flex items-center gap-2"
          >
            <span class="material-symbols-outlined text-sm">error</span>
            {error}
          </div>
        {/if}

        <div class="space-y-1">
          <label
            for="username"
            class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest"
            >Username</label
          >
          <div class="relative">
            <span
              class="absolute left-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-on-surface-variant text-lg"
              >person</span
            >
            <input
              id="username"
              type="text"
              bind:value={username}
              class="input-surface w-full pl-12"
              placeholder="Enter your username"
              required
            />
          </div>
        </div>

        <div class="space-y-1">
          <label
            for="password"
            class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest"
            >Password</label
          >
          <div class="relative">
            <span
              class="absolute left-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-on-surface-variant text-lg"
              >lock</span
            >
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              bind:value={password}
              class="input-surface w-full pl-12 pr-12"
              placeholder="••••••••••••"
              required
            />
            <button
              type="button"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-on-surface-variant hover:text-primary transition-colors focus:outline-none"
              on:click={() => (showPassword = !showPassword)}
            >
              <span class="material-symbols-outlined text-lg">
                {showPassword ? "visibility_off" : "visibility"}
              </span>
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          class="btn-gradient-primary w-full py-4 flex items-center justify-center gap-2 mt-4"
        >
          {#if loading}
            <span
              class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"
            ></span>
            Authenticating...
          {:else}
            <span class="material-symbols-outlined text-lg">login</span>
            Access Sentinel
          {/if}
        </button>
      </form>

      <div class="mt-8 pt-6 border-t border-outline-variant/10 text-center">
        <p
          class="text-[10px] text-on-surface-variant uppercase tracking-widest font-bold"
        >
          © 2026 Coastal Conservation Tech
        </p>
      </div>
    </div>
  </div>
</div>
