<script>
  import { onMount } from 'svelte';
  import { config, fetchSettings, updateConfig, createBackup, backups, contacts } from '../stores/settings';
  import { cameras, fetchCameras } from '../stores/cameras';
  import Badge from '../components/Badge.svelte';

  onMount(() => {
    fetchSettings();
    fetchCameras();
  });

  async function handleConfigUpdate() {
    const success = await updateConfig($config);
    if (success) {
      alert("Configuration updated successfully!");
    }
  }

  async function handleCreateBackup() {
    const success = await createBackup();
    if (success) {
      alert("Backup created!");
    }
  }
</script>

<div class="page-content">
  <header class="mb-10">
    <h2 class="text-3xl font-extrabold text-on-surface tracking-tight mb-2">System Settings</h2>
    <p class="text-on-surface-variant font-body">Manage sentinel configurations, surveillance streams, and conservationist contacts.</p>
  </header>

  <div class="bento-grid">
    <!-- System Configuration -->
    <section class="lg:col-span-8 bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/15 flex flex-col gap-8">
      <div class="flex items-center gap-2 mb-2">
        <span class="material-symbols-outlined text-primary">settings_input_component</span>
        <h3 class="text-xl font-bold font-headline">System Configuration</h3>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-10">
        <!-- Confidence Threshold -->
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <label for="conf-threshold" class="text-sm font-semibold text-on-surface">Confidence Threshold</label>
            <Badge text="{$config.confidence_threshold}%" variant="primary" />
          </div>
          <input id="conf-threshold" type="range" class="w-full accent-primary" min="50" max="100" bind:value={$config.confidence_threshold} on:change={handleConfigUpdate} />
          <p class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Minimum detection certainty for species identification.</p>
        </div>

        <!-- Frame Skip -->
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <label for="frame-skip" class="text-sm font-semibold text-on-surface">Frame Skip</label>
            <Badge text="{$config.frame_skip} Frames" variant="secondary" />
          </div>
          <input id="frame-skip" type="range" class="w-full accent-secondary" min="1" max="60" bind:value={$config.frame_skip} on:change={handleConfigUpdate} />
          <p class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Process every Nth frame to optimize hardware.</p>
        </div>

        <!-- SMS Cooldown -->
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <label for="sms-cooldown" class="text-sm font-semibold text-on-surface">SMS Cooldown</label>
            <Badge text="{$config.sms_cooldown} Mins" variant="tertiary" />
          </div>
          <input id="sms-cooldown" type="range" class="w-full accent-tertiary" min="1" max="120" bind:value={$config.sms_cooldown} on:change={handleConfigUpdate} />
          <p class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Silence interval between alert notifications.</p>
        </div>

        <div class="bg-surface-container-low rounded-xl p-4 flex flex-col justify-between border-l-4 border-primary">
          <div>
            <h4 class="text-sm font-bold mb-1">Export Detections</h4>
            <p class="text-xs text-on-surface-variant mb-4">Download all logged nesting and hatching data.</p>
          </div>
          <button class="btn-gradient-primary text-xs w-full">Export to CSV</button>
        </div>
      </div>
    </section>

    <!-- Account -->
    <section class="lg:col-span-4 bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/15">
      <div class="flex items-center gap-2 mb-6">
        <span class="material-symbols-outlined text-primary">account_circle</span>
        <h3 class="text-xl font-bold font-headline">Account</h3>
      </div>
      <form class="space-y-5">
        <div class="space-y-1">
          <label for="current-username" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Username</label>
          <input id="current-username" type="text" class="input-surface w-full" value="conservation_admin_01" readonly />
        </div>
        <div class="space-y-1">
          <label for="new-password" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">New Password</label>
          <input id="new-password" type="password" class="input-surface w-full" placeholder="••••••••••••" />
        </div>
        <button class="btn-secondary w-full text-xs">Update Credentials</button>
      </form>
    </section>

    <!-- Cameras -->
    <section class="lg:col-span-12 bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/15">
      <div class="flex justify-between items-center mb-6">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-primary">videocam</span>
          <h3 class="text-xl font-bold font-headline">Cameras</h3>
        </div>
        <button class="text-primary font-bold text-sm flex items-center gap-1 hover:underline underline-offset-4">
          <span class="material-symbols-outlined text-sm">add_circle</span>
          Add New Stream
        </button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {#each $cameras as cam}
          <div class="bg-surface-container-low p-4 rounded-xl flex flex-col gap-3 group">
            <div class="flex justify-between items-start">
              <div>
                <h4 class="font-bold text-sm">{cam.name}</h4>
                <p class="text-[10px] text-on-surface-variant font-mono truncate max-w-[150px]">{cam.rtsp_url}</p>
              </div>
              <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button class="p-1 text-on-surface-variant hover:text-primary"><span class="material-symbols-outlined text-lg">edit</span></button>
                <button class="p-1 text-on-surface-variant hover:text-error"><span class="material-symbols-outlined text-lg">delete</span></button>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <Badge text={cam.active ? "Active" : "Offline"} variant={cam.active ? "success" : "warning"} />
              <span class="text-[10px] text-on-surface-variant italic">ID: #{cam.id}</span>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Contacts -->
    <section class="lg:col-span-6 bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/15">
      <div class="flex justify-between items-center mb-6">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-primary">contact_mail</span>
          <h3 class="text-xl font-bold font-headline">Contacts</h3>
        </div>
        <button class="bg-surface-container-low p-2 rounded-lg hover:bg-surface-container-high transition-colors">
          <span class="material-symbols-outlined text-lg">person_add</span>
        </button>
      </div>
      <div class="space-y-3">
        {#each $contacts as contact}
          <div class="flex items-center justify-between p-3 bg-surface-container-low rounded-xl">
            <div class="flex items-center gap-3">
              <div class="w-8 h-8 rounded-full bg-primary-fixed flex items-center justify-center text-primary font-bold text-xs">
                {contact.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div>
                <p class="text-sm font-bold">{contact.name}</p>
                <p class="text-[10px] text-on-surface-variant">{contact.phone}</p>
              </div>
            </div>
            <span class="material-symbols-outlined text-on-surface-variant text-sm cursor-pointer">more_vert</span>
          </div>
        {/each}
      </div>
    </section>

    <!-- System Backup -->
    <section class="lg:col-span-6 bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/15">
      <div class="flex justify-between items-start mb-6">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-primary">backup</span>
          <h3 class="text-xl font-bold font-headline">System Backup</h3>
        </div>
        <button class="btn-gradient-primary text-xs px-4 py-2" on:click={handleCreateBackup}>
          Create Backup
        </button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left">
          <thead class="border-b border-outline-variant/15">
            <tr>
              <th class="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Date</th>
              <th class="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Size</th>
              <th class="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Status</th>
              <th class="py-3 text-right"></th>
            </tr>
          </thead>
          <tbody class="divide-y divide-outline-variant/10">
            {#each $backups as backup}
              <tr>
                <td class="py-3 text-xs font-medium">{backup.timestamp}</td>
                <td class="py-3 text-xs text-on-surface-variant">{backup.size}</td>
                <td class="py-3">
                  <span class="text-[10px] text-primary font-bold px-2 py-0.5 bg-primary-fixed rounded uppercase">Success</span>
                </td>
                <td class="py-3 text-right">
                  <button class="text-primary hover:bg-primary-fixed p-1 rounded-full transition-colors">
                    <span class="material-symbols-outlined text-sm">restore</span>
                  </button>
                </td>
              </tr>
            {:else}
              <tr>
                <td colspan="4" class="py-8 text-center text-xs text-on-surface-variant italic">No backups found.</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</div>
