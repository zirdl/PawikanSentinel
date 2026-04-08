<script>
  import { onMount } from 'svelte';
  import { config, fetchSettings, updateConfig, createBackup, backups, contacts, addContact, updateContact, deleteContact } from '../stores/settings';
  import { cameras, fetchCameras, addCamera, updateCamera, deleteCamera } from '../stores/cameras';
  import { changePassword, user } from '../stores/auth';
  import Badge from '../components/Badge.svelte';
  import Modal from '../components/Modal.svelte';
  import CameraModal from '../components/CameraModal.svelte';

  let isCameraModalOpen = false;
  let isContactModalOpen = false;
  
  let cameraForm = { name: '', rtsp_url: '', active: true };
  let contactForm = { name: '', phone: '' };
  
  let editingCameraId = null;
  let editingContactId = null;

  let oldPassword = '';
  let newPassword = '';
  let passwordStatus = '';
  let showOldPassword = false;
  let showNewPassword = false;

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

  async function handleExportCSV() {
    window.open('/api/detections/export', '_blank');
  }

  // Camera Handlers
  function openCameraModal(cam = null) {
    if (cam) {
      cameraForm = { name: cam.name, rtsp_url: cam.rtsp_url, active: cam.active };
      editingCameraId = cam.id;
    } else {
      cameraForm = { name: '', rtsp_url: '', active: true };
      editingCameraId = null;
    }
    isCameraModalOpen = true;
  }

  async function removeCamera(id) {
    if (confirm("Are you sure you want to delete this camera?")) {
      await deleteCamera(id);
    }
  }

  // Contact Handlers
  function openContactModal(contact = null) {
    if (contact) {
      contactForm = { name: contact.name, phone: contact.phone };
      editingContactId = contact.id;
    } else {
      contactForm = { name: '', phone: '' };
      editingContactId = null;
    }
    isContactModalOpen = true;
  }

  async function saveContact() {
    let success;
    if (editingContactId) {
      success = await updateContact(editingContactId, contactForm);
    } else {
      success = await addContact(contactForm);
    }
    if (success) {
      isContactModalOpen = false;
    }
  }

  async function removeContact(id) {
    if (confirm("Are you sure you want to delete this contact?")) {
      await deleteContact(id);
    }
  }

  async function handlePasswordChange() {
    if (!oldPassword || !newPassword) {
      passwordStatus = "Please fill all fields";
      return;
    }
    const result = await changePassword(oldPassword, newPassword);
    if (result.success) {
      passwordStatus = "Password updated successfully!";
      oldPassword = '';
      newPassword = '';
    } else {
      passwordStatus = result.error;
    }
  }
</script>

  <header class="mb-10">
    <h2 class="text-3xl font-extrabold text-on-surface tracking-tight mb-2">System Settings</h2>
    <p class="text-on-surface-variant font-body">Manage sentinel configurations, surveillance streams, and conservationist contacts.</p>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
    <!-- System Configuration -->
    <section class="lg:col-span-12 bg-surface-container-lowest rounded-xl p-6 shadow-md border border-outline-variant/15 flex flex-col gap-8">
      <div class="flex items-center gap-2 mb-2">
        <span class="material-symbols-outlined text-primary">settings_input_component</span>
        <h3 class="text-xl font-bold font-headline">System Configuration</h3>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-10">
        <!-- Confidence Threshold -->
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <label for="conf-threshold" class="text-sm font-semibold text-on-surface">Confidence Threshold</label>
            <Badge text="{$config.confidence_threshold}%" variant="primary" />
          </div>
          <input id="conf-threshold" type="range" class="w-full accent-primary" min="50" max="100" bind:value={$config.confidence_threshold} on:change={handleConfigUpdate} />
          <p class="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">Minimum certainty for Species Identification.</p>
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
      </div>
      <!-- Export button full width row underneath the 3 columns -->
      <div class="bg-surface-container-low rounded-xl p-4 flex flex-col md:flex-row justify-between items-center border-l-4 border-primary">
        <div>
          <h4 class="text-sm font-bold mb-1">Export Detections</h4>
          <p class="text-xs text-on-surface-variant md:mb-0 mb-4">Download all logged nesting and hatching data.</p>
        </div>
        <button class="btn-gradient-primary text-xs" on:click={handleExportCSV}>Export to CSV</button>
      </div>
    </section>

    <!-- Contacts (Left) -->
    <section class="lg:col-span-6 bg-surface-container-lowest rounded-xl p-6 shadow-md border border-outline-variant/15">
      <div class="flex justify-between items-center mb-6">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-primary">contact_mail</span>
          <h3 class="text-xl font-bold font-headline">Contacts</h3>
        </div>
        <button class="bg-surface-container-low p-2 rounded-lg hover:bg-primary-fixed hover:text-primary transition-all active:scale-90" on:click={() => openContactModal()}>
          <span class="material-symbols-outlined text-lg">person_add</span>
        </button>
      </div>
      <div class="space-y-3">
        {#each $contacts as contact}
          <div class="flex items-center justify-between p-3 bg-surface-container-low rounded-xl group transition-all hover:shadow-md border border-transparent hover:border-outline-variant/10">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-full bg-primary-fixed flex items-center justify-center text-primary font-bold text-xs shadow-inner">
                {contact.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div>
                <p class="text-sm font-bold">{contact.name}</p>
                <p class="text-[10px] text-on-surface-variant flex items-center gap-1">
                   <span class="material-symbols-outlined text-[10px]">call</span>
                   {contact.phone}
                </p>
              </div>
            </div>
            <div class="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button class="p-1.5 text-on-surface-variant hover:text-primary rounded-lg transition-all" on:click={() => openContactModal(contact)}>
                <span class="material-symbols-outlined text-lg">edit</span>
              </button>
              <button class="p-1.5 text-on-surface-variant hover:text-error rounded-lg transition-all" on:click={() => removeContact(contact.id)}>
                <span class="material-symbols-outlined text-lg">delete</span>
              </button>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Account (Right) -->
    <section class="lg:col-span-6 bg-surface-container-lowest rounded-xl p-6 shadow-md border border-outline-variant/15">
      <div class="flex items-center gap-2 mb-6">
        <span class="material-symbols-outlined text-primary">account_circle</span>
        <h3 class="text-xl font-bold font-headline">Account</h3>
      </div>
      <form class="space-y-5" on:submit|preventDefault={handlePasswordChange}>
        <div class="space-y-1">
          <label for="current-username" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Username</label>
          <input id="current-username" type="text" class="input-surface w-full" value={$user?.username || 'admin'} readonly />
        </div>
        <div class="space-y-1">
          <label for="old-password" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Current Password</label>
          <div class="relative">
            <input 
              id="old-password" 
              type={showOldPassword ? "text" : "password"} 
              class="input-surface w-full pr-12" 
              placeholder="••••••••••••" 
              bind:value={oldPassword} 
            />
            <button 
              type="button"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-on-surface-variant hover:text-primary transition-colors focus:outline-none"
              on:click={() => showOldPassword = !showOldPassword}
            >
              <span class="material-symbols-outlined text-lg">
                {showOldPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
        </div>
        <div class="space-y-1">
          <label for="new-password" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">New Password</label>
          <div class="relative">
            <input 
              id="new-password" 
              type={showNewPassword ? "text" : "password"} 
              class="input-surface w-full pr-12" 
              placeholder="••••••••••••" 
              bind:value={newPassword} 
            />
            <button 
              type="button"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-on-surface-variant hover:text-primary transition-colors focus:outline-none"
              on:click={() => showNewPassword = !showNewPassword}
            >
              <span class="material-symbols-outlined text-lg">
                {showNewPassword ? 'visibility_off' : 'visibility'}
              </span>
            </button>
          </div>
        </div>
        {#if passwordStatus}
          <p class="text-[10px] font-bold {passwordStatus.includes('successfully') ? 'text-green-500' : 'text-red-500'}">{passwordStatus}</p>
        {/if}
        <button class="btn-secondary w-full text-xs" type="submit">Update Credentials</button>
      </form>
    </section>

    <!-- System Backup -->
    <section class="lg:col-span-12 bg-surface-container-lowest rounded-xl p-6 shadow-md border border-outline-variant/15">
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

<CameraModal bind:isOpen={isCameraModalOpen} bind:camera={cameraForm} bind:cameraId={editingCameraId} onSave={fetchCameras} />

<!-- Contact Modal -->
<Modal title={editingContactId ? "Update Contact" : "Register Sentinel"} isOpen={isContactModalOpen} on:close={() => isContactModalOpen = false}>
  <div class="space-y-4">
    <div class="space-y-1">
      <label for="contact-name" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Full Name</label>
      <input id="contact-name" type="text" class="input-surface w-full" placeholder="Juan Dela Cruz" bind:value={contactForm.name} />
    </div>
    <div class="space-y-1">
      <label for="contact-phone" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Mobile Number</label>
      <input id="contact-phone" type="text" class="input-surface w-full font-mono text-xs" placeholder="+63 900 000 0000" bind:value={contactForm.phone} />
    </div>
  </div>
  <div slot="footer" class="flex gap-3">
    <button class="btn-secondary text-xs" on:click={() => isContactModalOpen = false}>Discard</button>
    <button class="btn-gradient-primary text-xs px-6" on:click={saveContact}>Save Sentinel</button>
  </div>
</Modal>
