import { writable } from 'svelte/store';

export const config = writable({
  confidence_threshold: 85,
  frame_skip: 2,
  sms_cooldown: 15
});

export const contacts = writable([]);
export const backups = writable([]);

export async function fetchSettings() {
  try {
    const configRes = await fetch('/api/config', { credentials: 'include' });
    if (configRes.ok) config.set(await configRes.json());

    const contactsRes = await fetch('/api/contacts/list', { credentials: 'include' });
    // The contacts/list currently returns HTML from the backend (HTMX legacy)
    // I should check if there's a JSON endpoint or if I need to update the backend.
    // Looking at src/api/contacts.py... it has /api/contacts (GET) returning List[Contact]
    const contactsApiRes = await fetch('/api/contacts', { credentials: 'include' });
    if (contactsApiRes.ok) contacts.set(await contactsApiRes.json());

    const backupRes = await fetch('/api/backup/history', { credentials: 'include' });
    if (backupRes.ok) {
        const data = await backupRes.json();
        backups.set(data.backups || []);
    }
  } catch (error) {
    console.error('Error fetching settings:', error);
  }
}

export async function updateConfig(newConfig) {
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        });
        return res.ok;
    } catch (error) {
        console.error('Error updating config:', error);
        return false;
    }
}

export async function createBackup() {
    try {
        const res = await fetch('/api/backup', { method: 'POST', credentials: 'include' });
        if (res.ok) {
            await fetchSettings(); // Refresh history
            return true;
        }
    } catch (error) {
        console.error('Error creating backup:', error);
    }
    return false;
}
