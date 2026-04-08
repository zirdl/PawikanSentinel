import { writable } from 'svelte/store';

export const sidebarOpen = writable(false);
export const toasts = writable([]);

export function addToast(message, type = 'info', duration = 4000) {
  const id = Date.now();
  toasts.update(all => [{ id, message, type }, ...all]);
  
  if (duration) {
    setTimeout(() => {
      removeToast(id);
    }, duration);
  }
}

export function removeToast(id) {
  toasts.update(all => all.filter(t => t.id !== id));
}
