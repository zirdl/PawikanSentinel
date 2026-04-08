import { writable } from 'svelte/store';

export const cameras = writable([]);
export const loading = writable(false);

export async function fetchCameras() {
  loading.set(true);
  try {
    const res = await fetch('/api/cameras', { credentials: 'include' });
    if (res.ok) {
      const data = await res.json();
      cameras.set(data);
    }
  } catch (error) {
    console.error('Error fetching cameras:', error);
  } finally {
    loading.set(false);
  }
}

export async function toggleInference(cameraId, active) {
  const endpoint = active ? `/api/inference/stop/${cameraId}` : `/api/inference/start/${cameraId}`;
  try {
    const res = await fetch(endpoint, { method: 'POST', credentials: 'include' });
    if (res.ok) {
      // Update local state
      cameras.update(list => list.map(c => 
        c.id === cameraId ? { ...c, active: !active } : c
      ));
      return true;
    }
  } catch (error) {
    console.error('Error toggling inference:', error);
  }
  return false;
}
