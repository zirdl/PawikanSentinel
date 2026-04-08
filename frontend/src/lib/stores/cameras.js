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
export async function addCamera(camera) {
  try {
    const res = await fetch('/api/cameras', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(camera)
    });
    if (res.ok) {
      await fetchCameras();
      return true;
    }
  } catch (error) {
    console.error('Error adding camera:', error);
  }
  return false;
}

export async function updateCamera(id, camera) {
  try {
    const res = await fetch(`/api/cameras/${id}`, {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(camera)
    });
    if (res.ok) {
      await fetchCameras();
      return true;
    }
  } catch (error) {
    console.error('Error updating camera:', error);
  }
  return false;
}

export async function deleteCamera(id) {
  try {
    const res = await fetch(`/api/cameras/${id}`, {
      method: 'DELETE',
      credentials: 'include'
    });
    if (res.ok) {
      await fetchCameras();
      return true;
    }
  } catch (error) {
    console.error('Error deleting camera:', error);
  }
  return false;
}
