import { writable } from 'svelte/store';

export const recentDetections = writable([]);
export const galleryImages = writable([]);
export const stats = writable({ total: 0, today: 0, this_month: 0, week_increase: 0, peak_time: 'N/A' });
export const lightboxStore = writable({ isOpen: false, image: null, index: -1, images: [] });

let socket;

export function connectWebSocket() {
  if (socket) return;
  
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // Note: During local dev with Vite, the backend is likely on a different port.
  // In production, it will be the same host.
  const host = window.location.host; 
  socket = new WebSocket(`${protocol}//${host}/ws`);

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'new_detection') {
      const detection = message.detection || message.data;
      if (!detection) return;

      recentDetections.update(list => [detection, ...list]);
      
      // Also format and prepend to gallery
      const imagePath = detection.image_path || detection.image;
      if (imagePath) {
        galleryImages.update(list => {
          const pathParts = imagePath.split('/');
          const filename = pathParts[pathParts.length - 1];
          const newImage = {
            filename: filename,
            path: `/detections/${filename}`,
            modified: detection.timestamp ? new Date(detection.timestamp.replace(' ', 'T')).getTime() / 1000 : Date.now() / 1000,
            camera_name: detection.camera_name || 'Camera'
          };
          return [newImage, ...list];
        });
      }
      
      // Increment stats locally for immediate feedback
      stats.update(s => ({
        ...s,
        total: s.total + 1,
        today: s.today + 1,
        this_month: s.this_month + 1
      }));
    }
  };

  socket.onclose = () => {
    socket = null;
    setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
  };
}

export async function fetchInitialData() {
  try {
    const statsRes = await fetch('/api/detections/stats', { credentials: 'include' });
    if (statsRes.ok) stats.set(await statsRes.json());
    
    const recentRes = await fetch('/api/detections/recent', { credentials: 'include' });
    if (recentRes.ok) recentDetections.set(await recentRes.json());

    const galleryRes = await fetch('/api/detections/gallery', { credentials: 'include' });
    if (galleryRes.ok) galleryImages.set(await galleryRes.json());
  } catch (error) {
    console.error('Error fetching initial data:', error);
  }
}
