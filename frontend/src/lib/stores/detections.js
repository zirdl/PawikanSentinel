import { writable } from 'svelte/store';

export const recentDetections = writable([]);
export const stats = writable({ total: 0, today: 0, this_month: 0 });

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
      const detection = message.data;
      recentDetections.update(list => [detection, ...list].slice(0, 10));
      
      // Increment stats locally for immediate feedback
      stats.update(s => ({
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
  } catch (error) {
    console.error('Error fetching initial data:', error);
  }
}
