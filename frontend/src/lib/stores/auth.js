import { writable } from 'svelte/store';
import { navigate } from 'svelte-routing';

export const user = writable(null);
export const isAuthenticated = writable(false);
export const loading = writable(true);

export async function checkAuth() {
  loading.set(true);
  try {
    const res = await fetch('/api/auth/me', { credentials: 'include' });
    if (res.ok) {
      const data = await res.json();
      user.set(data);
      isAuthenticated.set(true);
    } else {
      user.set(null);
      isAuthenticated.set(false);
    }
  } catch (e) {
    console.error("Auth check failed:", e);
    isAuthenticated.set(false);
  } finally {
    loading.set(false);
  }
}

export async function login(username, password) {
  try {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);
    
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: formData,
      credentials: 'include'
    });

    if (res.ok) {
      await checkAuth();
      navigate('/');
      return { success: true };
    } else {
      const data = await res.json();
      return { success: false, error: data.detail || "Invalid credentials" };
    }
  } catch (e) {
    return { success: false, error: "Network error" };
  }
}

export async function logout() {
  try {
    await fetch('/logout', { method: 'POST', credentials: 'include' });
    user.set(null);
    isAuthenticated.set(false);
    navigate('/login');
  } catch (e) {
    console.error("Logout failed:", e);
  }
}
