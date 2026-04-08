/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{svelte,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary Brand (Digital Naturalist)
        primary: "#134231",           // Deep forest green
        "primary-container": "#2d5a47", // Medium forest green
        "primary-fixed": "#bcedd4",    // Light mint
        "primary-fixed-dim": "#a1d1b9", // Soft mint
        "on-primary": "#ffffff",
        "on-primary-container": "#9fcfb7",
        "on-primary-fixed": "#002115",
        "on-primary-fixed-variant": "#214f3d",

        // Secondary
        secondary: "#276868",         // Deep teal
        "secondary-container": "#acebeb", // Light teal
        "secondary-fixed": "#afeeed",   // Pale teal
        "secondary-fixed-dim": "#93d2d1", // Soft teal
        "on-secondary": "#ffffff",
        "on-secondary-container": "#2c6c6c",
        "on-secondary-fixed": "#002020",
        "on-secondary-fixed-variant": "#004f50",

        // Tertiary
        tertiary: "#194135",          // Dark sea green
        "tertiary-container": "#31594b", // Medium sea green
        "tertiary-fixed": "#c0ecda",   // Light seafoam
        "tertiary-fixed-dim": "#a5d0be", // Soft seafoam
        "on-tertiary": "#ffffff",
        "on-tertiary-container": "#a3cebc",
        "on-tertiary-fixed": "#002117",
        "on-tertiary-fixed-variant": "#264e41",

        // Surface Hierarchy
        background: "#f8faf9",
        surface: "#f8faf9",
        "surface-bright": "#f8faf9",
        "surface-dim": "#d8dad9",
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f2f4f3",
        "surface-container": "#eceeed",
        "surface-container-high": "#e6e9e8",
        "surface-container-highest": "#e1e3e2",
        "surface-variant": "#e1e3e2",
        "surface-tint": "#3a6753",

        // Text & Contrast
        "on-surface": "#191c1c",       // Primary text
        "on-surface-variant": "#414944", // Secondary text
        "on-background": "#191c1c",
        "inverse-surface": "#2e3131",
        "inverse-on-surface": "#eff1f0",
        "inverse-primary": "#a1d1b9",

        // Outline & State
        outline: "#717974",
        "outline-variant": "#c0c8c2",
        error: "#ba1a1a",
        "error-container": "#ffdad6",
        "on-error": "#ffffff",
        "on-error-container": "#93000a",
      },
      fontFamily: {
        headline: ["Manrope", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        label: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        "xl": "12px",
        "2xl": "1.5rem", // xl in Material
      },
      boxShadow: {
        'card': '0px 12px 32px rgba(25, 28, 28, 0.06)',
        'elevated': '0px 16px 48px rgba(25, 28, 28, 0.1)',
      }
    },
  },
  plugins: [],
}
