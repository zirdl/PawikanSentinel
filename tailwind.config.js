/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/web/templates/**/*.html',
    './src/web/static/js/**/*.js',
  ],
  theme: {
    extend: {
      /* =========================================================
         TYPOGRAPHY — Digital Naturalist System
         Headlines: Manrope (bold, authoritative)
         Body: Inter (legible, neutral)
         ========================================================= */
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        headline: ['Manrope', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
        label: ['Inter', 'system-ui', 'sans-serif'],
      },

      /* =========================================================
         DESIGN TOKENS — Pawikan Sentinel (Stitch Design System)
         Theme: "The Digital Naturalist"
         Palette: Deep-Sea Ecology (mangroves + tropical shallows)
         ========================================================= */

      colors: {
        /* --- Primary Brand (Deep Forest Green) --- */
        primary: {
          DEFAULT: '#134231',
          container: '#2d5a47',
          fixed: '#bcedd4',
          fixedDim: '#a1d1b9',
          on: '#ffffff',
          onContainer: '#9fcfb7',
          onFixed: '#002115',
          onFixedVariant: '#214f3d',
        },

        /* --- Secondary (Deep Teal) --- */
        secondary: {
          DEFAULT: '#276868',
          container: '#acebeb',
          fixed: '#afeeed',
          fixedDim: '#93d2d1',
          on: '#ffffff',
          onContainer: '#2c6c6c',
          onFixed: '#002020',
          onFixedVariant: '#004f50',
        },

        /* --- Tertiary (Dark Sea Green) --- */
        tertiary: {
          DEFAULT: '#194135',
          container: '#31594b',
          fixed: '#c0ecda',
          fixedDim: '#a5d0be',
          on: '#ffffff',
          onContainer: '#a3cebc',
          onFixed: '#002117',
          onFixedVariant: '#264e41',
        },

        /* --- Surface Hierarchy (Tonal Layering) --- */
        surface: {
          DEFAULT: '#f8faf9',
          bright: '#f8faf9',
          dim: '#d8dad9',
          container: {
            DEFAULT: '#eceeed',
            low: '#f2f4f3',
            lowest: '#ffffff',
            high: '#e6e9e8',
            highest: '#e1e3e2',
          },
          tint: '#3a6753',
          variant: '#e1e3e2',
        },

        /* --- Text Colors --- */
        'on-surface': {
          DEFAULT: '#191c1c',
          variant: '#414944',
        },
        'on-background': '#191c1c',

        /* --- Outline & Borders --- */
        outline: {
          DEFAULT: '#717974',
          variant: '#c0c8c2',
        },

        /* --- Inverse Colors --- */
        'inverse-surface': '#2e3131',
        'inverse-on-surface': '#eff1f0',
        'inverse-primary': '#a1d1b9',

        /* --- Background --- */
        background: '#f8faf9',

        /* --- Error States --- */
        error: {
          DEFAULT: '#ba1a1a',
          container: '#ffdad6',
          on: '#ffffff',
          onContainer: '#93000a',
        },

        /* --- Legacy Aliases (for backward compatibility) --- */
        turtle: {
          50:  '#f0f7f2',
          100: '#dcf0e1',
          200: '#bce1c7',
          300: '#8ecba2',
          400: '#5eae78',
          500: '#4a7c59',  // Legacy: mapped to primary.container
          600: '#3a6448',
          700: '#3a5f42',
          800: '#2d4a35',
          900: '#1f3326',
          950: '#0f1d15',
        },
        ocean: {
          50:  '#eff6fc',
          100: '#dcecf8',
          200: '#c1ddf2',
          300: '#95c5e7',
          400: '#6b9ac4',
          500: '#5a8bb8',
          600: '#4a7b9d',
          700: '#3d6685',
          800: '#2f4f68',
          900: '#1e3447',
          950: '#101e2c',
        },

        /* --- Semantic / State Colors --- */
        success: {
          50:  '#f0fdf4', 100: '#dcfce7', 200: '#bbf7d0',
          300: '#86efac', 400: '#4ade80', 500: '#22c55e',
          600: '#16a34a', 700: '#15803d', 800: '#166534', 900: '#14532d',
        },
        warning: {
          50:  '#fffbeb', 100: '#fef3c7', 200: '#fde68a',
          300: '#fcd34d', 400: '#fbbf24', 500: '#f59e0b',
          600: '#d97706', 700: '#b45309', 800: '#92400e', 900: '#78350f',
        },
        danger: {
          50:  '#fef2f2', 100: '#fee2e2', 200: '#fecaca',
          300: '#fca5a5', 400: '#f87171', 500: '#ef4444',
          600: '#dc2626', 700: '#b91c1c', 800: '#991b1b', 900: '#7f1d1d',
        },
        info: {
          50:  '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe',
          300: '#93c5fd', 400: '#60a5fa', 500: '#3b82f6',
          600: '#2563eb', 700: '#1d4ed8', 800: '#1e40af', 900: '#1e3a8a',
        },
      },

      /* --- Spacing scale (in rem, base = 16px) --- */
      spacing: {
        '4.5': '1.125rem',
        '5.5': '1.375rem',
        '7.5': '1.875rem',
        '13':  '3.25rem',
        '15':  '3.75rem',
        '18':  '4.5rem',
        '88':  '22rem',
      },

      /* --- Typography Scale — Editorial System --- */
      fontSize: {
        'caption':  ['0.75rem',  { lineHeight: '1rem',    fontWeight: '500' }],
        'body-sm':  ['0.8125rem', { lineHeight: '1.25rem', fontWeight: '400' }],
        'body':     ['0.875rem',  { lineHeight: '1.5rem',  fontWeight: '400' }],
        'body-lg':  ['1rem',      { lineHeight: '1.75rem', fontWeight: '400' }],
        'subtitle': ['1.125rem',  { lineHeight: '1.75rem', fontWeight: '500' }],
        'title':    ['1.25rem',   { lineHeight: '1.75rem', fontWeight: '600' }],
        'h6':       ['1.125rem',  { lineHeight: '1.5rem',  fontWeight: '600' }],
        'h5':       ['1.25rem',   { lineHeight: '1.75rem', fontWeight: '600' }],
        'h4':       ['1.5rem',    { lineHeight: '2rem',    fontWeight: '600' }],
        'h3':       ['1.875rem',  { lineHeight: '2.25rem', fontWeight: '700' }],
        'h2':       ['2.25rem',   { lineHeight: '2.5rem',  fontWeight: '700' }],
        'h1':       ['3rem',      { lineHeight: '1.15',    fontWeight: '800' }],
        'display':  ['3.75rem',   { lineHeight: '1',      fontWeight: '800' }],
      },

      /* --- Border Radius (Soft, Natural) --- */
      borderRadius: {
        'sm':   '0.25rem',
        'md':   '0.375rem',
        'lg':   '0.5rem',
        'xl':   '0.75rem',
        '2xl':  '1rem',
        '3xl':  '1.25rem',
        'full': '9999px',
      },

      /* --- Shadows (Ambient, Ultra-Diffused) --- */
      boxShadow: {
        'card':   '0px 12px 32px rgba(25, 28, 28, 0.06)',
        'card-hover': '0px 16px 40px rgba(25, 28, 28, 0.10)',
        'elevated': '0px 20px 48px rgba(25, 28, 28, 0.08)',
        'overlay':  '0px 24px 56px rgba(25, 28, 28, 0.12)',
        'inset':    'inset 0 1px 2px rgba(25, 28, 28, 0.05)',
      },

      /* --- Transition durations --- */
      transitionDuration: {
        '250': '250ms',
        '350': '350ms',
      },

      /* --- Z-index --- */
      zIndex: {
        '60':  '60',
        '70':  '70',
        '80':  '80',
        '90':  '90',
        '100': '100',
      },
    },
  },

  plugins: [],

  safelist: [
    /* HTMX dynamic content helpers */
    'bg-turtle-500', 'text-white', 'bg-white',
    'bg-white/20', 'hover:bg-white/30',
    'bg-coral-500', 'hover:bg-coral-600',
    'bg-ocean-400', 'bg-green-100', 'text-green-800',
    'bg-red-100', 'text-red-800',
    'hx-loading', 'hx-request',
  ],
}