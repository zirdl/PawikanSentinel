/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/web/templates/**/*.html',
    './src/web/static/js/**/*.js',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'sans-serif'],
      },

      /* =========================================================
         DESIGN TOKENS — Pawikan Sentinel
         ========================================================= */

      colors: {
        /* --- Turtle Green (brand primary) --- */
        turtle: {
          50:  '#f0f7f2',
          100: '#dcf0e1',
          200: '#bce1c7',
          300: '#8ecba2',
          400: '#5eae78',
          500: '#4a7c59',  // original turtle-green
          600: '#3a6448',
          700: '#3a5f42',  // original turtle-dark-green
          800: '#2d4a35',
          900: '#1f3326',
          950: '#0f1d15',
        },

        /* --- Ocean Blue (secondary) --- */
        ocean: {
          50:  '#eff6fc',
          100: '#dcecf8',
          200: '#c1ddf2',
          300: '#95c5e7',
          400: '#6b9ac4',  // original ocean-blue
          500: '#5a8bb8',
          600: '#4a7b9d',  // original ocean-dark-blue
          700: '#3d6685',
          800: '#2f4f68',
          900: '#1e3447',
          950: '#101e2c',
        },

        /* --- Sand (warm neutral) --- */
        sand: {
          50:  '#fdf9f3',
          100: '#f5e7d3',  // original sand-light
          200: '#edd9be',
          300: '#e7d4b5',  // original sand-medium
          400: '#d9c297',  // original sand-dark
          500: '#c4a974',
          600: '#a88854',
          700: '#8c6d42',
          800: '#6e5435',
          900: '#4a3825',
          950: '#2c2015',
        },

        /* --- Shell / Earth accent --- */
        shell: {
          50:  '#f5f7ef',
          100: '#e9edd9',
          200: '#d4dbb5',
          300: '#b7c586',
          400: '#8b9c6d',  // original turtle-shell
          500: '#6f8153',
          600: '#56653e',
          700: '#424e31',
          800: '#333c28',
          900: '#232a1c',
          950: '#12160e',
        },

        /* --- Coral / Sunset (warning / accent) --- */
        coral: {
          50:  '#fef3ee',
          100: '#fde4d7',
          200: '#fac7ab',
          300: '#f4a261',  // original sunset-orange
          400: '#ee8342',
          500: '#e76f51',  // original coral
          600: '#d4543a',
          700: '#b04330',
          800: '#8e372b',
          900: '#602420',
          950: '#34110e',
        },

        /* --- Semantic / state colors --- */
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

      /* --- Typography scale --- */
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

      /* --- Border radius --- */
      borderRadius: {
        'sm':   '0.25rem',
        'md':   '0.375rem',
        'lg':   '0.5rem',
        'xl':   '0.75rem',
        '2xl':  '1rem',
        '3xl':  '1.25rem',
        'full': '9999px',
      },

      /* --- Shadows --- */
      boxShadow: {
        'card':   '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04)',
        'elevated': '0 10px 25px rgba(0,0,0,0.08), 0 4px 10px rgba(0,0,0,0.04)',
        'overlay':  '0 20px 40px rgba(0,0,0,0.15), 0 8px 16px rgba(0,0,0,0.08)',
        'inset':    'inset 0 1px 2px rgba(0,0,0,0.05)',
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