/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/pawikan/dashboard/templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        'primary': {
          light: '#64B5F6',
          DEFAULT: '#1976D2',
          dark: '#0D47A1',
        },
        'button-primary': {
          light: '#81C784',
          DEFAULT: '#388E3C',
          dark: '#1B5E20',
        },
        'secondary': {
          light: '#81C784',
          DEFAULT: '#4CAF50',
          dark: '#2E7D32',
        },
        'error': {
          DEFAULT: '#D32F2F',
        },
        'surface': {
          DEFAULT: '#FFFFFF',
        },
        'background': {
          DEFAULT: '#F5F5F5',
        },
        'on-primary': {
          DEFAULT: '#FFFFFF',
        },
        'on-secondary': {
          DEFAULT: '#FFFFFF',
        },
        'on-surface': {
          DEFAULT: '#212121',
        },
        'on-background': {
          DEFAULT: '#212121',
        },
        'on-error': {
          DEFAULT: '#FFFFFF',
        },
      },
      fontSize: {
        'display-large': ['57px', { lineHeight: '64px' }],
        'display-medium': ['45px', { lineHeight: '52px' }],
        'display-small': ['36px', { lineHeight: '44px' }],
        'headline-large': ['32px', { lineHeight: '40px' }],
        'headline-medium': ['28px', { lineHeight: '36px' }],
        'headline-small': ['24px', { lineHeight: '32px' }],
        'title-large': ['22px', { lineHeight: '28px' }],
        'title-medium': ['16px', { lineHeight: '24px' }],
        'title-small': ['14px', { lineHeight: '20px' }],
        'body-large': ['16px', { lineHeight: '24px' }],
        'body-medium': ['14px', { lineHeight: '20px' }],
        'body-small': ['12px', { lineHeight: '16px' }],
        'label-large': ['14px', { lineHeight: '20px' }],
        'label-medium': ['12px', { lineHeight: '16px' }],
        'label-small': ['11px', { lineHeight: '16px' }],
      },
      backgroundImage: {
        'gradient-ocean-turtle': 'linear-gradient(to bottom right, var(--tw-gradient-stops))',
      },
      gradientColorStops: theme => ({
        'ocean-start': theme('colors.primary.DEFAULT'),
        'turtle-end': theme('colors.secondary.DEFAULT'),
      }),
    },
  },
  plugins: [],
}

