module.exports = {
  content: [
    './src/web/templates/**/*.html',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'sans-serif'],
      },
      colors: {
        // Sandy Shore Colors
        'sand-light': '#f5e7d3',
        'sand-medium': '#e7d4b5',
        'sand-dark': '#d9c297',
        
        // Turtle Colors
        'turtle-green': '#4a7c59',
        'turtle-dark-green': '#3a5f42',
        'turtle-shell': '#8b9c6d',
        'turtle-skin': '#c4a484',
        
        // Ocean Colors
        'ocean-blue': '#6b9ac4',
        'ocean-dark-blue': '#4a7b9d',
        'ocean-light': '#a3c1d9',
        
        // Accent Colors
        'sunset-orange': '#f4a261',
        'coral': '#e76f51',
      }
    },
  },
  plugins: [],
  safelist: [
    'bg-turtle-green',
    'text-white',
    'bg-white',
    'bg-opacity-20',
    'hover:bg-opacity-30',
    'bg-coral',
    'hover:bg-opacity-80',
    'bg-ocean-blue',
    'bg-green-100',
    'text-green-800',
    'bg-red-100',
    'text-red-800',
  ]
}