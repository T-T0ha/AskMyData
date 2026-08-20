/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class', ':root[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: { extend: {} },
  plugins: [],
}
