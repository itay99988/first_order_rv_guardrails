/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Fira Code"', '"JetBrains Mono"', '"Source Code Pro"', 'monospace'],
        sans: ['"Fira Code"', '"JetBrains Mono"', 'monospace'],
      },
      colors: {
        dark: {
          primary: '#0a0a0a',
          secondary: '#0f0f0f',
          surface: '#141414',
          elevated: '#1a1a1a',
          hover: '#1f1f1f',
        },
        border: {
          subtle: '#1a1a1a',
          DEFAULT: '#262626',
          strong: '#333333',
        },
        accent: {
          DEFAULT: '#00ff41',
          hover: '#33ff66',
          muted: 'rgba(0, 255, 65, 0.08)',
          glow: 'rgba(0, 255, 65, 0.15)',
        },
        terminal: {
          green: '#00ff41',
          amber: '#ffb000',
          red: '#F7A291',
          cyan: '#00d4ff',
          dim: '#555555',
          text: '#b0b0b0',
          bright: '#e0e0e0',
        },
      },
      boxShadow: {
        glow: '0 0 20px rgba(0, 255, 65, 0.08)',
        'glow-sm': '0 0 8px rgba(0, 255, 65, 0.06)',
        'glow-accent': '0 0 12px rgba(0, 255, 65, 0.12)',
        card: '0 1px 2px rgba(0, 0, 0, 0.6)',
        'card-hover': '0 0 16px rgba(0, 255, 65, 0.06)',
      },
      animation: {
        'blink': 'blink 1s step-end infinite',
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-up': 'slideUp 0.15s ease-out',
        'scanline': 'scanline 8s linear infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
