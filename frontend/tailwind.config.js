/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#05070d",
          900: "#0a0e17",
          850: "#0d1220",
          800: "#111827",
          700: "#1c2333",
          600: "#2a3348",
        },
        accent: {
          500: "#3b82f6",
          400: "#60a5fa",
        },
        status: {
          ok: "#22c55e",
          warn: "#eab308",
          critical: "#ef4444",
          idle: "#64748b",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
