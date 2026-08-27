import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#140f1f",
        },
        plum: {
          50: "#f6f2ff",
          100: "#ede4ff",
          200: "#dbc7ff",
          300: "#c29eff",
          400: "#a86ef1",
          500: "#8c49df",
          600: "#7133be",
          700: "#5d2a9b",
          800: "#4a237b",
          900: "#33185b",
        },
      },
      boxShadow: {
        soft: "0 20px 60px rgba(72, 37, 116, 0.14)",
      },
      backgroundImage: {
        "clinical-gradient":
          "radial-gradient(circle at top, rgba(140, 73, 223, 0.18), transparent 40%), linear-gradient(180deg, #ffffff 0%, #faf7ff 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
