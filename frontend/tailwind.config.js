/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    {
      // 仅用于让 @neo4j-ndl/base 的 safelist 正则命中至少一组类，避免构建告警；
      // 不参与业务界面渲染，不改变现有样式体系。
      raw: "n-text-primary-100 hover:n-text-primary-100 n-bg-primary-100 hover:n-bg-primary-100 active:n-bg-primary-100 n-border-primary-100 hover:n-border-primary-100",
      extension: "html",
    },
  ],
  theme: {
    extend: {},
  },
  plugins: [],
  presets:[require('@neo4j-ndl/base').tailwindConfig],
  corePlugins: {
    preflight: false,
  },
  prefix:""
}

