/** @type {import('next').NextConfig} */
const nextConfig = {
  // 隔离开发缓存与生产构建，避免运行 `next build` 时破坏正在使用的
  // `next dev` 静态资源映射，造成 CSS/JS 全部 404。
  distDir: process.env.NODE_ENV === "production" ? ".next-build" : ".next",
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
