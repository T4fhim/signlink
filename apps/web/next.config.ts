import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@signlnk/landmarks", "@signlnk/schemas"],
};

export default config;
