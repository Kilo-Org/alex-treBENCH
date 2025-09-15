import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    DATABASE_URL: process.env.DATABASE_URL,
  },
  
  // Configure external packages for server components
  serverExternalPackages: ['pg'],
  
  // Configure webpack for better database client support
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Handle database client dependencies on server side
      config.externals.push('pg-native');
    }
    return config;
  },
};

export default nextConfig;
