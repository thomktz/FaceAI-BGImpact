const { defineConfig } = require("@vue/cli-service");
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: "0.0.0.0",
    port: 8082,
    allowedHosts: "all",
    server: process.env.VUE_APP_SERVER,
  },
});
