import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { createSvgIconsPlugin } from 'vite-plugin-svg-icons'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // 加载环境变量
  const env = loadEnv(mode, process.cwd())
  const baseApi = env.VITE_APP_BASE_API || 'http://10.134.53.143:5005'
  
  console.log('🚀 API地址:', baseApi)
  
  return {
    plugins: [
      vue(),
      createSvgIconsPlugin({
        iconDirs: [path.resolve(process.cwd(), 'src/icons/svg')],
        symbolId: 'icon-[name]',
        inject: 'body-last',
        customDomId: '__svg__icons__dom__',
      }),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    server: {
      port: 9528,
      host: '0.0.0.0',
      open: true,
      proxy: {
        '/dev-api': {
          target: baseApi,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/dev-api/, '/dev-api'),
        },
        '/flowviz': {
          target: baseApi,
          changeOrigin: true,
          secure: false,
          ws: false,
          rewrite: (path) => path.replace(/^\/flowviz/, '/flowviz'),
        },
        '/api': {
          target: baseApi,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, '/api'),
        },
      },
    },
    build: {
      outDir: 'dist',
      assetsDir: 'static',
      sourcemap: false,
      rollupOptions: {
        output: {
          manualChunks: {
            'element-plus': ['element-plus'],
            'echarts': ['echarts', 'echarts-gl', 'echarts-map'],
            'vue-vendor': ['vue', 'vue-router', 'vuex'],
          },
        },
      },
      chunkSizeWarningLimit: 1500,
    },
    css: {
      preprocessorOptions: {
        scss: {
          additionalData: (content, filePath) => {
            // 如果是variables.scss文件,不添加导入
            if (filePath.includes('variables.scss')) {
              return content
            }
            // 其他文件添加variables导入
            return `@use "@/styles/variables.scss" as *;\n${content}`
          },
        },
      },
    },
  }
})
