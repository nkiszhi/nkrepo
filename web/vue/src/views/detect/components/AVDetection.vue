<template>
  <div class="section">
    <div v-if="isElLoading" class="isElLoading">
      <p>正在加载数据...</p>
    </div>
    <div v-else-if="!isErrors && results.length > 0">
      <h2 style="text-align: center;">杀毒软件检测结果</h2>

      <!-- 环形进度条统计区域 -->
      <div style="width: 200px; height: 240px; margin: 0 auto; text-align: center; margin-bottom: 20px;">
        <!-- 环形进度条容器 -->
        <div style="position: relative; width: 180px; height: 180px; margin: 0 auto;">
          <!-- 底色圆环 -->
          <svg width="180" height="180" viewBox="0 0 180 180">
            <!-- 底色圆环 -->
            <circle cx="90" cy="90" r="80" fill="none" stroke="#e5e5e5" stroke-width="10" />
            <!-- 进度圆环（根据恶意检测占比计算角度） -->
            <circle
              cx="90"
              cy="90"
              r="80"
              fill="none"
              :stroke="maliciousCount > 0 ? '#f56c6c' : '#67c23a'"
              stroke-width="10"
              stroke-dasharray="502.65"
              :stroke-dashoffset="502.65 - (502.65 * maliciousCount / (validDetectorCount || 1))"
              stroke-linecap="round"
              transform="rotate(-90 90 90)"
            />
          </svg>
          <!-- 中间数字 -->
          <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <span
              :style="{
                fontSize: '40px',
                fontWeight: 'bold',
                color: maliciousCount > 0 ? '#f56c6c' : '#67c23a'
              }"
            >
              {{ maliciousCount }}
            </span>
            <span style="font-size: 20px; color: #999; margin-left: 5px;">/{{ validDetectorCount }}</span>
          </div>
        </div>
        <!-- 下方小字说明 -->
        <p style="font-size: 14px; color: #666; margin-top: 10px; margin-bottom: 0;">
          {{
            maliciousCount > 0
              ? `${maliciousCount}/${validDetectorCount} 个安全厂商标记此文件为恶意`
              : `0/${validDetectorCount} 个安全厂商标记此文件为恶意`
          }}
        </p>
      </div>

      <table>
        <thead>
          <tr>
            <th style="width:10%;">序号</th>
            <th style="width:20%;">杀毒软件</th>
            <th style="width:20%;">版本号</th>
            <th style="width:15%;">更新日期</th>
            <th style="width:15%;">检测方法</th>
            <th style="width:20%;">检测结果</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(result, index) in results" :key="result.engine_name" class="vt_table-row">
            <td>{{ index + 1 }}</td>
            <td><span>{{ result.engine_name }}</span></td>
            <td><span>{{ result.engine_version }}</span></td>
            <td><span>{{ result.engine_update }}</span></td>
            <td><span>{{ result.method }}</span></td>
            <td style="text-align: left;">
              <span v-if="result.result && result.result !== ''" :class="getCategoryColorClass(result.category)">
                <svg-icon :icon-class="getIconClass(result.category)" /> {{ result.result }}
              </span>
              <span v-else-if="result.category" :class="getCategoryColorClass(result.category)">
                <svg-icon :icon-class="getIconClass(result.category)" /> {{ result.category }}
              </span>
              <span v-else>
                <i class="fa fa-times" aria-hidden="true" />N/A
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-else-if="isErrors">
      <p class="error-message">杀软检测数据加载失败</p>
    </div>
    <div v-else>
      <p class="error-message">暂无杀软检测数据</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AVDetection',
  props: {
    results: {
      type: Array,
      default: () => []
    },
    isElLoading: {
      type: Boolean,
      default: false
    },
    isErrors: {
      type: Boolean,
      default: false
    },
    validDetectorCount: {
      type: Number,
      default: 0
    },
    maliciousCount: {
      type: Number,
      default: 0
    }
  },
  methods: {
    getIconClass(category) {
      switch (category) {
        case 'malicious':
        case 'suspicious':
          return 'vt_malicious'
        case 'undetected':
        case 'harmless':
          return 'vt_undetected'
        default:
          return 'vt_type-unsupported'
      }
    },

    getCategoryColorClass(category) {
      switch (category) {
        case 'malicious':
        case 'suspicious':
          return 'red-text'
        case 'undetected':
        case 'harmless':
          return 'black-text'
        default:
          return 'gray-text'
      }
    }
  }
}
</script>

<style scoped>
table {
  width: 60%;
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse;
  margin-top: 30px;
  border-bottom: 1px solid;
}

table th,
table td {
  padding: 8px;
  text-align: center;
  border-bottom: 1px solid #ddd;
}

.vt_table-row:hover {
  background-color: #f0f0f0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.red-text {
  color: red;
}

.black-text {
  color: black;
}

.gray-text {
  color: gray;
}

.error-message {
  text-align: center;
  color: red;
  margin: 20px 0;
  font-size: 16px;
}

.isElLoading {
  text-align: center;
  padding: 30px 0;
  color: #666;
  font-size: 16px;
}
</style>
