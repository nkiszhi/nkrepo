<template>
  <div class="section">
    <table class="detection-result-table">
      <thead>
        <tr>
          <th>检测模型</th>
          <th>恶意概率</th>
          <th>结果</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(resultData,model) in uploadResult.exe_result" :key="model">
          <td>{{ model }}</td>
          <td>{{ resultData.probability }}</td>
          <td :style="{ color: resultData.result === '恶意' ? 'red' : 'inherit' }">
            {{ resultData.result }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
export default {
  name: 'ModelDetection',
  props: {
    uploadResult: {
      type: Object,
      required: true
    }
  }
}
</script>

<style scoped>
.detection-result-table {
  width: 60%;
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse;
  margin-top: 30px;
}

.detection-result-table th,
.detection-result-table td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.detection-result-table td:last-child {
  text-align: center;
}
</style>
