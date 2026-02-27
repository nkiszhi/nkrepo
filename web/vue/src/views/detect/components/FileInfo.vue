<template>
  <div class="section">
    <table class="file-info-table">
      <tbody>
        <tr>
          <th>文件特征</th>
          <th />
        </tr>
        <tr>
          <td>文件名称：</td>
          <td>{{ uploadResult.original_filename }}</td>
        </tr>
        <tr>
          <td>文件大小：</td>
          <td>{{ uploadResult.file_size }}</td>
        </tr>
        <!-- 仅保留空值过滤，去掉nan/NaN判断 -->
        <template v-for="(value, key) in uploadResult.query_result">
          <tr
            v-if="
              value !== null &&
                value !== undefined &&
                (typeof value === 'string' ? value.trim() !== '' : value)
            "
            :key="key"
          >
            <td>{{ key.replace('_', ' ').replace(/^\w/, c => c.toUpperCase()) }}：</td>
            <td>{{ value }}</td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</template>

<script>
export default {
  name: 'FileInfo',
  props: {
    uploadResult: {
      type: Object,
      required: true
    }
  }
}
</script>

<style scoped>
.file-info-table {
  width: 60%;
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse;
  margin-top: 30px;
}

.file-info-table th,
.file-info-table td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.file-info-table tr:hover {
  background-color: #f5f5f5;
}
</style>
