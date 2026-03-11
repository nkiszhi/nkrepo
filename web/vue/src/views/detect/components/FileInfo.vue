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
        <!-- 根据数据库是否有数据显示不同字段 -->
        <template v-if="hasDatabaseInfo">
          <!-- 数据库有数据：显示完整信息 -->
          <tr v-if="uploadResult.query_result.MD5">
            <td>MD5：</td>
            <td>{{ uploadResult.query_result.MD5 }}</td>
          </tr>
          <tr v-if="uploadResult.query_result['SHA-256']">
            <td>SHA-256：</td>
            <td>{{ uploadResult.query_result['SHA-256'] }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.vhash">
            <td>Vhash：</td>
            <td>{{ uploadResult.query_result.vhash }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.Authentihash">
            <td>Authentihash：</td>
            <td>{{ uploadResult.query_result.Authentihash }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.Imphash">
            <td>Imphash：</td>
            <td>{{ uploadResult.query_result.Imphash }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.SSDEEP">
            <td>SSDEEP：</td>
            <td>{{ uploadResult.query_result.SSDEEP }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.类型">
            <td>类型：</td>
            <td>{{ uploadResult.query_result.类型 }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.平台">
            <td>平台：</td>
            <td>{{ uploadResult.query_result.平台 }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.家族">
            <td>家族：</td>
            <td>{{ uploadResult.query_result.家族 }}</td>
          </tr>
        </template>
        <template v-else>
          <!-- 数据库无数据：只显示基础信息 -->
          <tr v-if="uploadResult.query_result.MD5">
            <td>MD5：</td>
            <td>{{ uploadResult.query_result.MD5 }}</td>
          </tr>
          <tr v-if="uploadResult.query_result.SHA256">
            <td>SHA-256：</td>
            <td>{{ uploadResult.query_result.SHA256 }}</td>
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
  },
  computed: {
    // 判断数据库是否有详细信息
    hasDatabaseInfo() {
      const queryResult = this.uploadResult.query_result
      // 如果有类型、平台、家族等字段，说明数据库有详细信息
      return queryResult && (
        queryResult.类型 ||
        queryResult.平台 ||
        queryResult.家族 ||
        queryResult.vhash ||
        queryResult.Authentihash ||
        queryResult.Imphash ||
        queryResult.SSDEEP
      )
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
