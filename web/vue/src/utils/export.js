/**
 * 导出数据为CSV文件
 * @param {Array} data - 要导出的数据数组
 * @param {string} filename - 文件名（不含扩展名）
 */
export function exportToCSV(data, filename = 'export') {
  if (!data || data.length === 0) {
    console.error('没有数据可以导出')
    return
  }

  // 获取表头
  const headers = Object.keys(data[0])

  // 构建CSV内容
  let csvContent = ''

  // 添加表头
  csvContent += headers.join(',') + '\n'

  // 添加数据行
  data.forEach(row => {
    const values = headers.map(header => {
      const value = row[header]
      // 处理特殊字符
      if (value === null || value === undefined) {
        return ''
      }
      const stringValue = String(value)
      // 如果值包含逗号、引号或换行符，用引号包裹
      if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
        return `"${stringValue.replace(/"/g, '""')}"`
      }
      return stringValue
    })
    csvContent += values.join(',') + '\n'
  })

  // 创建Blob并下载
  const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')

  if (navigator.msSaveBlob) {
    // IE10+
    navigator.msSaveBlob(blob, filename)
  } else {
    // 其他浏览器
    const url = URL.createObjectURL(blob)
    link.setAttribute('href', url)
    link.setAttribute('download', filename)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
}

/**
 * 导出数据为Excel文件（需要xlsx库）
 */
export function exportToExcel(data, filename = 'export', sheetName = 'Sheet1') {
  // 如果有xlsx库可以使用，否则提示用户安装
  console.warn('需要安装xlsx库来支持Excel导出')
  console.log('可以使用: npm install xlsx')

  // 降级为CSV导出
  exportToCSV(data, filename + '.csv')
}

export default {
  exportToCSV,
  exportToExcel
}
