// vue/src/utils/exportExcel.js
import * as XLSX from 'xlsx'

/**
 * 导出JSON数据到Excel文件
 * @param {Array} jsonData - 要导出的JSON数据
 * @param {String} fileName - 文件名（不需要扩展名）
 * @param {Array} headers - 表头，格式：[{label: '显示名称', value: '字段名'}]
 */
export function exportJsonToExcel(jsonData, fileName = 'export', headers = null) {
  try {
    // 如果提供了headers，则重新组织数据
    let dataToExport = jsonData
    if (headers && Array.isArray(headers)) {
      dataToExport = jsonData.map(item => {
        const newItem = {}
        headers.forEach(header => {
          newItem[header.label] = item[header.value] || ''
        })
        return newItem
      })
    }

    // 创建工作簿
    const wb = XLSX.utils.book_new()

    // 创建工作表
    const ws = XLSX.utils.json_to_sheet(dataToExport)

    // 将工作表添加到工作簿
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1')

    // 生成Excel文件并下载
    XLSX.writeFile(wb, `${fileName}.xlsx`)

    return true
  } catch (error) {
    console.error('导出Excel失败:', error)
    return false
  }
}

/**
 * 导出CSV文件
 * @param {Array} jsonData - 要导出的JSON数据
 * @param {String} fileName - 文件名（不需要扩展名）
 */
export function exportToCsv(jsonData, fileName = 'export') {
  try {
    if (!jsonData || jsonData.length === 0) {
      console.warn('没有数据可导出')
      return false
    }

    // 获取表头
    const headers = Object.keys(jsonData[0])

    // 创建CSV内容
    let csvContent = headers.join(',') + '\n'

    // 添加数据行
    jsonData.forEach(item => {
      const row = headers.map(header => {
        const value = item[header]
        // 处理特殊字符
        if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value || ''
      })
      csvContent += row.join(',') + '\n'
    })

    // 创建Blob并下载
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)

    link.setAttribute('href', url)
    link.setAttribute('download', `${fileName}.csv`)
    link.style.visibility = 'hidden'

    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    return true
  } catch (error) {
    console.error('导出CSV失败:', error)
    return false
  }
}

/**
 * 导出ATT&CK矩阵数据
 * @param {Array} techniques - 技术列表
 * @param {String} exportType - 导出类型：excel 或 csv
 */
export function exportAttckMatrix(techniques, exportType = 'excel') {
  const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '')
  const fileName = `attck_matrix_${timestamp}`

  // 格式化数据
  const exportData = techniques.map(tech => ({
    '战术ID': tech.tactic_id,
    '战术名称': tech.tactic_name_cn,
    '技术ID': tech.technique_id,
    '技术名称': tech.technique_name,
    '函数数量': tech.function_count,
    '类型': tech.is_sub_technique ? '子技术' : '主技术',
    '父技术ID': tech.parent_technique_id || '',
    '子技术数量': tech.sub_techniques?.length || 0
  }))

  if (exportType === 'csv') {
    return exportToCsv(exportData, fileName)
  } else {
    return exportJsonToExcel(exportData, fileName)
  }
}
