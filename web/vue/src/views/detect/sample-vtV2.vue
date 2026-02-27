<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">基于可信度评估的多模型恶意文件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <!-- 检测模式选择 -->
    <div class="detection-mode-selector">
      <div class="mode-tabs">
        <button
          :class="['mode-tab', { 'active': detectionMode === 'file' }]"
          @click="switchMode('file')"
        >
          <svg-icon icon-class="file-upload" class="mode-icon" />
          <span>文件检测</span>
        </button>
        <button
          :class="['mode-tab', { 'active': detectionMode === 'sha256' }]"
          @click="switchMode('sha256')"
        >
          <svg-icon icon-class="hash" class="mode-icon" />
          <span>SHA256检测</span>
        </button>
      </div>
    </div>

    <!-- 文件检测模式 -->
    <div v-if="detectionMode === 'file'" class="file-detection-mode">
      <input
        id="file-upload-input"
        ref="file-upload-input"
        class="file-upload-input"
        type="file"
        @change="handleFileClick"
      >
      <div
        class="drop"
        @drop="handleDrop"
        @dragover="handleDragover"
        @click="handleUpload"
      >
        <div class="drop-content">
          <svg-icon icon-class="upload" class="upload-icon" />
          <div class="drop-text">
            把待检文件拖到这里或
            <el-button
              :loading="loading"
              style="font-size: 16px; margin-top: 10px;"
              size="default"
              type="primary"
              @click.stop="handleUpload"
            >
              选择待检文件
            </el-button>
          </div>
          <p class="file-hint">
            支持的文件类型：可执行文件（exe, dll, sys等）、文档文件（doc, docx, pdf, xls等）、压缩文件（zip, rar, 7z等）
          </p>
          <p class="file-hint">
            最大文件大小：100MB
          </p>
        </div>
      </div>
    </div>

    <!-- SHA256检测模式 -->
    <div v-if="detectionMode === 'sha256'" class="sha256-detection-mode">
      <div class="sha256-input-container">
        <div class="sha256-input-header">
          <svg-icon icon-class="hash" class="sha256-icon" />
          <span class="sha256-title">输入SHA256哈希值进行检测</span>
        </div>

        <div class="sha256-input-wrapper">
          <el-input
            v-model="sha256Input"
            type="text"
            placeholder="请输入64位SHA256哈希值（如：e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855）"
            :maxlength="64"
            clearable
            class="sha256-input"
            @keyup.enter="handleSha256Submit"
          >
            <template #prefix>
              <svg-icon icon-class="search" class="input-prefix-icon" />
            </template>
          </el-input>

          <div class="input-hint">
            <svg-icon icon-class="info" class="hint-icon" />
            <span>SHA256哈希值应为64位十六进制字符串（不区分大小写）</span>
          </div>
        </div>

        <div class="sha256-action-buttons">
          <el-button
            type="info"
            size="default"
            class="reset-button"
            @click="handleSha256Reset"
          >
            <svg-icon icon-class="refresh" class="button-icon" />
            重置
          </el-button>
          <el-button
            type="primary"
            size="default"
            :loading="sha256ElLoading"
            :disabled="!isValidSha256"
            class="submit-button"
            @click="handleSha256Submit"
          >
            <svg-icon icon-class="send" class="button-icon" />
            提交检测
          </el-button>
        </div>

        <!-- 哈希值示例 -->
        <div v-if="!sha256Input" class="sha256-examples">
          <div class="examples-title">SHA256示例：</div>
          <div class="example-item" @click="sha256Input = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'">
            <span class="example-hash">e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855</span>
            <span class="example-desc">（空文件的SHA256）</span>
          </div>
          <div class="example-item" @click="sha256Input = '7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069'">
            <span class="example-hash">7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069</span>
            <span class="example-desc">（字符串"hello world"的SHA256）</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 检测结果部分 - 添加滚动容器 -->
    <div v-if="uploadResult" class="report-container">
      <div style="text-align: center;" class="result-content">
        <svg-icon style="width: 30px; height: 30px;margin-right: 10px;margin-top:30px; " icon-class="detect-report" />
        <span class="result-status">检测报告如下</span>
      </div>

      <div
        style="display: flex; justify-content: space-between; align-items: center; margin-top:50px; width: 60%;margin: 0 auto"
      >
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'fileInfo' }"
          @click="showSection = 'fileInfo'"
        >
          <svg-icon icon-class="fileInfo" class="button-icon" />
          <span> 基础信息</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'modelDetection' }"
          @click="showSection = 'modelDetection'"
        >
          <svg-icon icon-class="modelDetection" class="button-icon" />
          <span> 模型检测</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'AV-Detection' }"
          @click="showSection = 'AV-Detection'"
        >
          <svg-icon icon-class="AV-Detection" class="button-icon" />
          <span> 杀软检测</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'DynamicDetection' }"
          @click="showSection = 'DynamicDetection'"
        >
          <svg-icon icon-class="DynamicDetection" class="button-icon" />
          <span> 动态检测</span>
        </button>
      </div>

      <!-- 上传文件基本信息 -->
      <div v-show="showSection === 'fileInfo'" class="section">
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

      <!-- 模型检测结果 -->
      <div v-show="showSection === 'modelDetection'" class="section">
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

      <!-- 杀毒软件检测结果 -->
      <div v-show="showSection === 'AV-Detection'" class="section">
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

      <!-- 动态检测结果 -->
      <div v-show="showSection === 'DynamicDetection'" class="section">
        <div v-if="isElLoadings" class="isElLoading">
          <p>正在加载数据...</p>
        </div>

        <!-- 失败或空数据状态 -->
        <div v-if="!isElLoadings && (isError || !hasValidData)">
          <h2 style="text-align: center;">无动态检测结果</h2>
          <p v-if="isError" style="text-align: center;">未检测到动态行为数据</p>
          <p v-else style="text-align: center;">未检测到任何动态行为数据</p>
        </div>

        <div v-if="!isElLoadings && !isError && hasValidData">
          <h2 style="text-align: center;">动态检测结果</h2>
          <!-- API调用情况 -->
          <div
            v-if="behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0"
            style="text-align: center;"
          >
            <h3 :title="toggleCall ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleCall = !toggleCall">
              API调用情况:{{ behaviour_results.calls_highlighted ? behaviour_results.calls_highlighted.length : 0 }}个API调用
            </h3>
            <div
              v-if="toggleCall &&behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0"
            >
              <ul class="flex-container">
                <li
                  v-for="(calls_highlighted, index) in behaviour_results.calls_highlighted"
                  :key="index"
                  style="text-align: left;"
                  class="list-row"
                >
                  {{ calls_highlighted }}
                </li>
              </ul>
            </div>
          </div>

          <!-- 服务情况 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0) ||
                (behaviour_results && behaviour_results.services_started && behaviour_results.services_started.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleServicesOpened ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleServicesOpened = !toggleServicesOpened"
            >
              服务情况:{{ behaviour_results.services_opened ? behaviour_results.services_opened.length : 0 }}个打开的服务；{{
                behaviour_results.services_started ? behaviour_results.services_started.length : 0 }} 个启动的服务
            </h3>

            <div
              v-if="toggleServicesOpened &&behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0"
            >
              <h4>打开的服务</h4>
              <ul class="flex-container">
                <li v-for="(services_opened, index) in behaviour_results.services_opened" :key="index" class="list-row">
                  {{ services_opened }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleServicesOpened &&behaviour_results && behaviour_results.services_started && behaviour_results.services_started.length > 0"
            >
              <h4>启动的服务</h4>
              <ul class="flex-container">
                <li
                  v-for="(services_started, index) in behaviour_results.services_started"
                  :key="index"
                  class="list-row"
                >
                  {{ services_started }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else>
            <h3>服务情况:未检测到动态行为</h3>
          </div>

          <!-- 文件行为 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.command_executions && behaviour_results.command_executions.length > 0) ||
                (behaviour_results && behaviour_results.files_attribute_changed && behaviour_results.files_attribute_changed.length > 0) ||
                (behaviour_results && behaviour_results.files_copied && behaviour_results.files_copied.length > 0) ||
                (behaviour_results && behaviour_results.files_deleted && behaviour_results.files_deleted.length > 0)||
                (behaviour_results && behaviour_results.files_dropped && behaviour_results.files_dropped.length > 0)||
                (behaviour_results && behaviour_results.files_opened && behaviour_results.files_opened.length > 0)||
                (behaviour_results && behaviour_results.files_written && behaviour_results.files_written.length > 0)
            "
            style="text-align: center;"
          >
            <h3 :title="toggleFiles ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleFiles = !toggleFiles">
              文件行为:{{ behaviour_results.command_executions ? behaviour_results.command_executions.length : 0 }}个执行；{{
                behaviour_results.files_attribute_changed ? behaviour_results.files_attribute_changed.length : 0 }}
              个属性变更；{{ behaviour_results.files_copied ? behaviour_results.files_copied.length : 0 }} 个复制；{{
                behaviour_results.files_deleted ? behaviour_results.files_deleted.length : 0 }} 个删除；{{
                behaviour_results.files_dropped ? behaviour_results.files_dropped.length : 0 }} 个释放；{{
                behaviour_results.files_opened ? behaviour_results.files_opened.length : 0 }} 个打开；{{
                behaviour_results.files_written ? behaviour_results.files_written.length : 0 }} 个写入</h3>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.command_executions && behaviour_results.command_executions.length > 0"
            >
              <h4>文件执行情况</h4>
              <ul class="flex-container">
                <li
                  v-for="(path, index) in behaviour_results.command_executions"
                  :key="index"
                  style="text-align: left;"
                  class="list-row"
                >
                  {{ path }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_attribute_changed && behaviour_results.files_attribute_changed.length > 0"
            >
              <h4>文件属性变更情况</h4>
              <ul class="flex-container">
                <li
                  v-for="(files_attribute, index) in behaviour_results.files_attribute_changed"
                  :key="index"
                  class="list-row"
                >
                  {{ files_attribute }}
                </li>
              </ul>
            </div>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_copied && behaviour_results.files_copied.length > 0"
            >
              <h4>文件复制情况</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:40%;">源文件</th>
                    <th style="width:40%;">目标文件</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(files_copied, index) in behaviour_results.files_copied"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ files_copied.key }}</span></td>
                    <td><span>{{ files_copied.value }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_deleted && behaviour_results.files_deleted.length > 0"
            >
              <h4>文件删除情况</h4>
              <ul class="flex-container">
                <li v-for="(path_del, index) in behaviour_results.files_deleted" :key="index" class="list-row">
                  {{ path_del }}
                </li>
              </ul>
            </div>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_dropped && behaviour_results.files_dropped.length > 0"
            >
              <h4>文件释放情况</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:35%;">路径</th>
                    <th style="width:35%;">哈希值</th>
                    <th style="width:20%;">文件类型</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(files_dropped, index) in behaviour_results.files_dropped"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ files_dropped.path }}</span></td>
                    <td><span>{{ files_dropped.sha256 }}</span></td>
                    <td><span>{{ files_dropped.type }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_opened && behaviour_results.files_opened.length > 0"
            >
              <h4>文件打开情况</h4>
              <ul class="flex-container">
                <li v-for="(path, index) in behaviour_results.files_opened" :key="index" class="list-row">
                  {{ path }}
                </li>
              </ul>
            </div>
            <div
              v-if="toggleFiles && behaviour_results && behaviour_results.files_written && behaviour_results.files_written.length > 0"
            >
              <h4>文件写入情况</h4>
              <ul class="flex-container">
                <li v-for="(path, index) in behaviour_results.files_written" :key="index" class="list-row">
                  {{ path }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else>
            <h3>文件行为:未检测到动态行为</h3>
          </div>

          <!-- 进程行为 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.modules_loaded && behaviour_results.modules_loaded.length > 0) ||
                (behaviour_results && behaviour_results.mutexes_created && behaviour_results.mutexes_created.length > 0) ||
                (behaviour_results && behaviour_results.mutexes_opened && behaviour_results.mutexes_opened.length > 0) ||
                (behaviour_results && behaviour_results.permissions_requested && behaviour_results.permissions_requested.length > 0)||
                (behaviour_results && behaviour_results.processes_terminated && behaviour_results.processes_terminated.length > 0)||
                (behaviour_results && behaviour_results.processes_tree && behaviour_results.processes_tree.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleProcesses ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleProcesses = !toggleProcesses"
            >
              进程行为:{{ behaviour_results.modules_loaded ? behaviour_results.modules_loaded.length : 0 }}个模块加载；{{
                behaviour_results.mutexes_created ? behaviour_results.mutexes_created.length : 0 }} 个互斥锁创建；{{
                behaviour_results.mutexes_opened ? behaviour_results.mutexes_opened.length : 0 }} 个互斥锁打开；{{
                behaviour_results.permissions_requested ? behaviour_results.permissions_requested.length : 0 }} 个权限请求；{{
                behaviour_results.processes_terminated ? behaviour_results.processes_terminated.length : 0 }} 个进程终止；{{
                behaviour_results.processes_tree ? behaviour_results.processes_tree.length : 0 }} 进程树</h3>
            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.modules_loaded && behaviour_results.modules_loaded.length > 0"
            >
              <h4>模块加载</h4>
              <ul class="flex-container">
                <li v-for="(module, index) in behaviour_results.modules_loaded" :key="index" class="list-row">
                  {{ module }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.mutexes_created && behaviour_results.mutexes_created.length > 0"
            >
              <h4>互斥锁创建</h4>
              <ul class="flex-container">
                <li v-for="(mutexes_created, index) in behaviour_results.mutexes_created" :key="index" class="list-row">
                  {{ mutexes_created }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.mutexes_opened && behaviour_results.mutexes_opened.length > 0"
            >
              <h4>互斥锁打开</h4>
              <ul class="flex-container">
                <li v-for="(mutexes_opened, index) in behaviour_results.mutexes_opened" :key="index" class="list-row">
                  {{ mutexes_opened }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.permissions_requested && behaviour_results.permissions_requested.length > 0"
            >
              <h4>权限请求</h4>
              <ul class="flex-container">
                <li
                  v-for="(permissions_requested, index) in behaviour_results.permissions_requested"
                  :key="index"
                  class="list-row"
                >
                  {{ permissions_requested }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.processes_terminated && behaviour_results.processes_terminated.length > 0"
            >
              <h4>进程终止</h4>
              <ul class="flex-container">
                <li
                  v-for="(processes_terminated, index) in behaviour_results.processes_terminated"
                  :key="index"
                  class="list-row"
                >
                  {{ processes_terminated }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleProcesses && behaviour_results && behaviour_results.processes_tree && behaviour_results.processes_tree.length > 0"
            >
              <h4>进程树</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:20%;">程序id</th>
                    <th style="width:30%;">程序名称</th>
                    <th style="width:40%;">子程序</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(processes_tree, index) in behaviour_results.processes_tree"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ processes_tree.process_id }}</span></td>
                    <td><span>{{ processes_tree.name }}</span></td>
                    <td><span>{{ processes_tree.children }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div v-else>
            <h3>进程行为:未检测到动态行为</h3>
          </div>

          <!-- 网络行为 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.dns_lookups && behaviour_results.dns_lookups.length > 0) ||
                (behaviour_results && behaviour_results.http_conversations && behaviour_results.http_conversations.length > 0) ||
                (behaviour_results && behaviour_results.ip_traffic && behaviour_results.ip_traffic.length > 0) ||
                (behaviour_results && behaviour_results.tls && behaviour_results.tls.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleNetwork ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleNetwork = !toggleNetwork"
            >
              网络行为:{{ behaviour_results.dns_lookups ? behaviour_results.dns_lookups.length : 0 }}个DNS查找；{{
                behaviour_results.http_conversations ? behaviour_results.http_conversations.length : 0 }} 个HTTP会话；{{
                behaviour_results.ip_traffic ? behaviour_results.ip_traffic.length : 0 }} 个IP流量；{{ behaviour_results.tls ?
                behaviour_results.tls.length : 0 }} 传输层安全协议</h3>
            <div
              v-if="toggleNetwork && behaviour_results && behaviour_results.dns_lookups && behaviour_results.dns_lookups.length > 0"
            >
              <h4>DNS查找记录</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:30%;">主机名</th>
                    <th style="width:60%;">IP地址</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(dns_lookups, index) in behaviour_results.dns_lookups"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ dns_lookups.hostname }}</span></td>
                    <td><span>{{ dns_lookups.resolved_ips }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleNetwork && behaviour_results && behaviour_results.http_conversations && behaviour_results.http_conversations.length > 0"
            >
              <h4>HTTP会话</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:30%;">网址</th>
                    <th style="width:20%;">请求方法</th>
                    <th style="width:30%;">请求头</th>
                    <th style="width:10%;">响应状态码</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(http_conversations, index) in behaviour_results.http_conversations"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ http_conversations.url }}</span></td>
                    <td><span>{{ http_conversations.request_method }}</span></td>
                    <td><span>{{ http_conversations.request_headers }}</span></td>
                    <td><span>{{ http_conversations.response_status_code }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleNetwork && behaviour_results && behaviour_results.ip_traffic && behaviour_results.ip_traffic.length > 0"
            >
              <h4>IP流量</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:30%;">目标IP地址</th>
                    <th style="width:30%;">目标端口</th>
                    <th style="width:30%;">传输层协议</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(ip_traffic, index) in behaviour_results.ip_traffic"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ ip_traffic.destination_ip }}</span></td>
                    <td><span>{{ ip_traffic.destination_port }}</span></td>
                    <td><span>{{ ip_traffic.transport_layer_protocol }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div v-if="toggleNetwork && behaviour_results && behaviour_results.tls && behaviour_results.tls.length > 0">
              <h4>传输层安全协议</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:20%;">身份信息</th>
                    <th style="width:10%;">颁发者</th>
                    <th style="width:10%;">序列号</th>
                    <th style="width:10%;">哈希值</th>
                    <th style="width:10%;">版本号</th>
                    <th style="width:10%;">服务器</th>
                    <th style="width:10%;">ja3</th>
                    <th style="width:10%;">ja3s</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(tls, index) in behaviour_results.tls" :key="index" class="behaviour_result_table_row">
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ tls.subject }}</span></td>
                    <td><span>{{ tls.issuer }}</span></td>
                    <td><span>{{ tls.serial_number }}</span></td>
                    <td><span>{{ tls.thumbprint }}</span></td>
                    <td><span>{{ tls.version }}</span></td>
                    <td><span>{{ tls.sni }}</span></td>
                    <td><span>{{ tls.ja3 }}</span></td>
                    <td><span>{{ tls.ja3s }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div v-else>
            <h3>网络行为:未检测到动态行为</h3>
          </div>

          <!-- 攻击行为 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.verdicts && behaviour_results.verdicts.length > 0) ||
                (behaviour_results && behaviour_results.attack_techniques && behaviour_results.attack_techniques.length > 0 )||
                (behaviour_results && behaviour_results.ids_alerts && behaviour_results.ids_alerts.length > 0)||
                (behaviour_results && behaviour_results.mbc && behaviour_results.mbc.length > 0)||
                (behaviour_results && behaviour_results.mitre_attack_techniques && behaviour_results.mitre_attack_techniques.length > 0)||
                (behaviour_results && behaviour_results.signature_matches && behaviour_results.signature_matches.length > 0)||
                (behaviour_results && behaviour_results.system_property_lookups && behaviour_results.system_property_lookups.length > 0)"
            style="text-align: center;"
          >
            <h3
              :title="toggleAttack ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleAttack = !toggleAttack"
            >
              攻击行为:{{ behaviour_results.verdicts ? behaviour_results.verdicts.length : 0 }} 个判决结果；{{
                Object.keys(behaviour_results.attack_techniques || {}).length }} 个攻击技术；{{ behaviour_results.ids_alerts ?
                behaviour_results.ids_alerts.length : 0 }} 个入侵警报；{{ behaviour_results.mbc ? behaviour_results.mbc.length :
                0 }} 个mbc；{{ behaviour_results.mitre_attack_techniques ? behaviour_results.mitre_attack_techniques.length
                : 0 }} 个MITRE ATT&CK技术；{{ behaviour_results.signature_matches ? behaviour_results.signature_matches.length
                : 0 }} 个签名;{{ behaviour_results.system_property_lookups ? behaviour_results.system_property_lookups.length
                : 0 }} 个系统属性查找</h3>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.verdicts && behaviour_results.verdicts.length > 0"
            >
              <h4>判决结果</h4>
              <ul class="flex-container">
                <li v-for="(verdicts, index) in behaviour_results.verdicts" :key="index" class="list-row">
                  {{ verdicts }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.verdict_confidence && behaviour_results.verdict_confidence.length > 0"
            >
              <h4>置信度{{ behaviour_results.verdict_confidence }}</h4>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.attack_techniques && Object.keys(behaviour_results.attack_techniques).length > 0"
            >
              <h4>攻击技术</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:20%;">技术ID</th>
                    <th style="width:20%;">严重性</th>
                    <th style="width:50%;">描述</th>
                  </tr>
                </thead>
                <tbody>
                  <template v-for="(techItems, techId, index) in behaviour_results.attack_techniques">
                    <tr
                      v-for="(item, itemIndex) in techItems"
                      :key="techId+'-'+itemIndex"
                      class="behaviour_result_table_row"
                    >
                      <td>{{ index + 1 }}</td>
                      <td><span>{{ techId }}</span></td>
                      <td><span>{{ item.severity }}</span></td>
                      <td><span>{{ item.description }}</span></td>
                    </tr>
                  </template>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.ids_alerts && behaviour_results.ids_alerts.length > 0"
            >
              <h4>入侵检测系统（IDS）警报</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:10%;">规则消息</th>
                    <th style="width:10%;">规则类型</th>
                    <th style="width:10%;">规则标识符</th>
                    <th style="width:10%;">告警严重性</th>
                    <th style="width:10%;">规则来源</th>
                    <th style="width:10%;">告警上下文</th>
                    <th style="width:10%;">规则URL</th>
                    <th style="width:10%;">规则引用</th>
                    <th style="width:10%;">规则原始数据</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(ids_alerts, index) in behaviour_results.ids_alerts"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ ids_alerts.rule_msg }}</span></td>
                    <td><span>{{ ids_alerts.rule_category }}</span></td>
                    <td><span>{{ ids_alerts.rule_id }}</span></td>
                    <td><span>{{ ids_alerts.alert_severity }}</span></td>
                    <td><span>{{ ids_alerts.rule_source }}</span></td>
                    <td><span>{{ ids_alerts.alert_context }}</span></td>
                    <td><span>{{ ids_alerts.rule_url }}</span></td>
                    <td><span>{{ ids_alerts.rule_references }}</span></td>
                    <td><span>{{ ids_alerts.rule_raw }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div v-if="toggleAttack && behaviour_results && behaviour_results.mbc && behaviour_results.mbc.length > 0">
              <h4>mbc攻击</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:10%;">id</th>
                    <th style="width:20%;">目标</th>
                    <th style="width:20%;">行为</th>
                    <th style="width:40%;">参考资料</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(mbc, index) in behaviour_results.mbc" :key="index" class="behaviour_result_table">
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ mbc.id }}</span></td>
                    <td><span>{{ mbc.objective }}</span></td>
                    <td><span>{{ mbc.behavior }}</span></td>
                    <td><span>{{ mbc.refs }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.mitre_attack_techniques && behaviour_results.mitre_attack_techniques.length > 0"
            >
              <h4>MITRE ATT&CK技术</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:10%;">id</th>
                    <th style="width:20%;">严重性</th>
                    <th style="width:20%;">签名</th>
                    <th style="width:40%;">参考资料</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(mitre_attack_techniques, index) in behaviour_results.mitre_attack_techniques"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ mitre_attack_techniques.id }}</span></td>
                    <td><span>{{ mitre_attack_techniques.severity }}</span></td>
                    <td><span>{{ mitre_attack_techniques.signature_description }}</span></td>
                    <td><span>{{ mitre_attack_techniques.refs }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.signature_matches && behaviour_results.signature_matches.length > 0"
            >
              <h4>签名匹配</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:10%;">格式</th>
                    <th style="width:30%;">名称</th>
                    <th style="width:20%;">作者</th>
                    <th style="width:30%;">来源</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(signature_matches, index) in behaviour_results.signature_matches"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ signature_matches.format }}</span></td>
                    <td><span>{{ signature_matches.name }}</span></td>
                    <td><span>{{ signature_matches.authors }}</span></td>
                    <td><span>{{ signature_matches.rule_src }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleAttack && behaviour_results && behaviour_results.system_property_lookups && behaviour_results.system_property_lookups.length > 0"
            >
              <h4>系统属性</h4>
              <ul class="flex-container">
                <li
                  v-for="(system_property_lookups, index) in behaviour_results.system_property_lookups"
                  :key="index"
                  class="list-row"
                >
                  {{ system_property_lookups }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else>
            <h3>攻击行为:未检测到动态行为</h3>
          </div>

          <!-- 内存情况 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.memory_dumps && behaviour_results.memory_dumps.length > 0) ||
                (behaviour_results && behaviour_results.memory_pattern_domains && behaviour_results.memory_pattern_domains.length > 0) ||
                (behaviour_results && behaviour_results.memory_pattern_urls && behaviour_results.memory_pattern_urls.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleMemory ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleMemory = !toggleMemory"
            >
              内存情况:{{ behaviour_results.memory_dumps ? behaviour_results.memory_dumps.length : 0 }} 个内容转储；{{
                behaviour_results.memory_pattern_domains ? behaviour_results.memory_pattern_domains.length : 0 }} 个域名模式；{{
                behaviour_results.memory_pattern_urls ? behaviour_results.memory_pattern_urls.length : 0 }} 个URL模式</h3>
            <div
              v-if="toggleMemory && behaviour_results && behaviour_results.memory_dumps && behaviour_results.memory_dumps.length > 0"
            >
              <h4>内存转储</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:15%;">文件名</th>
                    <th style="width:20%;">进程</th>
                    <th style="width:10%;">大小</th>
                    <th style="width:20%;">基地址</th>
                    <th style="width:15%;">阶段</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(memory_dumps, index) in behaviour_results.memory_dumps"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ memory_dumps.file_name }}</span></td>
                    <td><span>{{ memory_dumps.process }}</span></td>
                    <td><span>{{ memory_dumps.size }}</span></td>
                    <td><span>{{ memory_dumps.base_address }}</span></td>
                    <td><span>{{ memory_dumps.stage }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div
              v-if="toggleMemory && behaviour_results && behaviour_results.memory_pattern_domains && behaviour_results.memory_pattern_domains.length > 0"
            >
              <h4>内存中的域名模式</h4>
              <ul class="flex-container">
                <li
                  v-for="(memory_pattern_domains, index) in behaviour_results.memory_pattern_domains"
                  :key="index"
                  class="list-row"
                >
                  {{ memory_pattern_domains }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleMemory && behaviour_results && behaviour_results.memory_pattern_urls && behaviour_results.memory_pattern_urls.length > 0"
            >
              <h4>内存中的URL模式</h4>
              <ul class="flex-container">
                <li
                  v-for="(memory_pattern_urls, index) in behaviour_results.memory_pattern_urls"
                  :key="index"
                  class="list-row"
                >
                  {{ memory_pattern_urls }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else>
            <h3>内存行为:未检测到动态行为</h3>
          </div>

          <!-- 注册表 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.registry_keys_deleted && behaviour_results.registry_keys_deleted.length > 0) ||
                (behaviour_results && behaviour_results.registry_keys_opened && behaviour_results.registry_keys_opened.length > 0) ||
                (behaviour_results && behaviour_results.registry_keys_set && behaviour_results.registry_keys_set.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleRegistry ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleRegistry = !toggleRegistry"
            >
              注册表:{{ behaviour_results.registry_keys_deleted ? behaviour_results.registry_keys_deleted.length : 0 }}
              个注册表删除；{{ behaviour_results.registry_keys_opened ? behaviour_results.registry_keys_opened.length : 0 }}
              个注册表打开；{{ behaviour_results.registry_keys_set ? behaviour_results.registry_keys_set.length : 0 }} 个注册表设置
            </h3>

            <div
              v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_deleted && behaviour_results.registry_keys_deleted.length > 0"
            >
              <h4>注册表删除</h4>
              <ul class="flex-container">
                <li
                  v-for="(registry_keys_deleted, index) in behaviour_results.registry_keys_deleted"
                  :key="index"
                  class="list-row"
                >
                  {{ registry_keys_deleted }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_opened && behaviour_results.registry_keys_opened.length > 0"
            >
              <h4>注册表打开</h4>
              <ul class="flex-container">
                <li
                  v-for="(registry_keys_opened, index) in behaviour_results.registry_keys_opened"
                  :key="index"
                  class="list-row"
                >
                  {{ registry_keys_opened }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_set && behaviour_results.registry_keys_set.length > 0"
            >
              <h4>注册表设置</h4>
              <table class="behaviour_result_table">
                <thead>
                  <tr>
                    <th style="width:10%;">序号</th>
                    <th style="width:40%;">key</th>
                    <th style="width:50%;">value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(registry_keys_set, index) in behaviour_results.registry_keys_set"
                    :key="index"
                    class="behaviour_result_table_row"
                  >
                    <td>{{ index + 1 }}</td>
                    <td><span>{{ registry_keys_set.key }}</span></td>
                    <td :style="{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }">
                      <span>{{ registry_keys_set.value }}</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div v-else>
            <h3>注册表:未检测到动态行为</h3>
          </div>

          <!-- 加密情况 -->
          <div
            v-if="
              (behaviour_results && behaviour_results.crypto_algorithms_observed && behaviour_results.crypto_algorithms_observed.length > 0) ||
                (behaviour_results && behaviour_results.crypto_plain_text && behaviour_results.crypto_plain_text.length > 0) ||
                (behaviour_results && behaviour_results.text_highlighted && behaviour_results.text_highlighted.length > 0)
            "
            style="text-align: center;"
          >
            <h3
              :title="toggleCrypto ? '点击收起' : '点击显示详情'"
              style="cursor: pointer;"
              @click="toggleCrypto = !toggleCrypto"
            >
              加密情况:{{ behaviour_results.crypto_algorithms_observed ? behaviour_results.crypto_algorithms_observed.length
                : 0 }} 个加密算法；{{ behaviour_results.crypto_plain_text ? behaviour_results.crypto_plain_text.length : 0 }}
              个加密明文；{{ behaviour_results.text_highlighted ? behaviour_results.text_highlighted.length : 0 }} 个高亮文本</h3>

            <div
              v-if="toggleCrypto && behaviour_results && behaviour_results.crypto_algorithms_observed && behaviour_results.crypto_algorithms_observed.length > 0"
            >
              <h4>加密算法</h4>
              <ul class="flex-container">
                <li
                  v-for="(crypto_algorithms_observed, index) in behaviour_results.crypto_algorithms_observed"
                  :key="index"
                  class="list-row"
                >
                  {{ crypto_algorithms_observed }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleCrypto && behaviour_results && behaviour_results.crypto_plain_text && behaviour_results.crypto_plain_text.length > 0"
            >
              <h4>加密明文</h4>
              <ul class="flex-container">
                <li
                  v-for="(crypto_plain_text, index) in behaviour_results.crypto_plain_text"
                  :key="index"
                  class="list-row"
                >
                  {{ crypto_plain_text }}
                </li>
              </ul>
            </div>

            <div
              v-if="toggleCrypto && behaviour_results && behaviour_results.text_highlighted && behaviour_results.text_highlighted.length > 0"
            >
              <h4>高亮文本</h4>
              <ul class="flex-container">
                <li
                  v-for="(text_highlighted, index) in behaviour_results.text_highlighted"
                  :key="index"
                  class="list-row"
                >
                  {{ text_highlighted }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else>
            <h3>加密情况:未检测到动态行为</h3>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script>
import axios from 'axios'

// 创建axios实例（设置10分钟超时，统一处理Token）
const apiService = axios.create({
  timeout: 600000, // 10分钟超时（600秒）
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器：自动携带Token
apiService.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  error => Promise.reject(error)
)

export default {
  data() {
    return {
      // 检测模式：'file' 或 'sha256'
      detectionMode: 'file',

      // SHA256检测相关变量
      sha256Input: '',
      sha256ElLoading: false,

      // 原有变量保持不变
      showSection: 'fileInfo',
      loading: false,
      isElLoading: false,
      isElLoadings: false,
      results: [],
      behaviour_results: null,
      uploadResult: null,
      error: null,
      toggleServicesOpened: false,
      toggleCall: false,
      toggleFiles: false,
      toggleProcesses: false,
      toggleNetwork: false,
      toggleAttack: false,
      toggleMemory: false,
      toggleRegistry: false,
      toggleCrypto: false,
      isError: false,
      isErrors: false,
      apiBaseUrl: 'http://xxxx:5005' // 默认API地址
    }
  },

  computed: {
    // 验证SHA256输入是否有效
    isValidSha256() {
      if (!this.sha256Input) return false
      // SHA256应为64位十六进制字符串
      const sha256Regex = /^[a-fA-F0-9]{64}$/
      return sha256Regex.test(this.sha256Input.trim())
    },

    // 原有计算属性保持不变
    hasValidData() {
      if (!this.behaviour_results || typeof this.behaviour_results !== 'object') {
        return false
      }

      const r = this.behaviour_results
      return [
        'calls_highlighted', 'services_opened', 'services_started',
        'command_executions', 'files_attribute_changed', 'files_copied',
        'files_deleted', 'files_dropped', 'files_opened', 'files_written',
        'modules_loaded', 'mutexes_created', 'mutexes_opened',
        'permissions_requested', 'processes_terminated', 'processes_tree',
        'dns_lookups', 'http_conversations', 'ip_traffic', 'tls',
        'verdicts', 'attack_techniques', 'ids_alerts', 'mbc',
        'mitre_attack_techniques', 'signature_matches',
        'system_property_lookups',
        'memory_dumps', 'memory_pattern_domains', 'memory_pattern_urls',
        'registry_keys_deleted', 'registry_keys_opened', 'registry_keys_set',
        'crypto_algorithms_observed', 'crypto_plain_text', 'text_highlighted'
      ].some(key => r[key] && r[key].length > 0)
    },

    // 杀软检测统计
    unsupportedCount() {
      return this.results.filter(item => item.category === 'type-unsupported').length
    },
    validDetectorCount() {
      return this.results.length - this.unsupportedCount
    },
    maliciousCount() {
      return this.results.filter(item =>
        item.category &&
        item.category !== 'type-unsupported' &&
        item.category !== 'undetected' &&
        item.category !== 'harmless'
      ).length
    },
    harmlessCount() {
      return this.validDetectorCount - this.maliciousCount
    }
  },

  created() {
    // 组件创建时加载配置文件（保留原有逻辑）
    this.loadConfig()
  },

  methods: {
    // 新增：模式切换方法
    switchMode(mode) {
      // 清除检测结果
      this.uploadResult = null
      this.results = []
      this.behaviour_results = null
      this.sha256Input = ''

      // 重置所有切换状态
      this.toggleServicesOpened = false
      this.toggleCall = false
      this.toggleFiles = false
      this.toggleProcesses = false
      this.toggleNetwork = false
      this.toggleAttack = false
      this.toggleMemory = false
      this.toggleRegistry = false
      this.toggleCrypto = false

      // 切换到指定模式
      this.detectionMode = mode
      this.showSection = 'fileInfo'
    },

    // SHA256检测相关方法
    handleSha256Reset() {
      this.sha256Input = ''
      this.uploadResult = null
      this.results = []
      this.behaviour_results = null
    },

    async handleSha256Submit() {
      if (!this.isValidSha256) {
        this.$message.error('请输入有效的64位SHA256哈希值')
        return
      }

      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成，请稍后重试')
        return
      }

      this.sha256ElLoading = true
      this.uploadResult = null

      try {
        // 调用后端SHA256检测接口
        const response = await apiService.post(
          `${this.apiBaseUrl}/detect_by_sha256`,
          {
            sha256: this.sha256Input.trim().toLowerCase()
          },
          {
            timeout: 600000
          }
        )

        console.log('SHA256检测响应:', response.data)

        if (response.data && response.data.success) {
          // 假设后端返回的数据结构与文件上传类似
          this.uploadResult = response.data.data
          await this.fetchDetailAPI()
          this.$message.success('SHA256检测成功')
        } else {
          this.$message.error(response.data.message || 'SHA256检测失败')
        }
      } catch (error) {
        console.error('SHA256检测错误:', error)
        let errMsg = 'SHA256检测失败！'
        if (error.code === 'ECONNABORTED') {
          errMsg = '检测请求超时'
        } else if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录'
        } else if (error.response?.status === 404) {
          errMsg = '未找到该SHA256对应的文件信息'
        } else if (error.message) {
          errMsg += ' ' + error.message
        }
        this.$message.error(errMsg)
      } finally {
        this.sha256ElLoading = false
      }
    },

    // 修复文件上传相关方法
    handleFileClick(e) {
      const files = e.target.files
      const rawFile = files[0]
      if (!rawFile) return
      this.upload(rawFile)
    },

    // 修复上传按钮点击事件
    handleUpload() {
      // 直接触发隐藏的文件input点击
      const fileInput = document.getElementById('file-upload-input')
      if (fileInput) {
        fileInput.click()
      } else {
        console.error('找不到文件上传输入框')
        this.$message.error('文件上传功能异常，请刷新页面重试')
      }
    },

    // 原有方法保持不变
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000
        })

        const configContent = response.data
        const lines = configContent.split('\n')
        let inApiSection = false

        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine === '[api]') {
            inApiSection = true
            continue
          }

          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim()
              console.log('从配置文件加载API地址:', this.apiBaseUrl)
              break
            }
          }

          if (inApiSection && trimmedLine.startsWith('[')) {
            break
          }
        }

        // 配置文件未读取到baseUrl时，使用兜底地址
        if (!this.apiBaseUrl) {
          this.apiBaseUrl = 'http://10.134.13.242:5005'
          console.warn('配置文件未找到baseUrl，使用兜底地址')
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用兜底地址:', error.message)
        this.apiBaseUrl = 'http://10.134.13.242:5005'
      }
    },

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
    },

    handleDrop(e) {
      e.stopPropagation()
      e.preventDefault()
      if (this.loading) return
      const files = e.dataTransfer.files
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!')
        return
      }
      const rawFile = files[0]
      this.upload(rawFile)
    },

    handleDragover(e) {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },

    async upload(rawFile) {
      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成，请稍后重试')
        return
      }

      // 检查文件大小（假设最大100MB）
      const maxSize = 100 * 1024 * 1024 // 100MB
      if (rawFile.size > maxSize) {
        this.$message.error('文件大小不能超过100MB')
        return
      }

      this.uploadResult = null
      this.loading = true
      this.isElLoading = true
      this.isElLoadings = true

      const formData = new FormData()
      formData.append('file', rawFile)

      try {
        const response = await apiService.post(`${this.apiBaseUrl}/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 600000
        })

        console.log('Response status:', response.status)
        console.log('完整的Response data:', response.data)

        this.uploadResult = response.data

        if (this.uploadResult && this.uploadResult.original_filename) {
          // 清空之前的检测结果
          this.results = []
          this.behaviour_results = null

          this.$message.success('文件上传成功，正在获取检测结果...')

          setTimeout(() => {
            this.fetchDetailAPI()
          }, 500)
        } else {
          console.warn('返回数据格式不正确:', this.uploadResult)
          this.$message.warning('上传成功，但返回数据格式不正确')
        }
      } catch (error) {
        console.error('Error uploading file:', error)
        let errMsg = '文件上传失败！'
        if (error.code === 'ECONNABORTED') {
          errMsg = '文件上传超时（已设置10分钟，请检查后端处理速度）'
        } else if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录'
        } else if (error.message) {
          errMsg += ' ' + error.message
        }
        this.$message.error(errMsg)
      } finally {
        this.loading = false
      }
    },

    async fetchDetailAPI() {
      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成')
        this.isElLoading = false
        this.isElLoadings = false
        return
      }

      // 重置状态
      this.results = []
      this.behaviour_results = {}
      this.isError = false
      this.isErrors = false

      console.log('准备获取详情，uploadResult:', this.uploadResult)

      if (this.uploadResult && this.uploadResult.query_result && this.uploadResult.VT_API) {
        const sha256 = this.uploadResult.query_result.SHA256
        const VT_API = this.uploadResult.VT_API
        console.log('使用sha256:', sha256, 'VT_API:', VT_API)

        if (!sha256) {
          console.error('SHA256为空，无法获取详情')
          this.isElLoading = false
          this.isElLoadings = false
          this.$message.warning('文件特征信息不完整，无法获取检测详情')
          return
        }

        try {
          // 并行获取杀软检测和动态检测数据
          const [detectionResponse, behaviourResponse] = await Promise.all([
            apiService.get(`${this.apiBaseUrl}/detection_API/${sha256}`, {
              params: { VT_API: VT_API },
              timeout: 600000
            }).catch(err => {
              console.error('杀软检测请求失败:', err)
              return { data: [] }
            }),
            apiService.get(`${this.apiBaseUrl}/behaviour_API/${sha256}`, {
              params: { VT_API: VT_API },
              timeout: 600000
            }).catch(err => {
              console.error('动态检测请求失败:', err)
              return { data: {}}
            })
          ])

          // 处理杀软检测结果
          if (Array.isArray(detectionResponse.data) && detectionResponse.data.length > 0) {
            this.results = detectionResponse.data
            console.log('杀软检测结果获取成功，数量:', this.results.length)
          } else {
            console.warn('杀软检测数据格式异常或为空:', detectionResponse.data)
            this.isErrors = true
          }

          // 处理动态检测结果
          const data = behaviourResponse.data
          if (typeof data === 'object' && !Array.isArray(data)) {
            this.behaviour_results = data
            if (data.message) {
              console.warn('动态检测返回消息:', data.message)
              this.isError = true
            }
            console.log('动态检测结果获取成功')
          } else {
            console.warn('动态检测数据格式异常:', data)
            this.isError = true
          }

          // 默认显示基础信息
          this.showSection = 'fileInfo'
        } catch (error) {
          console.error('获取检测详情时出错:', error)
          this.isErrors = true
          this.isError = true
          this.$message.warning('部分检测数据获取失败，但基础信息已显示')
        } finally {
          this.isElLoading = false
          this.isElLoadings = false
        }
      } else {
        console.warn('缺少必要的查询信息:', {
          hasQueryResult: !!this.uploadResult?.query_result,
          hasVT_API: !!this.uploadResult?.VT_API
        })
        this.isElLoading = false
        this.isElLoadings = false
        this.$message.warning('缺少文件特征信息，无法获取检测详情')
      }
    }
  }
}
</script>

<style scoped>
/* 检测模式选择样式 */
.detection-mode-selector {
  margin: 30px auto 20px;
  max-width: 800px;
}

.mode-tabs {
  display: flex;
  justify-content: center;
  border-bottom: 2px solid #e8e8e8;
}

.mode-tab {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px 24px;
  background: none;
  border: none;
  border-bottom: 3px solid transparent;
  font-size: 16px;
  color: #666;
  cursor: pointer;
  transition: all 0.3s ease;
  margin: 0 10px;
}

.mode-tab:hover {
  color: #409eff;
}

.mode-tab.active {
  color: #409eff;
  border-bottom-color: #409eff;
  font-weight: bold;
}

.mode-icon {
  width: 20px;
  height: 20px;
  margin-right: 8px;
}

/* 文件检测模式样式 - 修复 */
.file-detection-mode {
  margin: 20px auto;
  max-width: 800px;
}

.file-upload-input {
  display: none;
  z-index: -9999;
}

.drop {
  border: 2px dashed #409eff;
  width: 100%;
  min-height: 200px;
  margin: 0 auto;
  font-size: 18px;
  border-radius: 8px;
  text-align: center;
  color: #666;
  position: relative;
  background-color: #f0f9ff;
  transition: all 0.3s ease;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.drop:hover {
  border-color: #67c23a;
  background-color: #f0f9eb;
}

.drop-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.upload-icon {
  width: 60px;
  height: 60px;
  color: #409eff;
  margin-bottom: 20px;
}

.drop-text {
  font-size: 18px;
  color: #666;
  margin-bottom: 15px;
}

.file-hint {
  font-size: 14px;
  color: #999;
  margin: 5px 0;
  line-height: 1.5;
}

/* SHA256检测模式样式 */
.sha256-detection-mode {
  margin: 20px auto;
  max-width: 800px;
}

/* 检测结果区域 - 新增滚动容器 */
.report-container {
  margin-top: 40px;
  max-height: 70vh; /* 设置最大高度为视口的70% */
  overflow-y: auto; /* 垂直方向滚动 */
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  scroll-behavior: smooth; /* 平滑滚动 */
}

/* 自定义滚动条样式 */
.report-container::-webkit-scrollbar {
  width: 8px;
}

.report-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.report-container::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.report-container::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.no-result-message {
  text-align: center;
  margin-top: 30px;
  padding: 20px;
  color: #f56c6c;
  font-size: 16px;
  background-color: #fef0f0;
  border-radius: 4px;
  border: 1px solid #fde2e2;
}

/* 从sample-vt.vue复制的样式 */
.file-upload-input{
  display: none;
  z-index: -9999;
}
.drop{
  border: 2px dashed #bbb;
  width: 60%;
  height: 160px;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
  position: relative;
}

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

.text-success {
color: green;
}

.fas.fa-check {
color: green;
}

.text-danger {
color: red;
}

.centered-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 90%;
  text-align: center;
  padding: 20px;
}

table {
  width: 60%;
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse;
  margin-top: 30px;
  border-bottom: 1px solid
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

.flex-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  list-style: none;
  padding: 0;
  margin: 0;
  text-align: left;
  width: 90%;
  margin: 0 auto;
}

.list-row {
  margin-bottom: 3px;
  width:100%;
  position: relative;
  padding: 5px 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.list-row > * {
  white-space: normal;
  word-break: break-all;
  text-indent: 1em;
}

h3 {
  color: black;
  width: 90%;
  margin: 15px auto;
  text-align: left;
  padding-top: 10px;
  padding-bottom: 10px;
  border-top: 1px solid grey;
  border-bottom: 1px solid grey;
  cursor: pointer;
}

h4 {
  color: black;
  width:90%;
  margin: 10px auto;
  text-align: left;
  padding-top: 10px;
  padding-bottom: 0px;
}

.behaviour_result_table {
  width: 90%;
  border-bottom: 0px solid;
  margin-top: 5px;
  border-collapse: collapse;
}

.behaviour_result_table th,
.behaviour_result_table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}

.custom-button {
  margin-top: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: calc(25% - 10px);
  height: 50px;
  background-color: white;
  border: 1px solid #ccc;
  cursor: pointer;
  padding: 0 10px;
  font-size: 16px;
  color: #333;
  transition: background-color 0.3s ease;
}

.active-button {
  background-color: #409EFF;
  color: white;
  border-color: #409EFF;
}

.icon {
  width: 24px;
  height: 24px;
  margin-right: 10px;
}

.button-icon {
  width: 24px;
  height: 24px;
  margin-right: 8px;
  margin-top: 2px;
}

.result-status {
  font-size: 30px;
  color: #333;
  font-weight: bold;
  text-align: center;
  display: inline-block;
  margin-top: 20px;
}
</style>
