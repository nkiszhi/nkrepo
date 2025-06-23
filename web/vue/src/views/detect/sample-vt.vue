<template>
  <main> <!-- 新的根元素 -->
    <div class="text-center">
      <h2 class="text-primary">基于可信度评估的多模型恶意文件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <input ref="file-upload-input" class="file-upload-input" type="file" @change="handleClick">
    <div class="drop" @drop="handleDrop" @dragover="handleDragover">
      把待检文件拖到这里或
      <el-button :loading="loading" style="margin-left:0%;font-size: 20px;" size="mini" type="primary" @click="handleUpload">
        选择待检文件
      </el-button>
    </div>

    <div v-if="uploadResult">
      <div style="text-align: center;" class="result-content">
        <svg-icon style="width: 30px; height: 30px;margin-right: 10px;margin-top:30px; " icon-class="detect-report" />
        <span class="result-status">检测报告如下</span>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center; margin-top:50px; width: 60%;margin: 0 auto">
        <button class="custom-button" :class="{ 'active-button': showSection === 'fileInfo' }" @click="showSection = 'fileInfo'">
          <svg-icon icon-class="fileInfo" class="button-icon" />
          <span> 基础信息</span>
        </button>
        <button class="custom-button" :class="{ 'active-button': showSection === 'modelDetection' }" @click="showSection = 'modelDetection'">
          <svg-icon icon-class="modelDetection" class="button-icon" />
          <span> 模型检测</span>
        </button>
        <button class="custom-button" :class="{ 'active-button': showSection === 'AV-Detection' }" @click="showSection = 'AV-Detection'">
          <svg-icon icon-class="AV-Detection" class="button-icon" />
          <span> 杀软检测</span>
        </button>
        <button class="custom-button" :class="{ 'active-button': showSection === 'DynamicDetection' }" @click="showSection = 'DynamicDetection'">
          <svg-icon icon-class="DynamicDetection" class="button-icon" />
          <span> 动态检测</span>
        </button>
      </div>

      <!-- 上传文件基本信息 -->
      <div v-show="showSection === 'fileInfo'" class="section">
        <table class="file-info-table">
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
          <tr v-for="(value, key) in uploadResult.query_result" v-if="value !== 'nan' && value !== 'NaN'" :key="key">
            <td>{{ key.replace('_', ' ') }}：</td>
            <td>{{ value }}</td>
          </tr>
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
        <div v-if="isLoading" class="isLoading">
          <p>正在加载数据...</p>
        </div>
        <div v-else-if="!isErrors && results.length > 0">
          <h2 style="text-align: center;">杀毒软件检测结果</h2>
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
      </div>

      <!-- 动态检测结果 -->
      
      <!-- 加载过程 -->
      <div v-show="showSection === 'DynamicDetection'" class="section">
        <div v-if="isLoadings" class="isLoading">
          <p>正在加载数据...</p>
        </div>
    
       <!-- 失败或空数据状态 -->
        <div v-if="!isLoadings && (isError || !hasValidData)">
          <h2 style="text-align: center;">无动态检测结果</h2>
          <p v-if="isError" style="text-align: center;">未检测到动态行为数据</p>
          <p v-else style="text-align: center;">未检测到任何动态行为数据</p>
        </div>

       <div v-if="!isLoadings && !isError && hasValidData">
         <h2 style="text-align: center;">动态检测结果</h2>
            <!-- API -->
            <div v-if="behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0" style="text-align: center;">
              <h3 :title="toggleCall ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleCall = !toggleCall">API调用情况:{{ behaviour_results.calls_highlighted ? behaviour_results.calls_highlighted.length : 0 }}个API调用</h3>
              <div v-if="toggleCall &&behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0">
                <ul class="flex-container">
                  <li v-for="(calls_highlighted, index) in behaviour_results.calls_highlighted" :key="index" style="text-align: left;" class="list-row">
                    {{ calls_highlighted }}
                  </li>
                </ul>
              </div>
            </div>

            <!-- 服务 -->
            <div
              v-if="
                (behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0) ||
                  (behaviour_results && behaviour_results.services_started && behaviour_results.services_started.length > 0)
              "
              style="text-align: center;"
            >
              <h3 :title="toggleServicesOpened ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleServicesOpened = !toggleServicesOpened">
                服务情况:{{ behaviour_results.services_opened ? behaviour_results.services_opened.length : 0 }}个打开的服务；{{ behaviour_results.services_started ? behaviour_results.services_started.length : 0 }} 个启动的服务
              </h3>
              
              <div v-if="toggleServicesOpened &&behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0">
                <h4>打开的服务</h4>
                <ul class="flex-container">
                  <li v-for="(services_opened, index) in behaviour_results.services_opened" :key="index" class="list-row">
                    {{ services_opened }}
                  </li>
                </ul>
              </div>
              
              <div v-if="toggleServicesOpened &&behaviour_results && behaviour_results.services_started && behaviour_results.services_started.length > 0">
                <h4>启动的服务</h4>
                <ul class="flex-container">
                  <li v-for="(services_started, index) in behaviour_results.services_started" :key="index" class="list-row">
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
                文件行为:{{ behaviour_results.command_executions ? behaviour_results.command_executions.length : 0 }}个执行；{{ behaviour_results.files_attribute_changed ? behaviour_results.files_attribute_changed.length : 0 }} 个属性变更；{{ behaviour_results.files_copied ? behaviour_results.files_copied.length : 0 }} 个复制；{{ behaviour_results.files_deleted ? behaviour_results.files_deleted.length : 0 }} 个删除；{{ behaviour_results.files_dropped ? behaviour_results.files_dropped.length : 0 }} 个释放；{{ behaviour_results.files_opened ? behaviour_results.files_opened.length : 0 }} 个打开；{{ behaviour_results.files_written ? behaviour_results.files_written.length : 0 }} 个写入</h3>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.command_executions && behaviour_results.command_executions.length > 0">
                <h4>文件执行情况</h4>
                <ul class="flex-container">
                  <li v-for="(path, index) in behaviour_results.command_executions" :key="index" style="text-align: left;" class="list-row">
                    {{ path }}
                  </li>
                </ul>
              </div>

              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_attribute_changed && behaviour_results.files_attribute_changed.length > 0">
                <h4>文件属性变更情况</h4>
                <ul class="flex-container">
                  <li v-for="(files_attribute, index) in behaviour_results.files_attribute_changed" :key="index" class="list-row">
                    {{ files_attribute }}
                  </li>
                </ul>
              </div>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_copied && behaviour_results.files_copied.length > 0">
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
                    <tr v-for="(files_copied, index) in behaviour_results.files_copied" :key="index" class="behaviour_result_table_row">
                      <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                      <td><span>{{ files_copied.key }}</span></td>
                      <td><span>{{ files_copied.value }}</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_deleted && behaviour_results.files_deleted.length > 0">
                <h4>文件删除情况</h4>
                <ul class="flex-container">
                  <li v-for="(path_del, index) in behaviour_results.files_deleted" :key="index" class="list-row">
                    {{ path_del }}
                  </li>
                </ul>
              </div>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_dropped && behaviour_results.files_dropped.length > 0">
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
                    <tr v-for="(files_dropped, index) in behaviour_results.files_dropped" :key="index" class="behaviour_result_table_row">
                      <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                      <td><span>{{ files_dropped.path }}</span></td>
                      <td><span>{{ files_dropped.sha256 }}</span></td>
                      <td><span>{{ files_dropped.type }}</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_opened && behaviour_results.files_opened.length > 0">
                <h4>文件打开情况</h4>
                <ul class="flex-container">
                  <li v-for="(path, index) in behaviour_results.files_opened" :key="index" class="list-row">
                    {{ path }}
                  </li>
                </ul>
              </div>
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_written && behaviour_results.files_written.length > 0">
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

              <!-- 进程 -->
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
                <h3 :title="toggleProcesses ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleProcesses = !toggleProcesses">
                  进程行为:{{ behaviour_results.modules_loaded ? behaviour_results.modules_loaded.length : 0 }}个模块加载；{{ behaviour_results.mutexes_created ? behaviour_results.mutexes_created.length : 0 }} 个互斥锁创建；{{ behaviour_results.mutexes_opened ? behaviour_results.mutexes_opened.length : 0 }} 个互斥锁打开；{{ behaviour_results.permissions_requested ? behaviour_results.permissions_requested.length : 0 }} 个权限请求；{{ behaviour_results.processes_terminated ? behaviour_results.processes_terminated.length : 0 }} 个进程终止；{{ behaviour_results.processes_tree ? behaviour_results.processes_tree.length : 0 }} 进程树</h3>
                <div v-if="toggleProcesses && behaviour_results && behaviour_results.modules_loaded && behaviour_results.modules_loaded.length > 0">
                  <h4>模块加载</h4>
                  <ul class="flex-container">
                    <li v-for="(module, index) in behaviour_results.modules_loaded" :key="index" class="list-row">
                      {{ module }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleProcesses && behaviour_results && behaviour_results.mutexes_created && behaviour_results.mutexes_created.length > 0">
                  <h4>互斥锁创建</h4>
                  <ul class="flex-container">
                    <li v-for="(mutexes_created, index) in behaviour_results.mutexes_created" :key="index" class="list-row">
                      {{ mutexes_created }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleProcesses && behaviour_results && behaviour_results.mutexes_opened && behaviour_results.mutexes_opened.length > 0">
                  <h4>互斥锁打开</h4>
                  <ul class="flex-container">
                    <li v-for="(mutexes_opened, index) in behaviour_results.mutexes_opened" :key="index" class="list-row">
                      {{ mutexes_opened }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleProcesses && behaviour_results && behaviour_results.permissions_requested && behaviour_results.permissions_requested.length > 0">
                  <h4>权限请求</h4>
                  <ul class="flex-container">
                    <li v-for="(permissions_requested, index) in behaviour_results.permissions_requested" :key="index" class="list-row">
                      {{ permissions_requested }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleProcesses && behaviour_results && behaviour_results.processes_terminated && behaviour_results.processes_terminated.length > 0">
                  <h4>进程终止</h4>
                  <ul class="flex-container">
                    <li v-for="(processes_terminated, index) in behaviour_results.processes_terminated" :key="index" class="list-row">
                      {{ processes_terminated }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleProcesses && behaviour_results && behaviour_results.processes_tree && behaviour_results.processes_tree.length > 0">
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
                      <tr v-for="(processes_tree, index) in behaviour_results.processes_tree" :key="index" class="behaviour_result_table_row">
                        <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
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

                <!-- 网络 -->
                <div
                  v-if="
                    (behaviour_results && behaviour_results.dns_lookups && behaviour_results.dns_lookups.length > 0) ||
                      (behaviour_results && behaviour_results.http_conversations && behaviour_results.http_conversations.length > 0) ||
                      (behaviour_results && behaviour_results.ip_traffic && behaviour_results.ip_traffic.length > 0) ||
                      (behaviour_results && behaviour_results.tls && behaviour_results.tls.length > 0)
                  "
                  style="text-align: center;"
                >
                  <h3 :title="toggleNetwork ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleNetwork = !toggleNetwork">
                    网络行为:{{ behaviour_results.dns_lookups ? behaviour_results.dns_lookups.length : 0 }}个DNS查找；{{ behaviour_results.http_conversations ? behaviour_results.http_conversations.length : 0 }} 个HTTP会话；{{ behaviour_results.ip_traffic ? behaviour_results.ip_traffic.length : 0 }} 个IP流量；{{ behaviour_results.tls ? behaviour_results.tls.length : 0 }} 传输层安全协议</h3>
                  <div v-if="toggleNetwork && behaviour_results && behaviour_results.dns_lookups && behaviour_results.dns_lookups.length > 0">
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
                        <tr v-for="(dns_lookups, index) in behaviour_results.dns_lookups" :key="index" class="behaviour_result_table_row">
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                          <td><span>{{ dns_lookups.hostname }}</span></td>
                          <td><span>{{ dns_lookups.resolved_ips }}</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div v-if="toggleNetwork && behaviour_results && behaviour_results.http_conversations && behaviour_results.http_conversations.length > 0">
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
                        <tr v-for="(http_conversations, index) in behaviour_results.http_conversations" :key="index" class="behaviour_result_table_row">
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                          <td><span>{{ http_conversations.url }}</span></td>
                          <td><span>{{ http_conversations.request_method }}</span></td>
                          <td><span>{{ http_conversations.request_headers }}</span></td>
                          <td><span>{{ http_conversations.response_status_code }}</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div v-if="toggleNetwork && behaviour_results && behaviour_results.ip_traffic && behaviour_results.ip_traffic.length > 0">
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
                        <tr v-for="(ip_traffic, index) in behaviour_results.ip_traffic" :key="index" class="behaviour_result_table_row">
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
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
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
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

              <!-- 攻击技术 -->
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
                <h3 :title="toggleAttack ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleAttack = !toggleAttack">
                  攻击行为:{{ behaviour_results.verdicts ? behaviour_results.verdicts.length : 0 }} 个判决结果；{{ Object.keys(behaviour_results.attack_techniques || {}).length }} 个攻击技术；{{ behaviour_results.ids_alerts ? behaviour_results.ids_alerts.length : 0 }} 个入侵警报；{{ behaviour_results.mbc ? behaviour_results.mbc.length : 0 }} 个mbc；{{ behaviour_results.mitre_attack_techniques ? behaviour_results.mitre_attack_techniques.length : 0 }} 个MITRE ATT&CK技术；{{ behaviour_results.signature_matches ? behaviour_results.signature_matches.length : 0 }} 个签名;{{ behaviour_results.system_property_lookups ? behaviour_results.system_property_lookups.length : 0 }} 个系统属性查找</h3>
               
                <div v-if="toggleAttack && behaviour_results && behaviour_results.verdicts && behaviour_results.verdicts.length > 0">
                  <h4>判决结果</h4>
                  <ul class="flex-container">
                    <li v-for="(verdicts, index) in behaviour_results.verdicts" :key="index" class="list-row">
                      {{ verdicts }}
                    </li>
                  </ul>
                </div>

                <div v-if="toggleAttack && behaviour_results && behaviour_results.verdict_confidence && behaviour_results.verdict_confidence.length > 0">
                  <h4>置信度{{ behaviour_results.verdict_confidence }}</h4>
                </div>

               <div v-if="toggleAttack && behaviour_results && behaviour_results.attack_techniques && Object.keys(behaviour_results.attack_techniques).length > 0">
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
                 <tr v-for="(item, itemIndex) in techItems" :key="techId+'-'+itemIndex" class="behaviour_result_table_row">
                   <td>{{ index + 1 }}</td>
                   <td><span>{{ techId }}</span></td>
                   <td><span>{{ item.severity }}</span></td>
                   <td><span>{{ item.description }}</span></td>
                </tr>
              </template>
              </tbody>
          </table>
          </div>

                <div v-if="toggleAttack && behaviour_results && behaviour_results.ids_alerts && behaviour_results.ids_alerts.length > 0">
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
                      <tr v-for="(ids_alerts, index) in behaviour_results.ids_alerts" :key="index" class="behaviour_result_table_row">
                        <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
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
                        <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                        <td><span>{{ mbc.id }}</span></td>
                        <td><span>{{ mbc.objective }}</span></td>
                        <td><span>{{ mbc.behavior }}</span></td>
                        <td><span>{{ mbc.refs }}</span></td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div v-if="toggleAttack && behaviour_results && behaviour_results.mitre_attack_techniques && behaviour_results.mitre_attack_techniques.length > 0">
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
                      <tr v-for="(mitre_attack_techniques, index) in behaviour_results.mitre_attack_techniques" :key="index" class="behaviour_result_table_row">
                        <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                        <td><span>{{ mitre_attack_techniques.id }}</span></td>
                        <td><span>{{ mitre_attack_techniques.severity }}</span></td>
                        <td><span>{{ mitre_attack_techniques.signature_description }}</span></td>
                        <td><span>{{ mitre_attack_techniques.refs }}</span></td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div v-if="toggleAttack && behaviour_results && behaviour_results.signature_matches && behaviour_results.signature_matches.length > 0">
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
                      <tr v-for="(signature_matches, index) in behaviour_results.signature_matches" :key="index" class="behaviour_result_table_row">
                        <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                        <td><span>{{ signature_matches.format }}</span></td>
                        <td><span>{{ signature_matches.name }}</span></td>
                        <td><span>{{ signature_matches.authors }}</span></td>
                        <td><span>{{ signature_matches.rule_src }}</span></td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div v-if="toggleAttack && behaviour_results && behaviour_results.system_property_lookups && behaviour_results.system_property_lookups.length > 0">
                  <h4>系统属性</h4>
                  <ul class="flex-container">
                    <li v-for="(system_property_lookups, index) in behaviour_results.system_property_lookups" :key="index" class="list-row">
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
                  <h3 :title="toggleMemory ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleMemory = !toggleMemory">
                    内存情况:{{ behaviour_results.memory_dumps ? behaviour_results.memory_dumps.length : 0 }} 个内容转储；{{ behaviour_results.memory_pattern_domains ? behaviour_results.memory_pattern_domains.length : 0 }} 个域名模式；{{ behaviour_results.memory_pattern_urls ? behaviour_results.memory_pattern_urls.length : 0 }} 个URL模式</h3>
                  <div v-if="toggleMemory && behaviour_results && behaviour_results.memory_dumps && behaviour_results.memory_dumps.length > 0">
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
                        <tr v-for="(memory_dumps, index) in behaviour_results.memory_dumps" :key="index" class="behaviour_result_table_row">
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
                          <td><span>{{ memory_dumps.file_name }}</span></td>
                          <td><span>{{ memory_dumps.process }}</span></td>
                          <td><span>{{ memory_dumps.size }}</span></td>
                          <td><span>{{ memory_dumps.base_address }}</span></td>
                          <td><span>{{ memory_dumps.stage }}</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  
                  <div v-if="toggleMemory && behaviour_results && behaviour_results.memory_pattern_domains && behaviour_results.memory_pattern_domains.length > 0">
                    <h4>内存中的域名模式</h4>
                    <ul class="flex-container">
                      <li v-for="(memory_pattern_domains, index) in behaviour_results.memory_pattern_domains" :key="index" class="list-row">
                        {{ memory_pattern_domains }}
                      </li>
                    </ul>
                  </div>
                  
                  <div v-if="toggleMemory && behaviour_results && behaviour_results.memory_pattern_urls && behaviour_results.memory_pattern_urls.length > 0">
                    <h4>内存中的URL模式</h4>
                    <ul class="flex-container">
                      <li v-for="(memory_pattern_urls, index) in behaviour_results.memory_pattern_urls" :key="index" class="list-row">
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
                  <h3 :title="toggleRegistry ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleRegistry = !toggleRegistry">
                    注册表:{{ behaviour_results.registry_keys_deleted ? behaviour_results.registry_keys_deleted.length : 0 }} 个注册表删除；{{ behaviour_results.registry_keys_opened ? behaviour_results.registry_keys_opened.length : 0 }} 个注册表打开；{{ behaviour_results.registry_keys_set ? behaviour_results.registry_keys_set.length : 0 }} 个注册表设置</h3>
                 
                  <div v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_deleted && behaviour_results.registry_keys_deleted.length > 0">
                    <h4>注册表删除</h4>
                    <ul class="flex-container">
                      <li v-for="(registry_keys_deleted, index) in behaviour_results.registry_keys_deleted" :key="index" class="list-row">
                        {{ registry_keys_deleted }}
                      </li>
                    </ul>
                  </div>
                  
                  <div v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_opened && behaviour_results.registry_keys_opened.length > 0">
                    <h4>注册表打开</h4>
                    <ul class="flex-container">
                      <li v-for="(registry_keys_opened, index) in behaviour_results.registry_keys_opened" :key="index" class="list-row">
                        {{ registry_keys_opened }}
                      </li>
                    </ul>
                  </div>

                  <div v-if="toggleRegistry && behaviour_results && behaviour_results.registry_keys_set && behaviour_results.registry_keys_set.length > 0">
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
                        <tr v-for="(registry_keys_set, index) in behaviour_results.registry_keys_set" :key="index" class="behaviour_result_table_row">
                          <td>{{ index + 1 }}</td> <!-- 显示序号，从1开始 -->
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
                  <h3 :title="toggleCrypto ? '点击收起' : '点击显示详情'" style="cursor: pointer;" @click="toggleCrypto = !toggleCrypto">
                    加密情况:{{ behaviour_results.crypto_algorithms_observed ? behaviour_results.crypto_algorithms_observed.length : 0 }} 个加密算法；{{ behaviour_results.crypto_plain_text ? behaviour_results.crypto_plain_text.length : 0 }} 个加密明文；{{ behaviour_results.text_highlighted ? behaviour_results.text_highlighted.length : 0 }} 个高亮文本</h3>

                  <div v-if="toggleCrypto && behaviour_results && behaviour_results.crypto_algorithms_observed && behaviour_results.crypto_algorithms_observed.length > 0">
                    <h4>加密算法</h4>
                    <ul class="flex-container">
                      <li v-for="(crypto_algorithms_observed, index) in behaviour_results.crypto_algorithms_observed" :key="index" class="list-row">
                        {{ crypto_algorithms_observed }}
                      </li>
                    </ul>
                  </div>

                  <div v-if="toggleCrypto && behaviour_results && behaviour_results.crypto_plain_text && behaviour_results.crypto_plain_text.length > 0">
                    <h4>加密明文</h4>
                    <ul class="flex-container">
                      <li v-for="(crypto_plain_text, index) in behaviour_results.crypto_plain_text" :key="index" class="list-row">
                        {{ crypto_plain_text }}
                      </li>
                    </ul>
                  </div>
                  
                  <div v-if="toggleCrypto && behaviour_results && behaviour_results.text_highlighted && behaviour_results.text_highlighted.length > 0">
                    <h4>高亮文本</h4>
                    <ul class="flex-container">
                      <li v-for="(text_highlighted, index) in behaviour_results.text_highlighted" :key="index" class="list-row">
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
          
        </div>
      </div>
     </div>
    </div>

  </main>
</template>

<script>

import axios from 'axios'

export default {
  data() {
    return {
      showSection: 'fileInfo',
      loading: false,
      isLoading: false,
      isLoadings: false,
      results: {},
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
      isError: false
    }
  },
  
computed: {
  hasValidData() {
    if (!this.behaviour_results || typeof this.behaviour_results !== 'object') {
      return false;
    }
    
    const r = this.behaviour_results;
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
    ].some(key => r[key]?.length > 0);
  }
},


  
  methods: {
    getIconClass(category) {
      // 这里是一个简单的映射示例
      // 你需要根据实际情况调整这个映射
      switch (category) {
        case 'malicious':
        case 'suspicious':
          return 'vt_malicious'
        case 'undetected':
        case 'harmless':
          return 'vt_undetected'
        // 添加其他 case 以覆盖所有可能的 category
        default:
          // 如果 category 不在映射中，可以返回一个默认的图标类
          // 或者返回一个空字符串/null，具体取决于 <svg-icon> 组件如何处理无效的属性值
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
      const rawFile = files[0] // only use files[0]
      this.upload(rawFile)
    },
    handleDragover(e) {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },
    handleUpload() {
      this.$refs['file-upload-input'].click()
    },
    handleClick(e) {
      const files = e.target.files
      const rawFile = files[0] // only use files[0]
      if (!rawFile) return
      this.upload(rawFile)
    },
    async upload(rawFile) {
      this.uploadResult = null
      this.loading = true
      const formData = new FormData()
      formData.append('file', rawFile)
      try {
        const response = await fetch('http://10.134.2.27:5005/upload', {
          method: 'POST',
          body: formData
        })
        console.log('Response status:', response.status)
        console.log('Response headers:', response.headers) // 打印响应头
        this.$forceUpdate()

        // 检查响应状态码，如果非200-299之间，则视为失败
        if (!response.ok) {
          throw new Error('Failed to upload file: ' + response.statusText)
        }

        // 尝试解析JSON
        const data = await response.json()
        console.log('Response data:', data) // 打印响应数据
        this.uploadResult = data
        await this.fetchDetailAPI()
        this.$forceUpdate()
      } catch (error) {
      // 在catch块中打印出具体的错误信息
        console.error('Error uploading file:', error)
        if (error instanceof Error) {
          // 如果是Error对象，可以打印出更详细的错误描述
          console.error('Error message:', error.message)
          console.error('Error stack:', error.stack)
        }
        this.$message.error('文件上传失败！')
      } finally {
        this.loading = false
      }
    },
    fetchDetailAPI() {
      this.isLoading = true
      this.results = []
      this.behaviour_results = {}
      this.isError = false
      this.isErrors = false

      if (this.uploadResult && this.uploadResult.query_result && this.uploadResult.VT_API) {
        const sha256 = this.uploadResult.query_result.SHA256
        const VT_API = this.uploadResult.VT_API
        console.log('sha256:', sha256)
        console.log('VT_API:', VT_API)

        axios.get(`http://10.134.2.27:5005/detection_API/${sha256}`, { params: { VT_API: VT_API }})
          .then(detectionResponse => {
            // 处理检测API的响应
            if (Array.isArray(detectionResponse.data) && detectionResponse.data.length > 0) {
              this.results = detectionResponse.data
            } else {
              this.error = 'Unexpected response format from detection API'
              this.isErrors = true
            }
          })
          .catch(error => {
            // 捕获检测API的错误
            this.isErrors = true
            console.error('Error fetching detection data:', error)
            this.error = 'Error fetching data from detection API'
          })
          .finally(() => {
            // 无论成功还是失败，都执行
            this.checkAndUpdateUI()
          })

        axios.get(`http://10.134.2.27:5005/behaviour_API/${sha256}`, { params: { VT_API: VT_API }})
          .then(behaviourResponse => {
            // 处理行为API的响应
            if (typeof behaviourResponse.data === 'object' && !Array.isArray(behaviourResponse.data)) {
              this.behaviour_results = behaviourResponse.data
            } else if (typeof behaviourResponse.data === 'object' && behaviourResponse.data.message) {
              // 处理后端返回的特定消息
              this.isError = true
              this.behaviour_results = behaviourResponse.data // 如果错误信息也需要显示在模板中，可以这样做
            } else {
              // 处理非预期的响应格式
              this.isError = true
              this.error = 'Unexpected response format from behaviour API'
            }
          })
          .catch(error => {
            // 捕获请求错误
            this.isError = true
            console.error('Error fetching behaviour data:', error)
            this.error = 'Error fetching data from behaviour API'
          })
          .finally(() => {
            // 无论成功还是失败，都执行
            this.checkAndUpdateUIs()
          })

        // 检查并更新UI的方法
        this.checkAndUpdateUIs = () => {
          if (!this.isLoadings) return // 如果不是加载状态，则直接返回
          this.$forceUpdate()
          this.isLoadings = false
        }
        this.checkAndUpdateUI = () => {
          if (!this.isLoading) return // 如果不是加载状态，则直接返回
          this.$forceUpdate()
          this.isLoading = false
        }

        // 初始设置为加载状态
        this.isLoading = true
        this.isLoadings = true
      }
    }
  }
}
</script>

<style scoped>
.file-upload-input{
  display: none;
  z-index: -9999;
}
.drop{
  border: 2px dashed #bbb;
  width: 60%;
  height: 10%;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
  position: relative;
}

.file-info-table {
width: 60%; /* 表格宽度设置为100% */
margin: 0 auto;
border: 1px solid #ccc;
border-collapse: collapse; /* 合并边框 */
margin-top: 30px;
}

.file-info-table th,
.file-info-table td {
padding: 8px; /* 单元格内边距 */
text-align: left; /* 文本左对齐 */
border-bottom: 1px solid #ddd; /* 底部边框 */
}

.file-info-table tr:hover {
background-color: #f5f5f5; /* 鼠标悬停时行背景色 */
}

/* 第二个表格样式 */
.detection-result-table {
width: 60%; /* 表格宽度设置为100% */
margin: 0 auto;
border: 1px solid #ccc;
border-collapse: collapse; /* 合并边框 */
margin-top: 30px;
}

.detection-result-table th,
.detection-result-table td {
padding: 8px; /* 单元格内边距 */
text-align: left; /* 文本左对齐 */
border-bottom: 1px solid #ddd; /* 底部边框 */
}

/* 结果列样式 */
.detection-result-table td:last-child {
text-align: center; /* 最后一列文本居中对齐 */
}

/* 成功的文本和图标样式 */
.text-success {
color: green; /* 文本颜色为绿色 */
}

.fas.fa-check {
color: green; /* 图标颜色为绿色 */
}

/* 危险的文本和图标样式 */
.text-danger {
color: red; /* 文本颜色为红色 */
}

.fas.fa-exclamation-triangle {
color: red; /* 图标颜色为红色 */
}
.centered-container {
  display: flex;
  flex-direction: column;
  align-items: center; /* 水平居中 */
  justify-content: center; /* 垂直居中 */
  height: 90%; /* 占满整个视口高度 */
  text-align: center; /* 确保文本内容也水平居中 */
  padding: 20px; /* 添加一些内边距 */
}

/* 可能还需要为表格或其他元素添加一些样式来优化显示效果 */
table {
  width: 60%; /* 表格宽度设置为100% */
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse; /* 合并边框 */
  margin-top: 30px;
  border-bottom: 1px solid
}
table th,
table td {
padding: 8px; /* 单元格内边距 */
text-align: center; /* 文本左对齐 */
border-bottom: 1px solid #ddd; /* 底部边框 */
}
.vt_table-row:hover {
  background-color: #f0f0f0; /* 或者你喜欢的任何颜色 */
  box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 添加阴影效果 */
  /* 如果需要，还可以添加其他样式，比如改变文字颜色等 */
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
/* 错误消息的样式 */
.error-message {
  text-align: center;
  color: red;
}
.isLoading {
  text-align: center;
}
.flex-container {
  display: flex;
  flex-direction: column; /* 假设你想垂直排列列表项 */
  align-items: center; /* 垂直居中（如果需要水平居中，请调整flex-direction） */
  list-style: none; /* 移除列表项前的默认标记 */
  padding: 0; /* 移除默认的padding */
  margin: 0; /* 移除默认的margin */
  text-align: left;
}
.list-row {
  margin-bottom: 3px; /* 列表项之间的间隔 */
  width:90%; /* 宽度根据需要调整，这里设为100%以填充容器宽度 */
  position: relative; /* 为可能的子元素定位做准备 */
}
.list-row > * {
  white-space: normal; /* 确保子元素也允许换行 */
  word-break: break-all; /* 如果需要，也可以允许在任意字符间换行（谨慎使用） */
  text-indent: 1em;
}
h3 {
  color: black;
  width: 90%;
  margin: 0 auto;
  text-align: left;
  padding-top: 10px; /* 在元素顶部添加内边距 */
  padding-bottom: 10px; /* 在元素底部添加内边距 */
  /* 只设置上下边框 */
  border-top: 1px solid grey;
  border-bottom: 1px solid grey;
}
h4 {
  color: black; /* 可选：如果你想要文本在蓝色背景上清晰可见，可以添加这个 */
  width:90%;
  margin: 0 auto;
  text-align: left;
  padding-top: 10px; /* 在元素顶部添加内边距 */
  padding-bottom: 0px; /* 在元素底部添加内边距 */
}
.behaviour_result_table {
  width: 90%;
  border-bottom: 0px solid;
  margin-top: 5px;
}
.custom-button {
  margin-top: 10px; /* 移除margin-top，因为flex布局会处理对齐 */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: calc(25% - 10px); /* 减去一些宽度以考虑按钮间的间隔 */
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
  background-color: blue; /* 活动按钮的背景颜色 */
  color: white; /* 可能还需要设置字体颜色为白色以增加对比度 */
}

.icon {
  width: 24px; /* 图标宽度 */
  height: 24px; /* 图标高度 */
  margin-right: 10px; /* 图标与文本之间的间距 */
}

.button-icon {
  /* 假设你的图标是SVG，并且你想设置它的大小为32x32像素 */
  width: 25%;
  height: 40px;
  margin-right: 20px; /* 图标和文本之间的间距 */
  margin-top:5px;
}

.result-status {
  /* 状态文本的样式 */
  font-size: 30px;
  color: #333;
  font-weight: bold;
  text-justify:center;
}
</style>
