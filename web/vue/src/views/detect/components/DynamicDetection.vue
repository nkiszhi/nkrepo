<template>
  <div class="section dynamic-detection-container">
    <!-- 加载状态 -->
    <div v-if="isElLoadings" class="isElLoading">
      <p>正在加载数据...</p>
    </div>

    <!-- 失败或空数据状态 -->
    <div v-else-if="isError || !hasValidData">
      <h2 style="text-align: center;">无动态检测结果</h2>
      <p style="text-align: center; color: #666;">
        {{
          isError
            ? '动态检测数据加载失败或格式错误'
            : '未检测到任何动态行为数据'
        }}
      </p>
      <!-- 调试信息 -->
      <div v-if="debug" style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
        <p>数据状态: isError={{ isError }}, hasValidData={{ hasValidData }}</p>
        <p>数据对象: {{ behaviourResults ? '存在' : '空' }}</p>
        <p v-if="behaviourResults">数据长度: {{ Object.keys(behaviourResults).length }}</p>
      </div>
    </div>

    <!-- 有数据的情况 -->
    <div v-else class="dynamic-content">
      <h2 style="text-align: center;">动态检测结果</h2>

      <!-- API调用情况 -->
      <div v-if="hasData('calls_highlighted')" class="data-section">
        <div class="section-header" @click="toggleSection('calls_highlighted')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.calls_highlighted ? '▼' : '▶' }}</span>
            API调用情况: {{ behaviourResults.calls_highlighted.length }} 个API调用
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.calls_highlighted" class="section-content">
          <ul class="data-list">
            <li
              v-for="(call, index) in behaviourResults.calls_highlighted"
              :key="`call-${index}`"
              class="list-item"
            >
              <span class="item-index">{{ index + 1 }}.</span>
              <span class="item-content">{{ call }}</span>
            </li>
          </ul>
        </div>
      </div>

      <!-- 服务情况 -->
      <div v-if="hasData('services_opened') || hasData('services_started')" class="data-section">
        <div class="section-header" @click="toggleSection('services')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.services ? '▼' : '▶' }}</span>
            服务情况:
            {{ behaviourResults.services_opened ? behaviourResults.services_opened.length : 0 }} 个打开的服务；
            {{ behaviourResults.services_started ? behaviourResults.services_started.length : 0 }} 个启动的服务
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.services" class="section-content">
          <!-- 打开的服务 -->
          <div v-if="hasData('services_opened')" class="subsection">
            <h4 class="subsection-title">打开的服务</h4>
            <ul class="data-list">
              <li
                v-for="(service, index) in behaviourResults.services_opened"
                :key="`service-opened-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ service }}</span>
              </li>
            </ul>
          </div>

          <!-- 启动的服务 -->
          <div v-if="hasData('services_started')" class="subsection">
            <h4 class="subsection-title">启动的服务</h4>
            <ul class="data-list">
              <li
                v-for="(service, index) in behaviourResults.services_started"
                :key="`service-started-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ service }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 文件行为 -->
      <div v-if="hasFileData()" class="data-section">
        <div class="section-header" @click="toggleSection('files')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.files ? '▼' : '▶' }}</span>
            文件行为:
            {{ getCount('command_executions') }} 个执行；
            {{ getCount('files_attribute_changed') }} 个属性变更；
            {{ getCount('files_copied') }} 个复制；
            {{ getCount('files_deleted') }} 个删除；
            {{ getCount('files_dropped') }} 个释放；
            {{ getCount('files_opened') }} 个打开；
            {{ getCount('files_written') }} 个写入
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.files" class="section-content">
          <!-- 文件执行情况 -->
          <div v-if="hasData('command_executions')" class="subsection">
            <h4 class="subsection-title">文件执行情况</h4>
            <ul class="data-list">
              <li
                v-for="(path, index) in behaviourResults.command_executions"
                :key="`command-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ path }}</span>
              </li>
            </ul>
          </div>

          <!-- 文件属性变更情况 -->
          <div v-if="hasData('files_attribute_changed')" class="subsection">
            <h4 class="subsection-title">文件属性变更情况</h4>
            <ul class="data-list">
              <li
                v-for="(fileAttr, index) in behaviourResults.files_attribute_changed"
                :key="`attr-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ fileAttr }}</span>
              </li>
            </ul>
          </div>

          <!-- 文件复制情况 -->
          <div v-if="hasData('files_copied')" class="subsection">
            <h4 class="subsection-title">文件复制情况</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th>源文件</th>
                  <th>目标文件</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(file, index) in behaviourResults.files_copied"
                  :key="`copy-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ file.key }}</span></td>
                  <td><span class="table-cell">{{ file.value }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 文件删除情况 -->
          <div v-if="hasData('files_deleted')" class="subsection">
            <h4 class="subsection-title">文件删除情况</h4>
            <ul class="data-list">
              <li
                v-for="(path, index) in behaviourResults.files_deleted"
                :key="`delete-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ path }}</span>
              </li>
            </ul>
          </div>

          <!-- 文件释放情况 -->
          <div v-if="hasData('files_dropped')" class="subsection">
            <h4 class="subsection-title">文件释放情况</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th>路径</th>
                  <th>哈希值</th>
                  <th>文件类型</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(file, index) in behaviourResults.files_dropped"
                  :key="`drop-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ file.path }}</span></td>
                  <td><span class="table-cell">{{ file.sha256 }}</span></td>
                  <td><span class="table-cell">{{ file.type }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 文件打开情况 -->
          <div v-if="hasData('files_opened')" class="subsection">
            <h4 class="subsection-title">文件打开情况</h4>
            <ul class="data-list">
              <li
                v-for="(path, index) in behaviourResults.files_opened"
                :key="`open-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ path }}</span>
              </li>
            </ul>
          </div>

          <!-- 文件写入情况 -->
          <div v-if="hasData('files_written')" class="subsection">
            <h4 class="subsection-title">文件写入情况</h4>
            <ul class="data-list">
              <li
                v-for="(path, index) in behaviourResults.files_written"
                :key="`write-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ path }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 进程行为 -->
      <div v-if="hasProcessData()" class="data-section">
        <div class="section-header" @click="toggleSection('processes')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.processes ? '▼' : '▶' }}</span>
            进程行为:
            {{ getCount('modules_loaded') }} 个模块加载；
            {{ getCount('mutexes_created') }} 个互斥锁创建；
            {{ getCount('mutexes_opened') }} 个互斥锁打开；
            {{ getCount('permissions_requested') }} 个权限请求；
            {{ getCount('processes_terminated') }} 个进程终止；
            {{ getCount('processes_tree') }} 个进程树
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.processes" class="section-content">
          <!-- 模块加载 -->
          <div v-if="hasData('modules_loaded')" class="subsection">
            <h4 class="subsection-title">模块加载</h4>
            <ul class="data-list">
              <li
                v-for="(module, index) in behaviourResults.modules_loaded"
                :key="`module-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ module }}</span>
              </li>
            </ul>
          </div>

          <!-- 互斥锁创建 -->
          <div v-if="hasData('mutexes_created')" class="subsection">
            <h4 class="subsection-title">互斥锁创建</h4>
            <ul class="data-list">
              <li
                v-for="(mutex, index) in behaviourResults.mutexes_created"
                :key="`mutex-created-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ mutex }}</span>
              </li>
            </ul>
          </div>

          <!-- 互斥锁打开 -->
          <div v-if="hasData('mutexes_opened')" class="subsection">
            <h4 class="subsection-title">互斥锁打开</h4>
            <ul class="data-list">
              <li
                v-for="(mutex, index) in behaviourResults.mutexes_opened"
                :key="`mutex-opened-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ mutex }}</span>
              </li>
            </ul>
          </div>

          <!-- 权限请求 -->
          <div v-if="hasData('permissions_requested')" class="subsection">
            <h4 class="subsection-title">权限请求</h4>
            <ul class="data-list">
              <li
                v-for="(permission, index) in behaviourResults.permissions_requested"
                :key="`permission-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ permission }}</span>
              </li>
            </ul>
          </div>

          <!-- 进程终止 -->
          <div v-if="hasData('processes_terminated')" class="subsection">
            <h4 class="subsection-title">进程终止</h4>
            <ul class="data-list">
              <li
                v-for="(process, index) in behaviourResults.processes_terminated"
                :key="`terminated-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ process }}</span>
              </li>
            </ul>
          </div>

          <!-- 进程树 -->
          <div v-if="hasData('processes_tree')" class="subsection">
            <h4 class="subsection-title">进程树</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th width="80">程序ID</th>
                  <th>程序名称</th>
                  <th>子程序</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(process, index) in behaviourResults.processes_tree"
                  :key="`process-tree-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ process.process_id }}</span></td>
                  <td><span class="table-cell">{{ process.name }}</span></td>
                  <td><span class="table-cell">{{ process.children }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- 网络行为 -->
      <div v-if="hasNetworkData()" class="data-section">
        <div class="section-header" @click="toggleSection('network')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.network ? '▼' : '▶' }}</span>
            网络行为:
            {{ getCount('dns_lookups') }} 个DNS查找；
            {{ getCount('http_conversations') }} 个HTTP会话；
            {{ getCount('ip_traffic') }} 个IP流量；
            {{ getCount('tls') }} 个传输层安全协议
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.network" class="section-content">
          <!-- DNS查找记录 -->
          <div v-if="hasData('dns_lookups')" class="subsection">
            <h4 class="subsection-title">DNS查找记录</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th>主机名</th>
                  <th>IP地址</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(dns, index) in behaviourResults.dns_lookups"
                  :key="`dns-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ dns.hostname }}</span></td>
                  <td><span class="table-cell">{{ dns.resolved_ips }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- HTTP会话 -->
          <div v-if="hasData('http_conversations')" class="subsection">
            <h4 class="subsection-title">HTTP会话</h4>
            <table class="data-table compact-table">
              <thead>
                <tr>
                  <th width="40">序号</th>
                  <th>网址</th>
                  <th width="80">请求方法</th>
                  <th>请求头</th>
                  <th width="60">响应码</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(http, index) in behaviourResults.http_conversations"
                  :key="`http-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ http.url }}</span></td>
                  <td class="text-center">{{ http.request_method }}</td>
                  <td><span class="table-cell">{{ http.request_headers }}</span></td>
                  <td class="text-center">{{ http.response_status_code }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- IP流量 -->
          <div v-if="hasData('ip_traffic')" class="subsection">
            <h4 class="subsection-title">IP流量</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th>目标IP地址</th>
                  <th width="80">目标端口</th>
                  <th>传输层协议</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(ip, index) in behaviourResults.ip_traffic"
                  :key="`ip-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ ip.destination_ip }}</span></td>
                  <td class="text-center">{{ ip.destination_port }}</td>
                  <td><span class="table-cell">{{ ip.transport_layer_protocol }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 传输层安全协议 -->
          <div v-if="hasData('tls')" class="subsection">
            <h4 class="subsection-title">传输层安全协议</h4>
            <table class="data-table compact-table">
              <thead>
                <tr>
                  <th width="40">序号</th>
                  <th>身份信息</th>
                  <th>颁发者</th>
                  <th>序列号</th>
                  <th>哈希值</th>
                  <th>版本号</th>
                  <th>服务器</th>
                  <th>ja3</th>
                  <th>ja3s</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(tls, index) in behaviourResults.tls"
                  :key="`tls-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ tls.subject }}</span></td>
                  <td><span class="table-cell">{{ tls.issuer }}</span></td>
                  <td><span class="table-cell">{{ tls.serial_number }}</span></td>
                  <td><span class="table-cell">{{ tls.thumbprint }}</span></td>
                  <td class="text-center">{{ tls.version }}</td>
                  <td><span class="table-cell">{{ tls.sni }}</span></td>
                  <td><span class="table-cell">{{ tls.ja3 }}</span></td>
                  <td><span class="table-cell">{{ tls.ja3s }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- 攻击行为 -->
      <div v-if="hasAttackData()" class="data-section">
        <div class="section-header" @click="toggleSection('attack')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.attack ? '▼' : '▶' }}</span>
            攻击行为:
            {{ getCount('verdicts') }} 个判决结果；
            {{ attackTechniquesCount }} 个攻击技术；
            {{ getCount('ids_alerts') }} 个入侵警报；
            {{ getCount('mbc') }} 个MBC攻击；
            {{ getCount('mitre_attack_techniques') }} 个MITRE ATT&CK技术；
            {{ getCount('signature_matches') }} 个签名匹配；
            {{ getCount('system_property_lookups') }} 个系统属性查找
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.attack" class="section-content">
          <!-- 判决结果 -->
          <div v-if="hasData('verdicts')" class="subsection">
            <h4 class="subsection-title">判决结果</h4>
            <ul class="data-list">
              <li
                v-for="(verdict, index) in behaviourResults.verdicts"
                :key="`verdict-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ verdict }}</span>
              </li>
            </ul>
          </div>

          <!-- 置信度 -->
          <div v-if="behaviourResults.verdict_confidence" class="subsection">
            <h4 class="subsection-title">置信度</h4>
            <p>{{ behaviourResults.verdict_confidence }}</p>
          </div>

          <!-- 攻击技术 -->
          <div v-if="hasData('attack_techniques')" class="subsection">
            <h4 class="subsection-title">攻击技术</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th width="120">技术ID</th>
                  <th width="80">严重性</th>
                  <th>描述</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="(techItems, techId, index) in behaviourResults.attack_techniques">
                  <tr
                    v-for="(item, itemIndex) in techItems"
                    :key="`attack-${techId}-${itemIndex}`"
                    class="table-row"
                  >
                    <td class="text-center">{{ index + 1 }}</td>
                    <td><span class="table-cell">{{ techId }}</span></td>
                    <td class="text-center">{{ item.severity }}</td>
                    <td><span class="table-cell">{{ item.description }}</span></td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>

          <!-- 入侵检测系统（IDS）警报 -->
          <div v-if="hasData('ids_alerts')" class="subsection">
            <h4 class="subsection-title">入侵检测系统（IDS）警报</h4>
            <table class="data-table compact-table">
              <thead>
                <tr>
                  <th width="40">序号</th>
                  <th width="150">规则消息</th>
                  <th width="80">规则类型</th>
                  <th width="100">规则标识符</th>
                  <th width="80">告警严重性</th>
                  <th width="100">规则来源</th>
                  <th>告警上下文</th>
                  <th width="100">规则URL</th>
                  <th width="100">规则引用</th>
                  <th>规则原始数据</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(alert, index) in behaviourResults.ids_alerts"
                  :key="`ids-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ alert.rule_msg }}</span></td>
                  <td><span class="table-cell">{{ alert.rule_category }}</span></td>
                  <td><span class="table-cell">{{ alert.rule_id }}</span></td>
                  <td class="text-center">{{ alert.alert_severity }}</td>
                  <td><span class="table-cell">{{ alert.rule_source }}</span></td>
                  <td><span class="table-cell">{{ alert.alert_context }}</span></td>
                  <td><span class="table-cell">{{ alert.rule_url }}</span></td>
                  <td><span class="table-cell">{{ alert.rule_references }}</span></td>
                  <td><span class="table-cell">{{ alert.rule_raw }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- MBC攻击 -->
          <div v-if="hasData('mbc')" class="subsection">
            <h4 class="subsection-title">MBC攻击</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th width="80">ID</th>
                  <th>目标</th>
                  <th>行为</th>
                  <th>参考资料</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(mbc, index) in behaviourResults.mbc"
                  :key="`mbc-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ mbc.id }}</span></td>
                  <td><span class="table-cell">{{ mbc.objective }}</span></td>
                  <td><span class="table-cell">{{ mbc.behavior }}</span></td>
                  <td><span class="table-cell">{{ mbc.refs }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- MITRE ATT&CK技术 -->
          <div v-if="hasData('mitre_attack_techniques')" class="subsection">
            <h4 class="subsection-title">MITRE ATT&CK技术</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th width="80">ID</th>
                  <th width="80">严重性</th>
                  <th>签名描述</th>
                  <th>参考资料</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(technique, index) in behaviourResults.mitre_attack_techniques"
                  :key="`mitre-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ technique.id }}</span></td>
                  <td class="text-center">{{ technique.severity }}</td>
                  <td><span class="table-cell">{{ technique.signature_description }}</span></td>
                  <td><span class="table-cell">{{ technique.refs }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 签名匹配 -->
          <div v-if="hasData('signature_matches')" class="subsection">
            <h4 class="subsection-title">签名匹配</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th width="80">格式</th>
                  <th>名称</th>
                  <th>作者</th>
                  <th>来源</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(signature, index) in behaviourResults.signature_matches"
                  :key="`signature-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ signature.format }}</span></td>
                  <td><span class="table-cell">{{ signature.name }}</span></td>
                  <td><span class="table-cell">{{ signature.authors }}</span></td>
                  <td><span class="table-cell">{{ signature.rule_src }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 系统属性 -->
          <div v-if="hasData('system_property_lookups')" class="subsection">
            <h4 class="subsection-title">系统属性查找</h4>
            <ul class="data-list">
              <li
                v-for="(property, index) in behaviourResults.system_property_lookups"
                :key="`property-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ property }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 内存情况 -->
      <div v-if="hasMemoryData()" class="data-section">
        <div class="section-header" @click="toggleSection('memory')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.memory ? '▼' : '▶' }}</span>
            内存情况:
            {{ getCount('memory_dumps') }} 个内存转储；
            {{ getCount('memory_pattern_domains') }} 个域名模式；
            {{ getCount('memory_pattern_urls') }} 个URL模式
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.memory" class="section-content">
          <!-- 内存转储 -->
          <div v-if="hasData('memory_dumps')" class="subsection">
            <h4 class="subsection-title">内存转储</h4>
            <table class="data-table">
              <thead>
                <tr>
                  <th width="60">序号</th>
                  <th>文件名</th>
                  <th>进程</th>
                  <th width="80">大小</th>
                  <th width="100">基地址</th>
                  <th width="80">阶段</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(dump, index) in behaviourResults.memory_dumps"
                  :key="`dump-${index}`"
                  class="table-row"
                >
                  <td class="text-center">{{ index + 1 }}</td>
                  <td><span class="table-cell">{{ dump.file_name }}</span></td>
                  <td><span class="table-cell">{{ dump.process }}</span></td>
                  <td class="text-center">{{ dump.size }}</td>
                  <td><span class="table-cell">{{ dump.base_address }}</span></td>
                  <td class="text-center">{{ dump.stage }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 内存中的域名模式 -->
          <div v-if="hasData('memory_pattern_domains')" class="subsection">
            <h4 class="subsection-title">内存中的域名模式</h4>
            <ul class="data-list">
              <li
                v-for="(domain, index) in behaviourResults.memory_pattern_domains"
                :key="`domain-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ domain }}</span>
              </li>
            </ul>
          </div>

          <!-- 内存中的URL模式 -->
          <div v-if="hasData('memory_pattern_urls')" class="subsection">
            <h4 class="subsection-title">内存中的URL模式</h4>
            <ul class="data-list">
              <li
                v-for="(url, index) in behaviourResults.memory_pattern_urls"
                :key="`url-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ url }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 注册表 -->
      <div v-if="hasRegistryData()" class="data-section">
        <div class="section-header" @click="toggleSection('registry')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.registry ? '▼' : '▶' }}</span>
            注册表:
            {{ getCount('registry_keys_deleted') }} 个注册表删除；
            {{ getCount('registry_keys_opened') }} 个注册表打开；
            {{ getCount('registry_keys_set') }} 个注册表设置
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.registry" class="section-content">
          <!-- 注册表删除 -->
          <div v-if="hasData('registry_keys_deleted')" class="subsection">
            <h4 class="subsection-title">注册表删除</h4>
            <div class="list-container">
              <ul class="data-list scrollable-list">
                <li
                  v-for="(key, index) in behaviourResults.registry_keys_deleted"
                  :key="`reg-del-${index}`"
                  class="list-item"
                >
                  <span class="item-index">{{ index + 1 }}.</span>
                  <span class="item-content">{{ key }}</span>
                </li>
              </ul>
              <div v-if="getCount('registry_keys_deleted') > 5" class="scroll-indicator">
                共有 {{ getCount('registry_keys_deleted') }} 条记录，滚动查看全部
              </div>
            </div>
          </div>

          <!-- 注册表打开 -->
          <div v-if="hasData('registry_keys_opened')" class="subsection">
            <h4 class="subsection-title">注册表打开</h4>
            <div class="list-container">
              <ul class="data-list scrollable-list">
                <li
                  v-for="(key, index) in behaviourResults.registry_keys_opened"
                  :key="`reg-open-${index}`"
                  class="list-item"
                >
                  <span class="item-index">{{ index + 1 }}.</span>
                  <span class="item-content">{{ key }}</span>
                </li>
              </ul>
              <div v-if="getCount('registry_keys_opened') > 5" class="scroll-indicator">
                共有 {{ getCount('registry_keys_opened') }} 条记录，滚动查看全部
              </div>
            </div>
          </div>

          <!-- 注册表设置 -->
          <div v-if="hasData('registry_keys_set')" class="subsection">
            <h4 class="subsection-title">注册表设置</h4>
            <div class="table-container">
              <table class="data-table scrollable-table">
                <thead class="table-header">
                  <tr>
                    <th width="60">序号</th>
                    <th>Key</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody class="table-body">
                  <tr
                    v-for="(reg, index) in behaviourResults.registry_keys_set"
                    :key="`reg-set-${index}`"
                    class="table-row"
                  >
                    <td class="text-center">{{ index + 1 }}</td>
                    <td class="key-column">
                      <div class="cell-content">{{ reg.key }}</div>
                    </td>
                    <td class="value-column">
                      <div class="cell-content scrollable-cell">{{ reg.value }}</div>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div v-if="getCount('registry_keys_set') > 5" class="scroll-indicator">
                共有 {{ getCount('registry_keys_set') }} 条记录，滚动查看全部
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 加密情况 -->
      <div v-if="hasCryptoData()" class="data-section">
        <div class="section-header" @click="toggleSection('crypto')">
          <h3 class="section-title">
            <span class="toggle-icon">{{ sections.crypto ? '▼' : '▶' }}</span>
            加密情况:
            {{ getCount('crypto_algorithms_observed') }} 个加密算法；
            {{ getCount('crypto_plain_text') }} 个加密明文；
            {{ getCount('text_highlighted') }} 个高亮文本
            <span class="section-subtitle">(点击展开/收起)</span>
          </h3>
        </div>

        <div v-show="sections.crypto" class="section-content">
          <!-- 加密算法 -->
          <div v-if="hasData('crypto_algorithms_observed')" class="subsection">
            <h4 class="subsection-title">加密算法</h4>
            <ul class="data-list">
              <li
                v-for="(algorithm, index) in behaviourResults.crypto_algorithms_observed"
                :key="`crypto-alg-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ algorithm }}</span>
              </li>
            </ul>
          </div>

          <!-- 加密明文 -->
          <div v-if="hasData('crypto_plain_text')" class="subsection">
            <h4 class="subsection-title">加密明文</h4>
            <ul class="data-list">
              <li
                v-for="(text, index) in behaviourResults.crypto_plain_text"
                :key="`crypto-text-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ text }}</span>
              </li>
            </ul>
          </div>

          <!-- 高亮文本 -->
          <div v-if="hasData('text_highlighted')" class="subsection">
            <h4 class="subsection-title">高亮文本</h4>
            <ul class="data-list">
              <li
                v-for="(text, index) in behaviourResults.text_highlighted"
                :key="`text-highlight-${index}`"
                class="list-item"
              >
                <span class="item-index">{{ index + 1 }}.</span>
                <span class="item-content">{{ text }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div v-if="!isElLoadings && hasValidData" class="action-buttons">
        <button class="btn-expand-all" @click="expandAll">展开所有</button>
        <button class="btn-collapse-all" @click="collapseAll">收起所有</button>
      </div>

      <!-- 数据统计信息 -->
      <div v-if="debug && behaviourResults" class="debug-info">
        <h4>数据统计</h4>
        <ul>
          <li
            v-for="(value, key) in behaviourResults"
            v-if="Array.isArray(value)"
            :key="key"
          >
            {{ key }}: {{ value.length }} 条记录
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DynamicDetection',
  props: {
    behaviourResults: {
      type: Object,
      default: () => ({})
    },
    isElLoadings: {
      type: Boolean,
      default: false
    },
    isError: {
      type: Boolean,
      default: false
    },
    hasValidData: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      sections: {
        calls_highlighted: false,
        services: false,
        files: false,
        processes: false,
        network: false,
        attack: false,
        memory: false,
        registry: false,
        crypto: false
      },
      debug: false // 生产环境设为 false
    }
  },
  computed: {
    // 计算攻击技术总数
    attackTechniquesCount() {
      if (!this.behaviourResults.attack_techniques) return 0
      let count = 0
      Object.values(this.behaviourResults.attack_techniques).forEach(items => {
        if (Array.isArray(items)) {
          count += items.length
        }
      })
      return count
    }
  },
  watch: {
    behaviourResults: {
      handler(newVal) {
        console.log('DynamicDetection 数据更新:', newVal)
        // 数据加载后，展开第一个有数据的部分
        if (newVal && Object.keys(newVal).length > 0) {
          this.autoExpandFirstSection()
        }
      },
      deep: true,
      immediate: true
    }
  },
  methods: {
    // 检查特定字段是否有数据
    hasData(field) {
      return this.behaviourResults &&
             this.behaviourResults[field] &&
             Array.isArray(this.behaviourResults[field]) &&
             this.behaviourResults[field].length > 0
    },

    // 获取特定字段的数据条数
    getCount(field) {
      return this.hasData(field) ? this.behaviourResults[field].length : 0
    },

    // 检查文件相关数据
    hasFileData() {
      const fileFields = [
        'command_executions',
        'files_attribute_changed',
        'files_copied',
        'files_deleted',
        'files_dropped',
        'files_opened',
        'files_written'
      ]
      return fileFields.some(field => this.hasData(field))
    },

    // 检查进程相关数据
    hasProcessData() {
      const processFields = [
        'modules_loaded',
        'mutexes_created',
        'mutexes_opened',
        'permissions_requested',
        'processes_terminated',
        'processes_tree'
      ]
      return processFields.some(field => this.hasData(field))
    },

    // 检查网络相关数据
    hasNetworkData() {
      const networkFields = [
        'dns_lookups',
        'http_conversations',
        'ip_traffic',
        'tls'
      ]
      return networkFields.some(field => this.hasData(field))
    },

    // 检查攻击相关数据
    hasAttackData() {
      const attackFields = [
        'verdicts',
        'attack_techniques',
        'ids_alerts',
        'mbc',
        'mitre_attack_techniques',
        'signature_matches',
        'system_property_lookups'
      ]
      return attackFields.some(field => {
        if (field === 'attack_techniques') {
          return this.behaviourResults[field] && Object.keys(this.behaviourResults[field]).length > 0
        }
        return this.hasData(field)
      })
    },

    // 检查内存相关数据
    hasMemoryData() {
      const memoryFields = [
        'memory_dumps',
        'memory_pattern_domains',
        'memory_pattern_urls'
      ]
      return memoryFields.some(field => this.hasData(field))
    },

    // 检查注册表相关数据
    hasRegistryData() {
      const registryFields = [
        'registry_keys_deleted',
        'registry_keys_opened',
        'registry_keys_set'
      ]
      return registryFields.some(field => this.hasData(field))
    },

    // 检查加密相关数据
    hasCryptoData() {
      const cryptoFields = [
        'crypto_algorithms_observed',
        'crypto_plain_text',
        'text_highlighted'
      ]
      return cryptoFields.some(field => this.hasData(field))
    },

    // 切换部分展开/收起状态
    toggleSection(section) {
      this.sections[section] = !this.sections[section]
    },

    // 自动展开第一个有数据的部分
    autoExpandFirstSection() {
      const sectionsOrder = [
        'calls_highlighted',
        'services',
        'files',
        'processes',
        'network',
        'attack',
        'memory',
        'registry',
        'crypto'
      ]

      for (const section of sectionsOrder) {
        if (section === 'services' && (this.hasData('services_opened') || this.hasData('services_started'))) {
          this.sections[section] = true
          break
        } else if (section === 'files' && this.hasFileData()) {
          this.sections[section] = true
          break
        } else if (section === 'processes' && this.hasProcessData()) {
          this.sections[section] = true
          break
        } else if (section === 'network' && this.hasNetworkData()) {
          this.sections[section] = true
          break
        } else if (section === 'attack' && this.hasAttackData()) {
          this.sections[section] = true
          break
        } else if (section === 'memory' && this.hasMemoryData()) {
          this.sections[section] = true
          break
        } else if (section === 'registry' && this.hasRegistryData()) {
          this.sections[section] = true
          break
        } else if (section === 'crypto' && this.hasCryptoData()) {
          this.sections[section] = true
          break
        } else if (this.hasData(section)) {
          this.sections[section] = true
          break
        }
      }
    },

    // 展开所有部分（调试用）
    expandAll() {
      Object.keys(this.sections).forEach(key => {
        this.sections[key] = true
      })
    },

    // 收起所有部分
    collapseAll() {
      Object.keys(this.sections).forEach(key => {
        this.sections[key] = false
      })
    }
  }
}
</script>

<style scoped>
.dynamic-detection-container {
  margin: 20px auto;
  max-width: 1200px;
  padding: 0 20px;
}

.dynamic-content {
  background-color: #f9f9f9;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.data-section {
  margin-bottom: 20px;
  border: 1px solid #e1e1e1;
  border-radius: 6px;
  overflow: hidden;
  background-color: white;
}

.section-header {
  background-color: #f5f7fa;
  padding: 15px 20px;
  cursor: pointer;
  border-bottom: 1px solid #e1e1e1;
  transition: background-color 0.3s;
}

.section-header:hover {
  background-color: #ebf0f7;
}

.section-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
  display: flex;
  align-items: center;
}

.toggle-icon {
  margin-right: 10px;
  font-size: 12px;
  width: 20px;
  display: inline-block;
}

.section-subtitle {
  font-size: 12px;
  color: #999;
  margin-left: 10px;
  font-weight: normal;
}

.section-content {
  padding: 20px;
  background-color: white;
}

.subsection {
  margin-bottom: 25px;
}

.subsection-title {
  font-size: 15px;
  font-weight: 600;
  color: #555;
  margin: 0 0 15px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid #eee;
}

.data-list {
  list-style-type: none;
  padding: 0;
  margin: 0;
  max-height: 400px;
  overflow-y: auto;
}

.list-item {
  padding: 10px 15px;
  border-bottom: 1px solid #f0f0f0;
  font-size: 14px;
  line-height: 1.5;
  display: flex;
  align-items: flex-start;
}

.list-item:hover {
  background-color: #f9f9f9;
}

.list-item:last-child {
  border-bottom: none;
}

.item-index {
  min-width: 40px;
  color: #666;
  font-weight: 600;
}

.item-content {
  flex: 1;
  word-break: break-all;
  white-space: pre-wrap;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 15px;
}

.data-table th {
  background-color: #f8f9fa;
  padding: 10px 8px;
  text-align: left;
  font-weight: 600;
  color: #555;
  border-bottom: 2px solid #e1e1e1;
}

.data-table td {
  padding: 8px;
  border-bottom: 1px solid #eee;
  vertical-align: top;
}

.table-row:hover {
  background-color: #f9f9f9;
}

.table-cell {
  display: block;
  word-break: break-all;
  white-space: pre-wrap;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.text-center {
  text-align: center;
}

.compact-table th,
.compact-table td {
  padding: 6px 4px;
  font-size: 12px;
}

/* 注册表模块特定样式 */
.list-container {
  position: relative;
  border: 1px solid #e1e1e1;
  border-radius: 6px;
  background-color: #fff;
  overflow: hidden;
}

.table-container {
  position: relative;
  border: 1px solid #e1e1e1;
  border-radius: 6px;
  background-color: #fff;
  overflow: hidden;
  max-height: 500px;
}

.scrollable-list {
  max-height: 300px;
  overflow-y: auto;
  padding: 0;
  margin: 0;
}

.scrollable-table {
  display: block;
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.scrollable-table thead.table-header {
  display: table;
  width: 100%;
  table-layout: fixed;
  background-color: #f8f9fa;
  position: sticky;
  top: 0;
  z-index: 10;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.scrollable-table tbody.table-body {
  display: block;
  max-height: 400px;
  overflow-y: auto;
  width: 100%;
}

.scrollable-table tr {
  display: table;
  width: 100%;
  table-layout: fixed;
}

.scrollable-table th,
.scrollable-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #eee;
  vertical-align: top;
  word-break: break-word;
}

.scrollable-table th {
  font-weight: 600;
  color: #555;
  background-color: #f8f9fa;
  position: sticky;
  top: 0;
}

.scrollable-table td {
  background-color: #fff;
}

.key-column {
  width: 30%;
  min-width: 150px;
}

.value-column {
  width: 70%;
  min-width: 300px;
}

.cell-content {
  padding: 8px 0;
  line-height: 1.5;
}

.scrollable-cell {
  max-height: 150px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
  padding-right: 5px;
}

/* 滚动条样式增强 */
.scrollable-list::-webkit-scrollbar,
.scrollable-table tbody.table-body::-webkit-scrollbar,
.scrollable-cell::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.scrollable-list::-webkit-scrollbar-track,
.scrollable-table tbody.table-body::-webkit-scrollbar-track,
.scrollable-cell::-webkit-scrollbar-track {
  background: #f5f5f5;
  border-radius: 4px;
}

.scrollable-list::-webkit-scrollbar-thumb,
.scrollable-table tbody.table-body::-webkit-scrollbar-thumb,
.scrollable-cell::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.scrollable-list::-webkit-scrollbar-thumb:hover,
.scrollable-table tbody.table-body::-webkit-scrollbar-thumb:hover,
.scrollable-cell::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 滚动提示 */
.scroll-indicator {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(to top, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 50%, rgba(255,255,255,0) 100%);
  padding: 10px 15px;
  font-size: 12px;
  color: #666;
  text-align: center;
  pointer-events: none;
  z-index: 5;
}

.table-container .scroll-indicator {
  position: sticky;
  bottom: 0;
}

/* 固定表头阴影效果 */
.table-header::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(to bottom, rgba(0,0,0,0.1) 0%, rgba(0,0,0,0) 100%);
}

/* 单元格内容过长时的省略号 */
.cell-content {
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

.scrollable-cell .cell-content {
  -webkit-line-clamp: unset;
  display: block;
}

/* 序号列样式优化 */
.text-center {
  text-align: center;
  font-weight: 600;
  color: #666;
  background-color: #fafafa;
}

/* 分隔线优化 */
.subsection {
  margin-bottom: 25px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f0f0f0;
}

.subsection:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}

.isElLoading {
  text-align: center;
  padding: 40px 0;
  color: #666;
  font-size: 16px;
}

.debug-info {
  margin-top: 30px;
  padding: 15px;
  background-color: #f0f8ff;
  border-radius: 6px;
  border-left: 4px solid #1890ff;
  font-size: 12px;
  color: #666;
}

.debug-info h4 {
  margin: 0 0 10px 0;
  color: #1890ff;
}

.debug-info ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.debug-info li {
  background-color: white;
  padding: 5px 10px;
  border-radius: 4px;
  border: 1px solid #d9d9d9;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #e1e1e1;
}

.btn-expand-all,
.btn-collapse-all {
  padding: 8px 20px;
  border: 1px solid #409EFF;
  border-radius: 4px;
  background-color: white;
  color: #409EFF;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.btn-expand-all:hover,
.btn-collapse-all:hover {
  background-color: #409EFF;
  color: white;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .dynamic-detection-container {
    padding: 0 10px;
  }

  .dynamic-content {
    padding: 10px;
  }

  .section-title {
    font-size: 14px;
    flex-wrap: wrap;
  }

  .section-subtitle {
    display: block;
    width: 100%;
    margin-left: 30px;
    margin-top: 5px;
  }

  .data-table {
    font-size: 12px;
  }

  .compact-table th,
  .compact-table td {
    padding: 4px 2px;
    font-size: 11px;
  }

  .table-cell {
    max-width: 150px;
  }

  .subsection-title {
    font-size: 14px;
  }

  .action-buttons {
    flex-direction: column;
    gap: 10px;
  }

  .btn-expand-all,
  .btn-collapse-all {
    width: 100%;
  }

  /* 注册表模块响应式 */
  .table-container {
    max-height: 400px;
  }

  .scrollable-table {
    font-size: 12px;
  }

  .scrollable-table th,
  .scrollable-table td {
    padding: 8px 10px;
  }

  .key-column {
    width: 40%;
    min-width: 120px;
  }

  .value-column {
    width: 60%;
    min-width: 200px;
  }

  .scrollable-cell {
    max-height: 120px;
  }

  .scrollable-list {
    max-height: 250px;
  }
}

/* 滚动条样式 */
.data-list::-webkit-scrollbar,
.data-table::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

.data-list::-webkit-scrollbar-track,
.data-table::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.data-list::-webkit-scrollbar-thumb,
.data-table::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.data-list::-webkit-scrollbar-thumb:hover,
.data-table::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
