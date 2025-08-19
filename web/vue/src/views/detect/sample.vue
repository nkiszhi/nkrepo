<template>  
  <main>  
    <div class="text-center">   
      <h2 class="text-primary">基于可信度评估的多模型恶意文件检测</h2>    
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>    
    </div>   
    <div>
      <input ref="file-upload-input" class="file-upload-input" type="file"  @change="handleClick">
      <div class="drop" @drop="handleDrop" @dragover="handleDragover">
        把待检文件拖到这里或
        <el-button :loading="loading" style="margin-left:0%;font-size: 20px;" size="mini" type="primary" @click="handleUpload">
          选择待检文件
        </el-button>
      </div>
      <div v-if="uploadResult">  
        <table class="file-info-table"> 
          <th>文件特征</th> 
          <th> </th> 
          <tr>  
            <td>文件名称：</td>  
            <td>{{ uploadResult.original_filename }}</td>  
          </tr>  
          <tr>  
            <td>文件大小：</td>  
            <td>{{ uploadResult.file_size }}</td>  
          </tr>  
          <tr v-for="(value, key) in uploadResult.query_result" :key="key" v-if="value !== 'nan' && value !== 'NaN'">  
            <td>{{ key.replace('_', ' ') }}：</td>  
            <td>{{ value }}</td>  
          </tr>  
        </table>   
        <table class="detection-result-table">  
          <thead>  
            <tr>  
              <th>检测模型</th>  
              <th>良性概率</th>  
              <th>恶意概率</th> 
              <th>结果</th>
            </tr>  
          </thead>  
          <tbody>  
            <tr v-for="(probabilities, model) in uploadResult.exe_result" :key="model">  
              <td>{{ model }}</td>  
              <td>{{ probabilities[0] }}</td>  
              <td>{{ probabilities[1] }}</td>
              <td :class="probabilities[0] > probabilities[1] ? 'text-success' : 'text-danger'">  
                <i :class="probabilities[0] > probabilities[1] ? 'fas fa-check' : 'fas fa-exclamation-triangle'"></i>  
                {{ probabilities[0] > probabilities[1] ? '安全' : '危险' }}  
              </td>  
            </tr>
          </tbody>  
        </table>
        <div v-if="isLoading" class="isLoading">  
          <p>正在加载数据...</p>  
        </div> 
        
        <!-- 杀毒软件检测结果 -->   
        <div v-if="!isErrors && results.length > 0">  
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
                <td > <span>{{ result.engine_name }}</span> </td>
                <td > <span>{{ result.engine_version }}</span> </td>  
                <td > <span>{{ result.engine_update }}</span> </td>
                <td > <span>{{ result.method}}</span> </td>
                <td style="text-align: left;">  
                  <span v-if="result.result && result.result !== ''" :class="getCategoryColorClass(result.category)">  
                    <svg-icon :icon-class="getIconClass(result.category)" /> {{ result.result }} </span>  
                  <span v-else-if="result.category":class="getCategoryColorClass(result.category)">  
                    <svg-icon :icon-class="getIconClass(result.category)" /> {{ result.category }} </span>  
                  <span v-else> <i class="fa fa-times" aria-hidden="true"></i>N/A</span>  
                </td>  
              </tr>  
            </tbody>  
          </table>  
        </div> 
        <div v-if="isLoadings" class="isLoading">  
          <p>正在加载数据...</p>  
        </div> 
        
        <div v-if="!isError">  
          <div v-if="typeof behaviour_results === 'object' && Object.keys(behaviour_results).length > 0">  
            <h2 style="text-align: center;">动态检测结果</h2>  
            <!-- API调用情况 -->  
            <div v-if="behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0" style="text-align: center;"> 
              <h3 @click="toggleCall = !toggleCall" :title="toggleCall ? '点击收起' : '点击显示详情'"  style="cursor: pointer;" >API调用情况:{{ behaviour_results.calls_highlighted ? behaviour_results.calls_highlighted.length : 0 }}个API调用</h3>
              <div v-if="toggleCall &&behaviour_results && behaviour_results.calls_highlighted && behaviour_results.calls_highlighted.length > 0"> 
                <ul class="flex-container">  
                  <li v-for="(calls_highlighted, index) in behaviour_results.calls_highlighted" :key="index" style="text-align: left;" class="list-row">  
                    {{ calls_highlighted }}  
                  </li>  
                </ul>  
              </div>
            </div>

            <!-- 服务情况 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0) ||  
              (behaviour_results && behaviour_results.services_started && behaviour_results.services_started.length > 0)   
            " style="text-align: center;"> 
              <h3 @click="toggleServicesOpened = !toggleServicesOpened"  :title="toggleServicesOpened ? '点击收起' : '点击显示详情'"  style="cursor: pointer;"  >
                服务情况:{{ behaviour_results.services_opened ? behaviour_results.services_opened.length : 0 }}个打开的服务；{{ behaviour_results.services_started ? behaviour_results.services_started.length : 0 }} 个启动的服务
              </h3> 
              <div v-if="toggleServicesOpened &&behaviour_results && behaviour_results.services_opened && behaviour_results.services_opened.length > 0">      
                <h4>打开的服务</h4>  
                <ul class="flex-container">  
                  <li v-for="(services_opened, index) in behaviour_results.services_opened" :key="index"  class="list-row">  
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
            
            <!-- 文件行为 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.command_executions && behaviour_results.command_executions.length > 0) ||  
              (behaviour_results && behaviour_results.files_attribute_changed && behaviour_results.files_attribute_changed.length > 0) || 
              (behaviour_results && behaviour_results.files_copied && behaviour_results.files_copied.length > 0) ||
              (behaviour_results && behaviour_results.files_deleted && behaviour_results.files_deleted.length > 0)||
              (behaviour_results && behaviour_results.files_dropped && behaviour_results.files_dropped.length > 0)||
              (behaviour_results && behaviour_results.files_opened && behaviour_results.files_opened.length > 0)||
              (behaviour_results && behaviour_results.files_written && behaviour_results.files_written.length > 0)
            " style="text-align: center;"> 
              <h3 @click="toggleFiles = !toggleFiles"  :title="toggleFiles ? '点击收起' : '点击显示详情'"  style="cursor: pointer;">
                文件行为:{{ behaviour_results.command_executions ? behaviour_results.command_executions.length : 0 }}个执行；{{ behaviour_results.files_attribute_changed ? behaviour_results.files_attribute_changed.length : 0 }} 个属性变更；{{ behaviour_results.files_copied ? behaviour_results.files_copied.length : 0 }} 个复制；{{ behaviour_results.files_deleted ? behaviour_results.files_deleted.length : 0 }} 个删除；{{ behaviour_results.files_dropped ? behaviour_results.files_dropped.length : 0 }} 个释放；{{ behaviour_results.files_opened ? behaviour_results.files_opened.length : 0 }} 个打开；{{ behaviour_results.files_written ? behaviour_results.files_written.length : 0 }} 个写入
              </h3>    
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
                      <td>{{ index + 1 }}</td>  
                      <td><span>{{ files_copied.key }}</span></td>  
                      <td><span>{{ files_copied.value }}</span></td>      
                    </tr>  
                  </tbody>  
                </table> 
              </div>  
              
              <div v-if="toggleFiles && behaviour_results && behaviour_results.files_deleted && behaviour_results.files_deleted.length > 0" >  
                <h4>文件删除情况</h4>  
                <ul class="flex-container">  
                  <li v-for="(path_del, index) in behaviour_results.files_deleted" :key="index"  class="list-row">  
                    {{ path_del}}  
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
                      <td>{{ index + 1 }}</td>  
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

            <!-- 进程行为 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.modules_loaded && behaviour_results.modules_loaded.length > 0) ||  
              (behaviour_results && behaviour_results.mutexes_created && behaviour_results.mutexes_created.length > 0) || 
              (behaviour_results && behaviour_results.mutexes_opened && behaviour_results.mutexes_opened.length > 0) ||
              (behaviour_results && behaviour_results.permissions_requested && behaviour_results.permissions_requested.length > 0)||
              (behaviour_results && behaviour_results.processes_terminated && behaviour_results.processes_terminated.length > 0)||
              (behaviour_results && behaviour_results.processes_tree && behaviour_results.processes_tree.length > 0)
            " style="text-align: center;"> 
              <h3 @click="toggleProcesses = !toggleProcesses" :title="toggleProcesses ? '点击收起' : '点击显示详情'"  style="cursor: pointer;" >
                进程行为:{{ behaviour_results.modules_loaded ? behaviour_results.modules_loaded.length : 0 }}个模块加载；{{ behaviour_results.mutexes_created ? behaviour_results.mutexes_created.length : 0 }} 个互斥锁创建；{{ behaviour_results.mutexes_opened ? behaviour_results.mutexes_opened.length : 0 }} 个互斥锁打开；{{ behaviour_results.permissions_requested ? behaviour_results.permissions_requested.length : 0 }} 个权限请求；{{ behaviour_results.processes_terminated ? behaviour_results.processes_terminated.length : 0 }} 个进程终止；{{ behaviour_results.processes_tree ? behaviour_results.processes_tree.length : 0 }} 进程树
              </h3>
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
                      <td>{{ index + 1 }}</td>  
                      <td><span>{{ processes_tree.process_id }}</span></td>  
                      <td><span>{{ processes_tree.name }}</span></td> 
                      <td><span>{{ processes_tree.children }}</span></td>      
                    </tr>  
                  </tbody>  
                </table> 
              </div>
            </div>

            <!-- 网络行为 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.dns_lookups && behaviour_results.dns_lookups.length > 0) ||  
              (behaviour_results && behaviour_results.http_conversations && behaviour_results.http_conversations.length > 0) || 
              (behaviour_results && behaviour_results.ip_traffic && behaviour_results.ip_traffic.length > 0) ||
              (behaviour_results && behaviour_results.tls && behaviour_results.tls.length > 0)
            " style="text-align: center;">  
              <h3 @click="toggleNetwork = !toggleNetwork" :title="toggleNetwork ? '点击收起' : '点击显示详情'"  style="cursor: pointer;">
                网络行为:{{ behaviour_results.dns_lookups ? behaviour_results.dns_lookups.length : 0 }}个DNS查找；{{ behaviour_results.http_conversations ? behaviour_results.http_conversations.length : 0 }} 个HTTP会话；{{ behaviour_results.ip_traffic ? behaviour_results.ip_traffic.length : 0 }} 个IP流量；{{ behaviour_results.tls ? behaviour_results.tls.length : 0 }} 传输层安全协议
              </h3> 
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
                      <td>{{ index + 1 }}</td>  
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
                      <td>{{ index + 1 }}</td>  
                      <td><span>{{ http_conversations.url}}</span></td>  
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
                      <td>{{ index + 1 }}</td>  
                      <td><span>{{ ip_traffic.destination_ip}}</span></td>  
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
                      <td><span>{{ tls.subject}}</span></td>  
                      <td><span>{{ tls.issuer }}</span></td>    
                      <td><span>{{ tls.serial_number }}</span></td>  
                      <td><span>{{ tls.thumbprint}}</span></td>  
                      <td><span>{{ tls.version }}</span></td>    
                      <td><span>{{ tls.sni }}</span></td>
                      <td><span>{{ tls.ja3}}</span></td>  
                      <td><span>{{ tls.ja3s }}</span></td>     
                    </tr>  
                  </tbody>  
                </table> 
              </div>
            </div>

            <!-- 攻击行为 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.verdicts && behaviour_results.verdicts.length > 0) ||  
              (behaviour_results && behaviour_results.verdict_confidence && behaviour_results.verdict_confidence.length > 0) || 
              (behaviour_results && behaviour_results.attack_techniques && behaviour_results.attack_techniques.length > 0 )||
              (behaviour_results && behaviour_results.ids_alerts && behaviour_results.ids_alerts.length > 0)||
              (behaviour_results && behaviour_results.mbc && behaviour_results.mbc.length > 0)||
              (behaviour_results && behaviour_results.mitre_attack_techniques && behaviour_results.mitre_attack_techniques.length > 0)||
              (behaviour_results && behaviour_results.signature_matches && behaviour_results.signature_matches.length > 0)||
              (behaviour_results && behaviour_results.system_property_lookups && behaviour_results.system_property_lookups.length > 0)||
              (behaviour_results && behaviour_results.tags && behaviour_results.tags.length > 0)||
              (behaviour_results && behaviour_results.verdict_labels && behaviour_results.verdict_labels.length > 0)" style="text-align: center;">
              <h3 @click="toggleAttack = !toggleAttack" :title="toggleAttack ? '点击收起' : '点击显示详情'"  style="cursor: pointer;" >
                攻击行为:{{ behaviour_results.verdicts ? behaviour_results.verdicts.length : 0 }} 个判决结果；{{ behaviour_results.attack_techniques ? behaviour_results.attack_techniques.length : 0 }} 个攻击技术；{{ behaviour_results.ids_alerts ? behaviour_results.ids_alerts.length : 0 }} 个入侵警报；{{ behaviour_results.mbc ? behaviour_results.mbc.length : 0 }} 个mbc；{{ behaviour_results.mitre_attack_techniques ? behaviour_results.mitre_attack_techniques.length : 0 }} 个MITRE ATT&CK技术；{{ behaviour_results.system_property_lookups ? behaviour_results.system_property_lookups.length : 0 }} 个系统属性查找；{{ behaviour_results.tags ? behaviour_results.tags.length : 0 }} 个标签；{{ behaviour_results.verdict_labels ? behaviour_results.verdict_labels.length : 0 }} 个判决标签
              </h3> 
              <div v-if="toggleAttack && behaviour_results && behaviour_results.verdicts && behaviour_results.verdicts.length > 0">      
                <h4>判决结果</h4> 
                <ul class="flex-container">  
                  <li v-for="(verdicts, index) in behaviour_results.verdicts" :key="index" class="list-row">  
                    {{ verdicts }}  
                  </li>  
                </ul>  
              </div>
              
              <div v-if="toggleAttack && behaviour_results && behaviour_results.verdict_confidence && behaviour_results.verdict_confidence.length > 0">      
                <h4>置信度{{ behaviour_results.verdict_confidence}}</h4>
              </div>
              
              <div v-if="toggleAttack && behaviour_results && behaviour_results.attack_techniques && behaviour_results.attack_techniques.length > 0">      
                <h4>攻击技术</h4>  
                <table class="behaviour_result_table">  
                  <thead>  
                    <tr>  
                      <th style="width:10%;">序号</th>
                      <th style="width:20%;">id</th>
                      <th style="width:20%;">严重性</th>
                      <th style="width:20%;">签名</th>
                      <th style="width:30%;">参考资料</th>
                    </tr>  
                  </thead>  
                  <tbody>  
                    <tr v-for="(attack_techniques, index) in behaviour_results.attack_techniques" :key="index" class="behaviour_result_table_row">    
                      <td>{{ index + 1 }}</td>  
                      <td><span>{{ attack_techniques.id }}</span></td>  
                      <td><span>{{ attack_techniques.severity }}</span></td>   
                      <td><span>{{ attack_techniques.signature_description }}</span></td> 
                      <td><span>{{ attack_techniques.refs }}</span></td>  
                    </tr>  
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
                      <td>{{ index + 1 }}</td>  
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
                      <td>{{ index + 1 }}</td>  
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

              <div v-if="toggleAttack && behaviour_results && behaviour_results.tags && behaviour_results.tags.length > 0">      
                <h4>标签</h4>  
                <ul class="flex-container">  
                  <li v-for="(tags, index) in behaviour_results.tags" :key="index" class="list-row">  
                    {{ tags }}  
                  </li>  
                </ul>  
              </div>

              <div v-if="toggleAttack && behaviour_results && behaviour_results.verdict_labels && behaviour_results.verdict_labels.length > 0">      
                <h4>判决标签</h4>  
                <ul class="flex-container">  
                  <li v-for="(verdict_labels, index) in behaviour_results.verdict_labels" :key="index" class="list-row">  
                    {{ verdict_labels }}  
                  </li>  
                </ul>  
              </div>
            </div>

            <!-- 内存情况 -->  
            <div v-if="  
              (behaviour_results && behaviour_results.memory_dumps && behaviour_results.memory_dumps.length > 0) ||  
              (behaviour_results && behaviour_results.memory_pattern_domains && behaviour_results.memory_pattern_domains.length > 0) ||  
              (behaviour_results && behaviour_results.memory_pattern_urls && behaviour_results.memory_pattern_urls.length > 0)  
            " style="text-align: center;"> 
              <h3 @click="toggleMemory = !toggleMemory" :title="toggleMemory ? '点击收起' : '点击显示详情'"  style="cursor: pointer;" >
                内存情况:{{ behaviour_results.memory_dumps ? behaviour_results.memory_dumps.length : 0 }} 个内容转储；{{ behaviour_results.memory_pattern_domains ? behaviour_results.memory_pattern_domains.length : 0 }} 个域名模式；{{ behaviour_results.memory_pattern_urls ? behaviour_results.memory_pattern_urls.length : 0 }} 个URL模式
              </h3> 
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

            <!-- 注册表 -->  
            <div style="text-align: center;"> 
              <h3 @click="toggleRegistry = !toggleRegistry" :title="toggleRegistry ? '点击收起' : '点击显示详情'"  style="cursor: pointer;" >
                注册表:{{ behaviour_results.registry_keys_deleted ? behaviour_results.registry_keys_deleted.length : 0 }} 个注册表删除；{{ behaviour_results.registry_keys_opened ? behaviour_results.registry_keys_opened.length : 0 }} 个注册表打开；{{ behaviour_results.registry_keys_set ? behaviour_results.registry_keys_set.length : 0 }} 个注册表设置
              </h3> 
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

            <!-- 加密情况 -->  
            <div style="text-align: center;"> 
              <h3 @click="toggleCrypto = !toggleCrypto"  :title="toggleCrypto ? '点击收起' : '点击显示详情'"  style="cursor: pointer;">
                加密情况:{{ behaviour_results.crypto_algorithms_observed ? behaviour_results.crypto_algorithms_observed.length : 0 }} 个加密算法；{{ behaviour_results.crypto_plain_text ? behaviour_results.crypto_plain_text.length : 0 }} 个加密明文；{{ behaviour_results.text_highlighted ? behaviour_results.text_highlighted.length : 0 }} 个高亮文本
              </h3> 

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
          </div>     
        </div>
        <div v-else-if="typeof behaviour_results === 'object' && behaviour_results.message">  
          <!-- 显示错误信息 -->  
          <p>{{ behaviour_results.message }}</p>  
        </div>  
      </div>    
    </div>
  </main>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
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
      isError: false,
      isErrors: false, // 补充之前遗漏的属性
      apiBaseUrl: 'http://xxxxxx:5005' // 默认API地址
    }
  },
  created() {
    // 组件创建时加载配置文件
    this.loadConfig();
  },
  methods: {
    // 加载配置文件获取API地址
    async loadConfig() {
      try {
        // 与其他页面保持一致的路径解析逻辑
        const response = await axios.get('/config.ini', {
          responseType: 'text'
        });
        
        // 解析INI格式内容
        const configContent = response.data;
        const lines = configContent.split('\n');
        let inApiSection = false;
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          // 查找[api]部分
          if (trimmedLine === '[api]') {
            inApiSection = true;
            continue;
          }
          
          // 在[api]部分下查找baseUrl配置
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=');
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim();
              console.log('从配置文件加载API地址:', this.apiBaseUrl);
              break;
            }
          }
          
          // 遇到其他部分则退出查找
          if (inApiSection && trimmedLine.startsWith('[')) {
            break;
          }
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用默认API地址:', error.message);
        // 继续使用默认地址
      }
    },

    getIconClass(category) {  
      switch (category) {  
        case 'malicious':  
        case 'suspicious':  
          return 'vt_malicious';  
        case 'undetected':  
        case 'harmless':  
          return 'vt_undetected';  
        default:  
          return 'vt_type-unsupported';  
      }  
    },
    
    getCategoryColorClass(category) {  
      switch (category) {  
        case 'malicious':  
        case 'suspicious':  
          return 'red-text';  
        case 'undetected':  
        case 'harmless':  
          return 'black-text';  
        default:  
          return 'gray-text';  
      }  
    },  

    handleDrop(e) {
      e.stopPropagation();
      e.preventDefault();
      if (this.loading) return;
      const files = e.dataTransfer.files;
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!');
        return;
      }
      const rawFile = files[0];
      this.upload(rawFile);
    },

    handleDragover(e) {
      e.stopPropagation();
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    },

    handleUpload() {
      this.$refs['file-upload-input'].click();
    },

    handleClick(e) {
      const files = e.target.files;
      const rawFile = files[0];
      if (!rawFile) return;
      this.upload(rawFile);
    },

    async upload(rawFile) {    
      this.uploadResult = null;  
      this.loading = true;    
      const formData = new FormData();    
      formData.append('file', rawFile);    
      
      try {      
        // 使用从配置文件读取的API地址
        const response = await fetch(`${this.apiBaseUrl}/upload`, {      
          method: 'POST',      
          body: formData,      
        });    
        
        console.log('Response status:', response.status);    
        console.log('Response headers:', response.headers);
        this.$forceUpdate();
      
        if (!response.ok) {      
          throw new Error('Failed to upload file: ' + response.statusText);      
        }  

        const data = await response.json();  
        console.log('Response data:', data);  
        this.uploadResult = data;
        await this.fetchDetailAPI(); 
        this.$forceUpdate();  
      } catch (error) {      
        console.error('Error uploading file:', error);  
        if (error instanceof Error) {  
          console.error('Error message:', error.message);  
          console.error('Error stack:', error.stack);  
        }  
        this.$message.error('文件上传失败！');    
      } finally {    
        this.loading = false;    
      }   
    },

    fetchDetailAPI() {  
      this.isLoading = true;
      this.results = []; 
      this.behaviour_results = {};
      this.isError = false;
      this.isErrors = false;
      
      if (this.uploadResult && this.uploadResult.query_result && this.uploadResult.VT_API) {  
        const sha256 = this.uploadResult.query_result.SHA256;  
        const VT_API = this.uploadResult.VT_API;  
        console.log('sha256:', sha256);  
        console.log('VT_API:', VT_API);  

        // 使用从配置文件读取的API地址
        axios.get(`${this.apiBaseUrl}/detection_API/${sha256}`, { params: { VT_API: VT_API } })  
          .then(detectionResponse => {  
            if (Array.isArray(detectionResponse.data) && detectionResponse.data.length > 0) {  
              this.results = detectionResponse.data;  
            } else {  
              this.error = 'Unexpected response format from detection API';  
              this.isErrors = true;  
            }  
          })  
          .catch(error => {  
            this.isErrors = true;  
            console.error('Error fetching detection data:', error);  
            this.error = 'Error fetching data from detection API';  
          })  
          .finally(() => {  
            this.checkAndUpdateUI();  
          });  

        // 使用从配置文件读取的API地址
        axios.get(`${this.apiBaseUrl}/behaviour_API/${sha256}`, { params: { VT_API: VT_API } })  
          .then(behaviourResponse => {  
            if (typeof behaviourResponse.data === 'object' && !Array.isArray(behaviourResponse.data)) {  
              this.behaviour_results = behaviourResponse.data;  
            } else if (typeof behaviourResponse.data === 'object' && behaviourResponse.data.message) {  
              this.isError = true;  
              this.behaviour_results = behaviourResponse.data;  
            } else {  
              this.isError = true;  
              this.error = 'Unexpected response format from behaviour API';  
            }  
          })  
          .catch(error => {  
            this.isError = true;  
            console.error('Error fetching behaviour data:', error);  
            this.error = 'Error fetching data from behaviour API';  
          }) 
          .finally(() => {  
            this.checkAndUpdateUIs();  
          });  

        this.checkAndUpdateUIs = () => {  
          if (!this.isLoadings) return;  
          this.$forceUpdate();  
          this.isLoadings = false;  
        }; 

        this.checkAndUpdateUI = () => {  
          if (!this.isLoading) return;  
          this.$forceUpdate();  
          this.isLoading = false;  
        };   

        this.isLoading = true;
        this.isLoadings = true;   
      }
    } 
  },  
};
</script>

<style scoped>
/* 样式部分保持不变 */
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

.fas.fa-exclamation-triangle {  
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
} 

.isLoading {  
  text-align: center;    
}

.flex-container {  
  display: flex;  
  flex-direction: column;  
  align-items: center;  
  list-style: none;  
  padding: 0;  
  margin: 0;  
  text-align: left;  
} 

.list-row {  
  margin-bottom: 3px;  
  width:90%;  
  position: relative;   
} 

.list-row > * {  
  white-space: normal;  
  word-break: break-all;  
  text-indent: 1em;
}

h3 {  
  color: black;  
  width: 90%;  
  margin: 0 auto;  
  text-align: left;  
  padding-top: 10px;  
  padding-bottom: 10px;  
  border-top: 1px solid grey;  
  border-bottom: 1px solid grey; 
} 

h4 {  
  color: black; 
  width:90%; 
  margin: 0 auto;
  text-align: left;
  padding-top: 10px;  
  padding-bottom: 0px;  
}

.behaviour_result_table {  
  width: 90%; 
  border-bottom: 0px solid;
  margin-top: 5px;
}
</style>