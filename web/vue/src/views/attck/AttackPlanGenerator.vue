<template>
  <div class="attack-plan-container">
    <div class="page-header">
      <h1 class="page-title">AI 智能分析</h1>
      <p class="page-description">
        使用大语言模型进行代码解释和攻击方案构建
      </p>
    </div>

    <el-row :gutter="24">
      <!-- 代码分析 -->
      <el-col :xs="24" :lg="12">
        <el-card class="analysis-card">
          <div slot="header" class="card-header">
            <span style="display: flex; align-items: center;">
              <i class="el-icon-s-data" style="margin-right: 8px;" />
              <span>代码分析</span>
            </span>
          </div>

          <el-form
            ref="codeAnalysisForm"
            :model="codeAnalysisForm"
            :rules="codeAnalysisRules"
            label-width="100px"
          >
            <el-form-item label="选择函数" prop="function_ids">
              <el-select
                v-model="codeAnalysisForm.function_ids"
                multiple
                placeholder="请选择函数ID"
                style="width: 100%;"
              >
                <el-option
                  v-for="func in functionList"
                  :key="func.id"
                  :label="`${func.alias} (${func.hash_id})`"
                  :value="func.id"
                />
              </el-select>
            </el-form-item>

            <el-form-item label="分析类型" prop="analysis_type">
              <el-select v-model="codeAnalysisForm.analysis_type" style="width: 100%;">
                <el-option label="代码解释" value="code_explanation" />
                <el-option label="攻击场景" value="attack_scenario" />
                <el-option label="缓解措施" value="mitigation" />
              </el-select>
            </el-form-item>

            <el-form-item label="模型选择" prop="model">
              <el-select v-model="codeAnalysisForm.model" style="width: 100%;">
                <el-option label="GPT-4" value="gpt-4" />
                <el-option label="GPT-3.5 Turbo" value="gpt-3.5-turbo" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                icon="el-icon-s-data"
                :loading="codeAnalysisLoading"
                style="width: 100%;"
                @click="handleCodeAnalysis"
              >
                开始分析
              </el-button>
            </el-form-item>
          </el-form>

          <!-- 分析结果 -->
          <div v-if="analysisResults.length > 0" class="analysis-results">
            <el-divider>分析结果</el-divider>
            <div v-for="(result, index) in analysisResults" :key="index" class="result-item">
              <el-card shadow="never">
                <div class="result-header">
                  <h4>函数 ID: {{ result.function_id }}</h4>
                  <div>
                    <el-tag type="success" size="small">
                      置信度: {{ (result.confidence_score * 100).toFixed(1) }}%
                    </el-tag>
                    <el-tag v-if="result.cached" type="info" size="small" style="margin-left: 8px;">
                      缓存
                    </el-tag>
                  </div>
                </div>
                <div class="result-content">
                  {{ result.result }}
                </div>
                <div class="result-meta">
                  <span>模型: {{ result.model_used }}</span>
                  <span style="margin-left: 16px;">Token: {{ result.token_usage }}</span>
                </div>
              </el-card>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 攻击方案生成 -->
      <el-col :xs="24" :lg="12">
        <el-card class="analysis-card">
          <div slot="header" class="card-header">
            <span style="display: flex; align-items: center;">
              <i class="el-icon-s-promotion" style="margin-right: 8px;" />
              <span>攻击方案生成</span>
            </span>
          </div>

          <el-form
            ref="attackPlanForm"
            :model="attackPlanForm"
            :rules="attackPlanRules"
            label-width="100px"
          >
            <el-form-item label="攻击目标" prop="objective">
              <el-input
                v-model="attackPlanForm.objective"
                type="textarea"
                :rows="3"
                placeholder="描述您的攻击目标，例如：获取系统管理员权限"
              />
            </el-form-item>

            <el-form-item label="选择技术" prop="selected_techniques">
              <el-select
                v-model="attackPlanForm.selected_techniques"
                multiple
                filterable
                placeholder="选择要使用的ATT&CK技术"
                style="width: 100%;"
              >
                <el-option
                  v-for="tech in techniques"
                  :key="tech.technique_id"
                  :label="`${tech.technique_id}: ${tech.technique_name}`"
                  :value="tech.technique_id"
                />
              </el-select>
            </el-form-item>

            <el-form-item label="约束条件">
              <el-input
                v-model="attackPlanForm.constraints"
                type="textarea"
                :rows="2"
                placeholder="输入约束条件，多个条件用逗号分隔"
              />
            </el-form-item>

            <el-form-item label="环境描述">
              <el-input
                v-model="attackPlanForm.environment"
                type="textarea"
                :rows="2"
                placeholder="描述目标环境，例如：Windows 10 企业版，防火墙开启"
              />
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                icon="el-icon-s-opportunity"
                :loading="attackPlanLoading"
                style="width: 100%;"
                @click="handleAttackPlan"
              >
                生成攻击方案
              </el-button>
            </el-form-item>
          </el-form>

          <!-- 攻击方案结果 -->
          <div v-if="attackPlanResult" class="attack-plan-results">
            <el-divider>攻击方案</el-divider>
            <el-alert
              title="⚠️ 警告"
              description="此分析仅用于防御研究和安全测试目的，请勿用于恶意攻击。"
              type="warning"
              show-icon
              style="margin-bottom: 16px;"
            />

            <el-collapse v-model="activeCollapse">
              <el-collapse-item title="🎯 执行步骤" name="steps">
                <ol class="steps-list">
                  <li v-for="(step, index) in attackPlanResult.execution_steps" :key="index">
                    {{ step }}
                  </li>
                </ol>
              </el-collapse-item>

              <el-collapse-item title="🔧 技术组合" name="techniques">
                <div v-for="(tech, index) in attackPlanResult.techniques" :key="index" class="technique-item">
                  <el-tag type="primary">{{ tech.technique_id }}</el-tag>
                  <span style="margin-left: 8px;">{{ tech.technique_name }}</span>
                </div>
              </el-collapse-item>

              <el-collapse-item title="⚠️ 风险评估" name="risk">
                <p>{{ attackPlanResult.risk_assessment }}</p>
              </el-collapse-item>

              <el-collapse-item title="🛡️ 缓解建议" name="mitigation">
                <ul>
                  <li v-for="(advice, index) in attackPlanResult.mitigation_advice" :key="index">
                    {{ advice }}
                  </li>
                </ul>
              </el-collapse-item>
            </el-collapse>

            <div class="plan-meta">
              <span>方案ID: {{ attackPlanResult.plan_id }}</span>
              <span style="margin-left: 16px;">Token: {{ attackPlanResult.token_usage }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import attckApi from '@/api/attck'

export default {
  name: 'AttackPlanGenerator',
  data() {
    return {
      // 代码分析相关
      codeAnalysisForm: {
        function_ids: [],
        analysis_type: 'code_explanation',
        model: 'gpt-4',
        temperature: 0.7
      },
      codeAnalysisRules: {
        function_ids: [
          { required: true, message: '请选择要分析的函数', trigger: 'change' }
        ]
      },
      codeAnalysisLoading: false,
      analysisResults: [],
      functionList: [
        { id: 1, alias: 'MalAPI_LzmaDecompressor', hash_id: 'abc123' },
        { id: 2, alias: 'MalAPI_Commandlineparser', hash_id: 'def456' },
        { id: 3, alias: 'MalAPI_Threadpoolworkercleanup', hash_id: 'ghi789' }
      ],

      // 攻击方案相关
      attackPlanForm: {
        objective: '',
        selected_techniques: [],
        constraints: '',
        environment: ''
      },
      attackPlanRules: {
        objective: [
          { required: true, message: '请描述攻击目标', trigger: 'blur' }
        ],
        selected_techniques: [
          { required: true, message: '请选择ATT&CK技术', trigger: 'change' }
        ]
      },
      attackPlanLoading: false,
      attackPlanResult: null,
      activeCollapse: ['steps', 'techniques', 'risk', 'mitigation'],

      // 技术列表
      techniques: [],
      loading: false
    }
  },
  created() {
    this.loadTechniques()
  },
  methods: {
    // 加载技术列表
    async loadTechniques() {
      try {
        this.loading = true
        const res = await attckApi.getTechniquesList()
        this.techniques = res.data || []
      } catch (error) {
        console.error('加载技术列表失败:', error)
        this.$message.error('加载技术列表失败')
      } finally {
        this.loading = false
      }
    },

    // 执行代码分析
    async handleCodeAnalysis() {
      try {
        const valid = await this.$refs.codeAnalysisForm.validate()
        if (!valid) return

        this.codeAnalysisLoading = true

        // 构造请求参数
        const requestData = {
          ...this.codeAnalysisForm,
          temperature: parseFloat(this.codeAnalysisForm.temperature)
        }

        // 调用API
        const res = await attckApi.analyzeCode(requestData)
        this.analysisResults = res.data || []

        this.$message.success('分析完成')
      } catch (error) {
        console.error('代码分析失败:', error)
        this.$message.error('代码分析失败')
      } finally {
        this.codeAnalysisLoading = false
      }
    },

    // 执行攻击方案生成
    async handleAttackPlan() {
      try {
        const valid = await this.$refs.attackPlanForm.validate()
        if (!valid) return

        this.attackPlanLoading = true

        // 构造请求参数
        const requestData = {
          ...this.attackPlanForm,
          constraints: this.attackPlanForm.constraints
            ? this.attackPlanForm.constraints.split(',').map(s => s.trim())
            : [],
          model: 'gpt-4',
          temperature: 0.7
        }

        // 调用API
        const res = await attckApi.createAttackPlan(requestData)
        this.attackPlanResult = res.data

        this.$message.success('攻击方案生成成功')
      } catch (error) {
        console.error('生成攻击方案失败:', error)
        this.$message.error('生成攻击方案失败')
      } finally {
        this.attackPlanLoading = false
      }
    }
  }
}
</script>

<style scoped>
.attack-plan-container {
  padding: 20px;
  background: #f0f2f5;
  min-height: calc(100vh - 60px);
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: #1890ff;
  margin: 0 0 8px 0;
}

.page-description {
  color: #666;
  font-size: 14px;
  margin: 0;
}

.analysis-card {
  margin-bottom: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card-header {
  font-weight: 600;
  font-size: 16px;
}

.analysis-results {
  margin-top: 20px;
}

.result-item {
  margin-bottom: 16px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.result-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.result-content {
  color: #595959;
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 12px;
}

.result-meta {
  font-size: 12px;
  color: #8c8c8c;
}

.attack-plan-results {
  margin-top: 20px;
}

.steps-list {
  padding-left: 20px;
  line-height: 1.8;
}

.steps-list li {
  margin-bottom: 8px;
}

.technique-item {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.plan-meta {
  margin-top: 16px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
  font-size: 12px;
  color: #8c8c8c;
}

.el-collapse-item {
  margin-bottom: 8px;
}
</style>
