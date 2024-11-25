<template>
  <el-row :gutter="40" class="panel-group">
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('total_domain')">
        <div class="card-panel-icon-wrapper icon-people">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">  
            <span class="card-panel-text">总数量：</span>  
            <count-to :start-val="0" :end-val="56368" :duration="2600" class="card-panel-num" />  
            <span class="card-panel-text">万条</span>  
          </div>  
          <div class="card-panel-text">
            统计日期：2019年10月-2024年5月
          </div>
        </div>
      </div>
    </el-col>
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('messages')">
        <div class="card-panel-icon-wrapper icon-message">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">  
            <span class="card-panel-text">近一年数量：</span>  
            <count-to :start-val="0" :end-val="23126" :duration="2600" class="card-panel-num" />  
            <span class="card-panel-text">万条</span>  
          </div>  
          <div class="card-panel-text">
            统计日期：2023年6月-2024年5月
          </div>
        </div>
      </div>
    </el-col>
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('purchases')">
        <div class="card-panel-icon-wrapper icon-money">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">  
            <span class="card-panel-text">近一个月数量：</span>  
            <count-to :start-val="0" :end-val="2769" :duration="2600" class="card-panel-num" />  
            <span class="card-panel-text">万条</span>  
          </div>  
          <div class="card-panel-text">
            统计日期：2024年5月1日-31日
          </div>
        </div>
      </div>
    </el-col>
    <!-- <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('shoppings')">
        <div class="card-panel-icon-wrapper icon-shopping">
          <svg-icon icon-class="shopping" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-text">
            Shoppings
          </div>
          <count-to :start-val="0" :end-val="13600" :duration="3600" class="card-panel-num" />
        </div>
      </div>
    </el-col> -->
  </el-row>
</template>

<script>
import CountTo from 'vue-count-to'

export default {
  components: {
    CountTo
  },
  methods: {
    handleSetLineChartData(type) {
      this.$emit('handleSetLineChartData', type)
    }
  }
}
</script>

<style lang="scss" scoped>
.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-left: 5%;
    margin-right: 2%;
    margin-bottom: 30px;
  }

  .card-panel {
    height: 108px;
    width: 100%;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
    border-color: rgba(0, 0, 0, .05);

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-people {
        background: #40c9c6;
      }

      .icon-message {
        background: #36a3f7;
      }

      .icon-money {
        background: #f4516c;
      }

      .icon-shopping {
        background: #34bfa3
      }
    }

    .icon-people {
      color: #40c9c6;
    }

    .icon-message {
      color: #36a3f7;
    }

    .icon-money {
      color: #f4516c;
    }

    .icon-shopping {
      color: #34bfa3
    }

    .card-panel-icon-wrapper {
      float: left;
      margin: 14px 0 0 14px;
      padding: 16px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-icon {
      float: left;
      font-size: 48px;
    }

    .flex-container {  
      display: flex;  
      align-items: center; /* 垂直居中 */  
    }  
  
    .card-panel-text {  
      margin-right: 0px; /* 根据需要调整间距 */  
    }  

    .card-panel-description {  
  font-weight: bold;  
  margin: 26px 10%; /* 简化margin设置 */  
  
  .card-panel-text,  
  .card-panel-num {  
    line-height: 18px;  
    color: rgba(0, 0, 0, 0.45);  
    /* 使用vw单位设置字体大小，可以根据需要调整基数 */  
    font-size: 0.79vw; /* 示例值，根据设计需求调整 */  
    margin-bottom: 12px;  
  }  
}
  }
}

@media (max-width:550px) {
  .card-panel-description {
    display: none;
  }

  .card-panel-icon-wrapper {
    float: none !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;

    .svg-icon {
      display: block;
      margin: 14px auto !important;
      float: none !important;
    }
  }
}
</style>
