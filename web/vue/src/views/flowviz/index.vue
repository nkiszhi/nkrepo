<!-- vue/src/views/flowviz/index.vue -->
<template>
  <div class="flowviz-container">
    <transition name="fade" mode="out-in">
      <router-view />
    </transition>
  </div>
</template>

<script>
export default {
  name: 'FlowVizLayout',
  data() {
    return {}
  }
}
</script>

<style scoped>
.flowviz-container {
  height: 100%;
  width: 100%;
  overflow: hidden;
}

.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s;
}
.fade-enter, .fade-leave-to {
  opacity: 0;
}
</style>
