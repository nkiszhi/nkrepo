<script>
import { onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

export default {
  setup() {
    const route = useRoute()
    const router = useRouter()

    onMounted(() => {
      const { params, query } = route
      const { path } = params
      router.replace({ path: '/' + path, query })
    })

    return () => null
  }
}
</script>
