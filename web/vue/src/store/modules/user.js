import { getToken, setToken, removeToken } from '@/utils/auth'
import router, { resetRouter } from '@/router'
import service from '@/utils/request'

const state = {
  token: getToken(), // 从Cookies/LocalStorage获取Token
  name: '',
  avatar: '', // 存储用户头像地址
  introduction: '',
  roles: []
}

const mutations = {
  SET_TOKEN: (state, token) => {
    state.token = token
  },
  SET_INTRODUCTION: (state, introduction) => {
    state.introduction = introduction
  },
  SET_NAME: (state, name) => {
    state.name = name
  },
  SET_AVATAR: (state, avatar) => {
    state.avatar = avatar
  },
  SET_ROLES: (state, roles) => {
    state.roles = roles
  }
}

const actions = {
  // 用户登录（适配后端接口+同步Token/头像存储）
  login({ commit, dispatch }, userInfo) {
    const { username, password } = userInfo
    return new Promise((resolve, reject) => {
      service.post('/api/login', {
        username: username.trim(),
        password: password
      }).then(response => {
        const { token, username, avatar } = response.data // 假设接口返回avatar字段
        // 存储Token和头像
        commit('SET_TOKEN', token)
        commit('SET_NAME', username)
        commit('SET_AVATAR', avatar) // 保存头像到Vuex
        setToken(token) // 同步到Cookies/LocalStorage

        // 登录成功后加载用户信息和动态路由
        dispatch('getInfo').then(() => {
          dispatch('permission/generateRoutes', state.roles).then(accessRoutes => {
            router.addRoutes(accessRoutes) // 动态添加路由
          })
        })

        resolve()
      }).catch(error => {
        reject(error)
      })
    })
  },

  // 获取用户信息（固定admin角色+头像兜底）
  getInfo({ commit, state }) {
    return new Promise((resolve) => {
      // 若接口未返回头像，使用默认头像兜底
      const defaultAvatar = 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif'
      // 固定返回管理员角色，适配权限逻辑
      const data = {
        roles: ['admin'],
        name: state.name || 'admin',
        avatar: state.avatar || defaultAvatar, // 优先用接口返回，否则用默认
        introduction: '系统管理员'
      }
      // 更新用户状态
      commit('SET_ROLES', data.roles)
      commit('SET_NAME', data.name)
      commit('SET_AVATAR', data.avatar)
      commit('SET_INTRODUCTION', data.introduction)
      resolve(data)
    })
  },

  // 用户登出（清除所有存储+重置路由）
  logout({ commit, dispatch }) {
    return new Promise(resolve => {
      // 清除Token和头像
      commit('SET_TOKEN', '')
      commit('SET_AVATAR', '')
      commit('SET_ROLES', [])
      removeToken() // 同步清理所有存储
      // 重置路由
      resetRouter()
      // 清空标签页
      dispatch('tagsView/delAllViews', null, { root: true })
      resolve()
    })
  },

  // 重置Token（用于过期/退出登录）
  resetToken({ commit }) {
    return new Promise(resolve => {
      commit('SET_TOKEN', '')
      commit('SET_AVATAR', '')
      commit('SET_ROLES', [])
      removeToken() // 同步清理所有存储
      resolve()
    })
  },

  // 动态修改角色（保留原有逻辑，适配权限切换）
  async changeRoles({ commit, dispatch }, role) {
    const token = role + '-token'
    // 更新Token并存储
    commit('SET_TOKEN', token)
    setToken(token)
    // 获取新角色信息
    const { roles } = await dispatch('getInfo')
    // 重置并加载新路由
    resetRouter()
    const accessRoutes = await dispatch('permission/generateRoutes', roles, { root: true })
    router.addRoutes(accessRoutes)
    // 清空标签页
    dispatch('tagsView/delAllViews', null, { root: true })
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
