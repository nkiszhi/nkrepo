import Cookies from 'js-cookie'

// 关键：统一 Token Key（和登录逻辑保持一致，或修改登录逻辑匹配此 Key）
const TokenKey = 'token' // 改为和 localStorage 相同的 Key，或保持 Admin-Token 但同步修改登录逻辑

// 从 Cookies 获取 Token（保持你原有存储方式）
export function getToken() {
  return Cookies.get(TokenKey)
}

// 存储 Token 到 Cookies（覆盖原有逻辑，确保统一）
export function setToken(token) {
  // 同时存储到 localStorage（兼容登录逻辑，双重保险）
  localStorage.setItem(TokenKey, token)
  sessionStorage.setItem(TokenKey, token)
  // 保留你原有Cookies存储
  return Cookies.set(TokenKey, token, { expires: 1 }) // expires:1 表示1天有效期
}

// 移除 Token（同步清理所有存储位置）
export function removeToken() {
  localStorage.removeItem(TokenKey)
  sessionStorage.removeItem(TokenKey)
  return Cookies.remove(TokenKey)
}
