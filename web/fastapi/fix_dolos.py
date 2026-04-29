import re

file_path = 'app/services/dolos_service.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换
content = content.replace(
    'self.use_docker = self._check_docker_available()',
    'self.use_docker = True  # 强制使用Docker'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 修改完成")
