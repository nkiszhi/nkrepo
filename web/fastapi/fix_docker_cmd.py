file_path = 'app/services/dolos_service.py'

with open(file_path, 'r') as f:
    content = f.read()

# 修复：删除多余的"dolos"
content = content.replace(
    '"ghcr.io/dodona-edu/dolos:latest",\n            "dolos", "run"',
    '"ghcr.io/dodona-edu/dolos:latest",\n            "run"'
)

with open(file_path, 'w') as f:
    f.write(content)

print("✓ 修复完成")
