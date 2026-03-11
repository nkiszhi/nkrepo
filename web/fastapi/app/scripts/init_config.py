#!/usr/bin/env python3
# -*-coding: utf-8-*-
#数据库配置存放初始化脚本
import os
import sys
import configparser
import pymysql
import bcrypt

def init_database():
    """初始化数据库配置表"""
    
    # 读取config.ini
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    cp = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"❌ 错误：配置文件 {config_path} 不存在")
        return False
    
    cp.read(config_path, encoding='utf-8')
    
    # 获取数据库连接信息
    db_config = {
        'host': cp.get('mysql', 'host', fallback='localhost'),
        'user': cp.get('mysql', 'user', fallback='root'),
        'password': cp.get('mysql', 'passwd', fallback=''),
        'database': cp.get('mysql', 'db_web', fallback='webdatadb'),
        'charset': cp.get('mysql', 'charset', fallback='utf8mb4')
    }
    
    print(f"📊 数据库配置: {db_config['host']}/{db_config['database']}")
    
    try:
        # 连接数据库
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        print("✅ 数据库连接成功")
        
        # 检查表是否存在
        cursor.execute("SHOW TABLES LIKE 'system_config'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("📝 创建配置表...")
            # 创建配置表 - 修复SQL语法
            create_table_sql = """
            CREATE TABLE `system_config` (
              `id` int(11) NOT NULL AUTO_INCREMENT,
              `config_key` varchar(100) NOT NULL,
              `config_value` text,
              `config_module` varchar(50) NOT NULL,
              `description` varchar(200) DEFAULT NULL,
              `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
              `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (`id`),
              UNIQUE KEY `unique_key_module` (`config_key`,`config_module`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_table_sql)
            print("✅ 配置表创建完成")
        else:
            print("✅ 配置表已存在")
        
        # 清空表（可选，如果需要重置的话）
        # cursor.execute("TRUNCATE TABLE system_config")
        # print("✅ 清空配置表")
        
        # 检查表是否为空
        cursor.execute("SELECT COUNT(*) FROM system_config")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📝 插入初始配置数据...")
            
            # 初始配置数据
            initial_configs = [
                # fixed_user 配置
                ('username', 'admin', 'fixed_user', '系统管理员用户名'),
                ('password', bcrypt.hashpw('123456'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'), 'fixed_user', '系统管理员密码（bcrypt加密）'),
                
                # API 配置
                ('vt_key', '', 'api', 'VirusTotal API密钥'),
                
                # AI Provider 配置
                ('openai_api_key', '', 'ai_provider', 'OpenAI API密钥'),
                ('openai_base_url', 'https://api.closeai-asia.com/v1', 'ai_provider', 'OpenAI API地址'),
                ('openai_model', 'gpt-4o', 'ai_provider', 'OpenAI模型'),
                ('claude_api_key', '', 'ai_provider', 'Claude API密钥'),
                ('claude_base_url', 'https://api.openai-proxy.org/anthropic', 'ai_provider', 'Claude API地址'),
                ('claude_model', 'claude-3-opus-20240229', 'ai_provider', 'Claude模型'),
                ('default_ai_provider', 'openai', 'ai_provider', '默认AI提供商')
            ]
            
            # 插入数据
            for key, value, module, description in initial_configs:
                # 检查是否已存在
                cursor.execute(
                    "SELECT id FROM system_config WHERE config_key = %s AND config_module = %s",
                    (key, module)
                )
                exists = cursor.fetchone()
                
                if exists:
                    # 更新现有配置
                    update_sql = """
                    UPDATE system_config 
                    SET config_value = %s, description = %s 
                    WHERE config_key = %s AND config_module = %s
                    """
                    cursor.execute(update_sql, (value, description, key, module))
                    print(f"  → 更新配置: {module}.{key}")
                else:
                    # 插入新配置
                    insert_sql = """
                    INSERT INTO system_config (config_key, config_value, config_module, description)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(insert_sql, (key, value, module, description))
                    print(f"  → 插入配置: {module}.{key}")
            
            conn.commit()
            print(f"✅ 成功初始化 {len(initial_configs)} 条配置")
        else:
            print(f"📊 配置表中已有 {count} 条记录，跳过初始化")
        
        # 显示配置
        cursor.execute("SELECT config_module, config_key, config_value FROM system_config ORDER BY config_module, config_key")
        rows = cursor.fetchall()
        
        print("\n📋 当前配置表内容:")
        print("-" * 80)
        current_module = None
        for module, key, value in rows:
            if module != current_module:
                print(f"\n[{module}]")
                current_module = module
            
            # 隐藏敏感信息
            display_value = value
            if 'key' in key.lower() or 'password' in key.lower():
                if value and len(value) > 5:
                    display_value = value[:3] + '***' + value[-3:] if value else '空'
                else:
                    display_value = '空'
            elif value is None:
                display_value = '空'
            
            print(f"  {key:30} = {display_value}")
        print("-" * 80)
        
        cursor.close()
        conn.close()
        
        return True
        
    except pymysql.err.OperationalError as e:
        if "Unknown database" in str(e):
            print(f"❌ 数据库 {db_config['database']} 不存在，请先创建数据库")
            print(f"   创建命令: CREATE DATABASE {db_config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        else:
            print(f"❌ 数据库连接失败: {str(e)}")
        return False
    except pymysql.err.ProgrammingError as e:
        print(f"❌ SQL语法错误: {str(e)}")
        print("请检查您的MySQL版本，可能需要调整SQL语句")
        return False
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_mysql_version():
    """检查MySQL版本"""
    try:
        # 读取config.ini
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        cp = configparser.ConfigParser()
        if not os.path.exists(config_path):
            return None
        
        cp.read(config_path, encoding='utf-8')
        
        # 获取数据库连接信息
        db_config = {
            'host': cp.get('mysql', 'host', fallback='localhost'),
            'user': cp.get('mysql', 'user', fallback='root'),
            'password': cp.get('mysql', 'passwd', fallback=''),
            'database': cp.get('mysql', 'db_web', fallback='webdatadb'),
            'charset': cp.get('mysql', 'charset', fallback='utf8mb4')
        }
        
        # 连接数据库
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"📊 MySQL版本: {version}")
        return version
    except Exception as e:
        print(f"❌ 无法获取MySQL版本: {str(e)}")
        return None

if __name__ == '__main__':
    print("🚀 开始初始化数据库配置...")
    print("=" * 60)
    
    # 检查MySQL版本
    mysql_version = check_mysql_version()
    
    if init_database():
        print("\n✅ 初始化成功！")
        print("\n📝 下一步：")
        print("1. 启动后端服务: python app.py")
        print("2. 登录系统 (默认账号: admin, 密码: 123456)")
        print("3. 访问 '系统设置' 页面配置API密钥")
    else:
        print("\n❌ 初始化失败！")
        sys.exit(1)