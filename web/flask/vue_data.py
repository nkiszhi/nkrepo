import os
import configparser
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# 读取配置文件获取数据库信息（当前路径下的config.ini）
def get_db_config():
    config = configparser.ConfigParser()
    # 读取当前路径下的config.ini文件
    config_path = os.path.join(os.path.dirname(__file__) if os.path.dirname(__file__) else os.getcwd(), 'config.ini')
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        return None
    
    config.read(config_path, encoding='utf-8')
    
    # 从[mysql]部分获取配置
    try:
        return {
            'host': config.get('mysql', 'host'),
            'database': config.get('mysql', 'db'),
            'user': config.get('mysql', 'user'),
            'password': config.get('mysql', 'passwd'),
            'port': 3306,
            'pool_name': 'mypool',
            'pool_size': 8,
            'autocommit': True
        }
    except Exception as e:
        print(f"读取配置文件出错: {e}")
        return None

# 确保保存目录存在
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# 获取保存文件的路径（../vue/src/data）
def get_save_path():
    current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else os.getcwd()
    # 构建上级目录下的vue/src/data路径
    save_dir = os.path.join(current_dir, '../vue/src/data')
    # 标准化路径（处理相对路径中的..）
    save_dir = os.path.normpath(save_dir)
    return ensure_directory_exists(save_dir)

DB_CONFIG = get_db_config()

def create_db_pool():
    """创建数据库连接池"""
    if not DB_CONFIG:
        print("数据库配置无效，无法创建连接池")
        return None
        
    try:
        pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
        print(f"成功创建数据库连接池，大小: {DB_CONFIG['pool_size']}")
        return pool
    except Error as e:
        print(f"创建连接池时发生错误: {e}")
    return None

def process_single_table(table_name, current_year, pool):
    """处理单个表格并返回统计结果"""
    try:
        # 从连接池获取连接
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # 合并查询：一次获取所有需要的字段
        query = f"""
        SELECT src_file, date, category, platform, family,
               filetype, detection, behaviour_summary, behaviour_mitre_trees
        FROM {table_name} 
        WHERE src_file LIKE 'VirusShare_%' OR src_file LIKE '%ZGC%' OR src_file LIKE '%QAX%'
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        # 初始化表格级别的统计数据
        year_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        category_counter = Counter()
        platform_counter = Counter()
        family_counter = Counter()
        table_samples = 0
        table_recent_samples = 0
        
        # 新增统计变量：细分每种文件类型的卡巴结果
        filetype_kaspersky = {
            'exe32': {'total': 0, 'has_result': 0, 'no_result': 0},
            'exe64': {'total': 0, 'has_result': 0, 'no_result': 0},
            'dll32': {'total': 0, 'has_result': 0, 'no_result': 0},
            'dll64': {'total': 0, 'has_result': 0, 'no_result': 0},
            'total': {'total': 0, 'has_result': 0, 'no_result': 0}  # 汇总
        }
        
        # 其他行为统计
        detection_1 = 0
        behaviour_summary_1 = 0
        behaviour_mitre_trees_1 = 0
        
        # 处理查询结果
        for row in results:
            table_samples += 1
            date_str = row['date']
            
            # 解析日期
            date_obj = None
            if date_str:
                date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%Y-%m-%d %H:%M:%S']
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            
            # 按年份统计
            if date_obj:
                year = date_obj.year
                year_counts[year] += 1
                
                # 近一年数据统计
                if date_obj.year == current_year:
                    month = date_obj.month
                    month_key = f"{current_year}-{month:02d}"
                    monthly_counts[month_key] += 1
                    table_recent_samples += 1
            
            # 处理分类数据
            if row['category'] and str(row['category']).lower() != 'nan':
                category_counter[row['category']] += 1
            if row['platform'] and str(row['platform']).lower() != 'nan':
                platform_counter[row['platform']] += 1
            if row['family'] and str(row['family']).lower() != 'nan':
                family_counter[row['family']] += 1
            
            # 处理filetype和卡巴结果统计
            filetype = row['filetype']
            if filetype:
                filetype_str = str(filetype).lower()
                # 检查是否为我们关注的四种类型
                if filetype_str in ['exe32', 'exe64', 'dll32', 'dll64']:
                    # 检查是否有卡巴结果
                    has_result = 1 if (row['category'] and str(row['category']).lower() != 'nan') else 0
                    no_result = 0 if has_result else 1
                    
                    # 更新该类型的统计
                    filetype_kaspersky[filetype_str]['total'] += 1
                    filetype_kaspersky[filetype_str]['has_result'] += has_result
                    filetype_kaspersky[filetype_str]['no_result'] += no_result
                    
                    # 更新总计
                    filetype_kaspersky['total']['total'] += 1
                    filetype_kaspersky['total']['has_result'] += has_result
                    filetype_kaspersky['total']['no_result'] += no_result
            
            # 处理行为相关统计
            if row['detection'] is not None:
                try:
                    if int(row['detection']) == 1:
                        detection_1 += 1
                except (ValueError, TypeError):
                    pass
            
            if row['behaviour_summary'] is not None:
                try:
                    if int(row['behaviour_summary']) == 1:
                        behaviour_summary_1 += 1
                except (ValueError, TypeError):
                    pass
            
            if row['behaviour_mitre_trees'] is not None:
                try:
                    if int(row['behaviour_mitre_trees']) == 1:
                        behaviour_mitre_trees_1 += 1
                except (ValueError, TypeError):
                    pass
        
        return {
            'year_counts': year_counts,
            'monthly_counts': monthly_counts,
            'category_counter': category_counter,
            'platform_counter': platform_counter,
            'family_counter': family_counter,
            'samples': table_samples,
            'recent_samples': table_recent_samples,
            'table_name': table_name,
            'filetype_kaspersky': filetype_kaspersky,
            'detection_1': detection_1,
            'behaviour_summary_1': behaviour_summary_1,
            'behaviour_mitre_trees_1': behaviour_mitre_trees_1
        }
        
    except Error as e:
        print(f"处理表格 {table_name} 时出错: {e}")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

def merge_results(results):
    """合并多个表格的统计结果"""
    total_year_counts = defaultdict(int)
    total_monthly_counts = defaultdict(int)
    total_category = Counter()
    total_platform = Counter()
    total_family = Counter()
    total_samples = 0
    total_recent = 0
    
    # 初始化合并的文件类型和卡巴结果统计
    total_filetype_kaspersky = {
        'exe32': {'total': 0, 'has_result': 0, 'no_result': 0},
        'exe64': {'total': 0, 'has_result': 0, 'no_result': 0},
        'dll32': {'total': 0, 'has_result': 0, 'no_result': 0},
        'dll64': {'total': 0, 'has_result': 0, 'no_result': 0},
        'total': {'total': 0, 'has_result': 0, 'no_result': 0}
    }
    
    # 其他行为统计合并变量
    total_detection_1 = 0
    total_behaviour_summary_1 = 0
    total_behaviour_mitre_trees_1 = 0
    
    for result in results:
        if not result:
            continue
            
        # 合并年份统计
        for year, count in result['year_counts'].items():
            total_year_counts[year] += count
            
        # 合并月度统计
        for month, count in result['monthly_counts'].items():
            total_monthly_counts[month] += count
            
        # 合并分类统计
        total_category.update(result['category_counter'])
        total_platform.update(result['platform_counter'])
        total_family.update(result['family_counter'])
        
        # 累计样本数
        total_samples += result['samples']
        total_recent += result['recent_samples']
        
        # 合并文件类型和卡巴结果统计
        ft = result['filetype_kaspersky']
        for filetype in ['exe32', 'exe64', 'dll32', 'dll64', 'total']:
            total_filetype_kaspersky[filetype]['total'] += ft[filetype]['total']
            total_filetype_kaspersky[filetype]['has_result'] += ft[filetype]['has_result']
            total_filetype_kaspersky[filetype]['no_result'] += ft[filetype]['no_result']
        
        # 合并行为统计
        total_detection_1 += result['detection_1']
        total_behaviour_summary_1 += result['behaviour_summary_1']
        total_behaviour_mitre_trees_1 += result['behaviour_mitre_trees_1']
    
    return {
        'year_counts': total_year_counts,
        'monthly_counts': total_monthly_counts,
        'category_counter': total_category,
        'platform_counter': total_platform,
        'family_counter': total_family,
        'total_samples': total_samples,
        'recent_year_samples': total_recent,
        'filetype_kaspersky': total_filetype_kaspersky,
        'detection_1': total_detection_1,
        'behaviour_summary_1': total_behaviour_summary_1,
        'behaviour_mitre_trees_1': total_behaviour_mitre_trees_1
    }

def process_pie_top10(counter):
    """处理饼图数据：取Top10，其余合并为Others"""
    all_items = counter.most_common()
    if len(all_items) <= 10:
        return [{'name': name, 'value': count} for name, count in all_items]
    top10 = all_items[:10]
    others_count = sum(count for _, count in all_items[10:])
    top10_with_others = [{'name': name, 'value': count} for name, count in top10]
    top10_with_others.append({'name': 'Others', 'value': others_count})
    return top10_with_others

def process_full_pie(counter):
    """处理完整分类数据（所有类别）"""
    return [{'name': name, 'value': count} for name, count in counter.most_common()]

def prepare_chart_data(processed_data, current_year):
    """准备图表所需的数据格式（适配Vue组件）"""
    # 1. 年份折线图数据
    year_items = sorted(processed_data['year_counts'].items(), key=lambda x: x[0])
    year_chart_data = {
        'date_data': [f"{item[0]}年" for item in year_items],
        'amount_data': [item[1] for item in year_items]
    }
    
    # 2. 近一年月度折线图数据
    all_months = [f"{current_year}-{month:02d}" for month in range(1, 13)]
    full_monthly = {month: 0 for month in all_months}
    for month, count in processed_data['monthly_counts'].items():
        if month in full_monthly:
            full_monthly[month] = count
    sorted_monthly = sorted(full_monthly.items())
    monthly_chart_data = {
        'date_data': [f"{item[0].split('-')[0]}年{item[0].split('-')[1]}月" for item in sorted_monthly],
        'amount_data': [item[1] for item in sorted_monthly]
    }
    
    # 3. 饼图数据
    category_top10 = process_pie_top10(processed_data['category_counter'])
    category_full = process_full_pie(processed_data['category_counter'])
    platform_top10 = process_pie_top10(processed_data['platform_counter'])
    platform_full = process_full_pie(processed_data['platform_counter'])
    family_top10 = process_pie_top10(processed_data['family_counter'])
    family_full = process_full_pie(processed_data['family_counter'])
    
    # 整理文件类型和卡巴结果统计数据
    filetype_stats = {}
    for filetype in ['exe32', 'exe64', 'dll32', 'dll64', 'total']:
        data = processed_data['filetype_kaspersky'][filetype]
        # 计算有卡巴结果的百分比
        percentage = round(
            (data['has_result'] / data['total'] * 100) 
            if data['total'] > 0 else 0, 2
        )
        filetype_stats[filetype] = {
            'total': data['total'],
            'has_result': data['has_result'],
            'no_result': data['no_result'],
            'percentage': percentage
        }
    
    # 行为统计数据
    behavior_stats = {
        'detection_1': processed_data['detection_1'],
        'behaviour_summary_1': processed_data['behaviour_summary_1'],
        'behaviour_mitre_trees_1': processed_data['behaviour_mitre_trees_1']
    }
    
    # 整合所有数据
    return {
        'line_chart': {
            'total_amount': year_chart_data,
            'year_amount': monthly_chart_data
        },
        'pie_top10': {
            'category': category_top10,
            'platform': platform_top10,
            'family': family_top10
        },
        'pie_full': {
            'category': category_full,
            'platform': platform_full,
            'family': family_full
        },
        'filetype_stats': filetype_stats,
        'behavior_stats': behavior_stats,
        'summary': {
            'total_samples': processed_data['total_samples'],
            'recent_year_samples': processed_data['recent_year_samples'],
            'current_year': current_year,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

def save_to_js(data, filename='chart_data.js'):
    """生成Vue可用的JS文件（ES6模块格式）"""
    try:
        save_dir = get_save_path()
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("export default {\n")
            
            # 写入折线图数据
            f.write("  lineChartData: {\n")
            f.write(f"    total_amount: {data['line_chart']['total_amount']},\n")
            f.write(f"    year_amount: {data['line_chart']['year_amount']}\n")
            f.write("  },\n")
            
            # 写入饼图Top10数据
            f.write("  pieTop10Data: {\n")
            f.write(f"    category: {data['pie_top10']['category']},\n")
            f.write(f"    platform: {data['pie_top10']['platform']},\n")
            f.write(f"    family: {data['pie_top10']['family']}\n")
            f.write("  },\n")
            
            # 写入完整分类数据
            f.write("  pieFullData: {\n")
            f.write(f"    category: {data['pie_full']['category']},\n")
            f.write(f"    platform: {data['pie_full']['platform']},\n")
            f.write(f"    family: {data['pie_full']['family']}\n")
            f.write("  },\n")
            
            # 写入文件类型和卡巴结果统计数据
            f.write("  filetypeStats: {\n")
            for i, filetype in enumerate(['exe32', 'exe64', 'dll32', 'dll64', 'total']):
                stats = data['filetype_stats'][filetype]
                f.write(f"    {filetype}: {{\n")
                f.write(f"      total: {stats['total']},\n")
                f.write(f"      hasResult: {stats['has_result']},\n")
                f.write(f"      noResult: {stats['no_result']},\n")
                f.write(f"      percentage: {stats['percentage']}\n")
                # 最后一个不需要逗号
                f.write(f"    }}{',' if i < 4 else ''}\n")
            f.write("  },\n")
            
            # 写入行为统计数据
            f.write("  behaviorStats: {\n")
            f.write(f"    detection1: {data['behavior_stats']['detection_1']},\n")
            f.write(f"    behaviourSummary1: {data['behavior_stats']['behaviour_summary_1']},\n")
            f.write(f"    behaviourMitreTrees1: {data['behavior_stats']['behaviour_mitre_trees_1']}\n")
            f.write("  },\n")
            
            # 写入统计概览
            f.write(f"  summary: {data['summary']}\n")
            
            f.write("}")
        print(f"Vue可用JS文件已保存到 {file_path}")
    except Exception as e:
        print(f"保存JS文件时出错: {e}")

def save_to_txt(data, filename='stats_summary.txt'):
    """生成TXT统计文件（方便查看详细数据）"""
    try:
        save_dir = get_save_path()
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # 1. 统计概览
            f.write("="*50 + "\n")
            f.write("统计概览\n")
            f.write("-"*50 + "\n")
            f.write(f"总样本数: {data['summary']['total_samples']:,}\n")
            f.write(f"{data['summary']['current_year']}年样本数: {data['summary']['recent_year_samples']:,}\n")
            f.write(f"数据生成时间: {data['summary']['generated_at']}\n")
            f.write("\n"*2)
            
            # 2. 文件类型与卡巴结果统计
            f.write("="*50 + "\n")
            f.write("文件类型与卡巴结果统计\n")
            f.write("-"*50 + "\n")
            # 表头
            f.write(f"{'文件类型':<10} | {'总数':<10} | {'有卡巴结果':<12} | {'无卡巴结果':<12} | {'占比(%)':<8}\n")
            f.write("-"*50 + "\n")
            # 详细数据
            for filetype in ['exe32', 'exe64', 'dll32', 'dll64']:
                stats = data['filetype_stats'][filetype]
                f.write(f"{filetype:<10} | {stats['total']:<10,} | {stats['has_result']:<12,} | {stats['no_result']:<12,} | {stats['percentage']:<8}\n")
            # 总计
            f.write("-"*50 + "\n")
            total = data['filetype_stats']['total']
            f.write(f"{'总计':<10} | {total['total']:<10,} | {total['has_result']:<12,} | {total['no_result']:<12,} | {total['percentage']:<8}\n")
            f.write("\n"*2)
            
            # 3. 行为特征统计
            f.write("="*50 + "\n")
            f.write("行为特征统计（值为1的数量）\n")
            f.write("-"*50 + "\n")
            bs = data['behavior_stats']
            f.write(f"detection: {bs['detection_1']:,}\n")
            f.write(f"behaviour_summary: {bs['behaviour_summary_1']:,}\n")
            f.write(f"behaviour_mitre_trees: {bs['behaviour_mitre_trees_1']:,}\n")
            f.write("\n"*2)
            
            # 4. 年份分布
            f.write("="*50 + "\n")
            f.write("年份分布\n")
            f.write("-"*50 + "\n")
            f.write(f"{'年份':<10} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for year, amount in zip(
                data['line_chart']['total_amount']['date_data'],
                data['line_chart']['total_amount']['amount_data']
            ):
                f.write(f"{year:<10} | {amount:15,}\n")
            f.write("\n"*2)
            
            # 5. 近一年月度分布
            f.write("="*50 + "\n")
            f.write(f"{data['summary']['current_year']}年月度分布\n")
            f.write("-"*50 + "\n")
            f.write(f"{'月份':<12} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for month, amount in zip(
                data['line_chart']['year_amount']['date_data'],
                data['line_chart']['year_amount']['amount_data']
            ):
                f.write(f"{month:<12} | {amount:15,}\n")
            f.write("\n"*2)
            
            # 6. Category完整分类
            f.write("="*50 + "\n")
            f.write("Category完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pie_full']['category']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
            f.write("\n"*2)
            
            # 7. Platform完整分类
            f.write("="*50 + "\n")
            f.write("Platform完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pie_full']['platform']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
            f.write("\n"*2)
            
            # 8. Family完整分类
            f.write("="*50 + "\n")
            f.write("Family完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pie_full']['family']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
        
        print(f"TXT统计文件已保存到 {file_path}")
    except Exception as e:
        print(f"保存TXT文件时出错: {e}")

def main():
    """主函数：协调各步骤执行"""
    start_time = datetime.now()
    print(f"===== 开始执行数据提取和处理 =====")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查数据库配置
    if not DB_CONFIG:
        print("数据库配置无效，程序退出")
        return
    
    # 获取当前年份
    current_year = datetime.now().year
    print(f"当前年份: {current_year}")
    
    # 创建数据库连接池
    pool = create_db_pool()
    if not pool:
        print("无法创建数据库连接池，程序退出")
        return
    
    try:
        # 生成256个表格名称: sample_00到sample_ff
        table_names = [f"sample_{i:02x}" for i in range(256)]
        print(f"共需处理 {len(table_names)} 个表格")
        
        # 使用线程池并发处理表格
        results = []
        max_workers = min(DB_CONFIG['pool_size'], 32)
        print(f"使用 {max_workers} 个线程并发处理...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_table, table, current_year, pool): table 
                for table in table_names
            }
            
            for future in as_completed(futures):
                table_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"已完成: {table_name} (样本数: {result['samples']})")
                except Exception as e:
                    print(f"处理 {table_name} 时发生异常: {e}")
        
        # 合并所有结果
        print("\n===== 开始合并统计结果 =====")
        merged_data = merge_results(results)
        merged_data['current_year'] = current_year
        
        # 准备图表数据（适配Vue）
        chart_data = prepare_chart_data(merged_data, current_year)
        
        # 保存为JS和TXT
        save_to_js(chart_data)
        save_to_txt(chart_data)
        
        # 打印统计信息和执行时间
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print("\n===== 数据统计完成 =====")
        print(f"总样本量: {merged_data['total_samples']:,}")
        print(f"{current_year}年样本量: {merged_data['recent_year_samples']:,}")
        print(f"年份数量: {len(merged_data['year_counts'])}")
        print(f"category类别数量: {len(merged_data['category_counter'])}")
        print(f"platform类别数量: {len(merged_data['platform_counter'])}")
        print(f"family类别数量: {len(merged_data['family_counter'])}")
        
        # 打印文件类型和卡巴结果统计
        print("\n文件类型与卡巴结果统计:")
        print(f"{'类型':<10} | {'总数':<10} | {'有结果':<10} | {'无结果':<10} | {'占比(%)'}")
        print("-"*55)
        for filetype in ['exe32', 'exe64', 'dll32', 'dll64']:
            stats = merged_data['filetype_kaspersky'][filetype]
            percentage = round((stats['has_result'] / stats['total'] * 100) if stats['total'] > 0 else 0, 2)
            print(f"{filetype:<10} | {stats['total']:<10,} | {stats['has_result']:<10,} | {stats['no_result']:<10,} | {percentage}%")
        # 打印总计
        total = merged_data['filetype_kaspersky']['total']
        total_percentage = round((total['has_result'] / total['total'] * 100) if total['total'] > 0 else 0, 2)
        print("-"*55)
        print(f"{'总计':<10} | {total['total']:<10,} | {total['has_result']:<10,} | {total['no_result']:<10,} | {total_percentage}%")
        
        # 打印行为统计
        print(f"\n行为特征统计:")
        print(f"detection=1: {merged_data['detection_1']:,}")
        print(f"behaviour_summary=1: {merged_data['behaviour_summary_1']:,}")
        print(f"behaviour_mitre_trees=1: {merged_data['behaviour_mitre_trees_1']:,}")
        
        print(f"\n执行时间: {elapsed:.2f} 秒")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        print("\n程序执行结束")

if __name__ == "__main__":
    main()