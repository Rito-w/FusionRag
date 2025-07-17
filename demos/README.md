# 演示和测试目录

本目录包含FusionRAG系统的各种演示和集成测试脚本。

## 文件说明

### Neo4j 集成演示
- `demo_neo4j_integration.py` - Neo4j集成完整演示
- `test_neo4j_integration.py` - Neo4j集成测试
- `test_neo4j_manual.py` - 手动Neo4j测试
- `test_neo4j_quick.py` - 快速Neo4j测试

## 使用方法

### 1. Neo4j集成演示
```bash
cd demos
python demo_neo4j_integration.py
```

展示功能：
- Neo4j连接测试
- 图数据库构建
- 实体关系抽取
- 图检索演示
- 性能对比

### 2. 集成测试
```bash
cd demos
python test_neo4j_integration.py
```

测试内容：
- 基础连接测试
- 数据写入测试
- 查询功能测试
- 错误处理测试

### 3. 手动测试
```bash
cd demos
python test_neo4j_manual.py
```

提供：
- 交互式测试环境
- 自定义查询测试
- 详细调试信息

### 4. 快速测试
```bash
cd demos
python test_neo4j_quick.py
```

执行：
- 基础功能验证
- 快速性能检查
- 简化测试流程

## 前置条件

1. **Neo4j服务运行**
   ```bash
   docker-compose up -d
   ```

2. **环境配置**
   - 确保neo4j依赖已安装
   - 检查配置文件设置
   - 验证网络连接

3. **数据准备**
   - 下载BEIR数据集
   - 完成数据预处理

## 故障排除

如果遇到连接问题：
1. 检查Neo4j服务状态
2. 验证连接配置
3. 查看日志输出
4. 尝试手动连接测试

系统会自动回退到内存模式，确保基础功能可用。