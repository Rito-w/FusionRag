# 实现计划

## 0. 实验脚本优化与拆分 (高优先级)

- [x] 0.1 拆分实验脚本
  - 将run_experiments.py拆分为独立的脚本文件
  - 创建run_standard_experiments.py（标准性能评估）
  - 创建run_ablation_experiments.py（消融实验）
  - 创建run_query_analysis_experiments.py（查询类型分析）
  - 每个脚本支持独立运行和配置
  - _需求: 6.1, 6.2_

- [x] 0.2 创建轻量级配置文件
  - 创建lightweight_config.json（内存优化配置）
  - 创建quick_test_config.json（快速测试配置）
  - 优化模型选择和批处理大小
  - 禁用资源密集型功能（如文档扩展）
  - _需求: 6.2_

- [x] 0.3 实现渐进式测试功能
  - 创建progressive_test.py脚本
  - 支持小样本测试（5-10个查询）
  - 支持单数据集测试
  - 支持逐步扩展测试规模
  - 添加内存使用监控
  - _需求: 6.1, 6.2_

- [x] 0.4 内存和性能优化
  - 实现批处理文档编码，避免一次性加载所有向量
  - 添加模型缓存管理，支持模型卸载
  - 优化SemanticBM25的内存使用
  - 添加垃圾回收机制
  - 实现检索结果缓存
  - _需求: 6.2_

- [x] 0.5 创建性能监控工具
  - 实现内存使用监控
  - 实现CPU使用监控
  - 添加性能瓶颈检测
  - 创建性能报告生成器
  - _需求: 6.2_

## 1. 索引层面优化

- [x] 1.1 实现高效向量索引
  - 创建EfficientVectorIndex类，继承自BaseRetriever
  - 实现HNSW索引封装，支持参数配置
  - 实现IVF索引封装，支持参数配置
  - 添加索引类型自动选择逻辑
  - 编写单元测试验证检索性能
  - _需求: 1.1, 1.3_

- [x] 1.2 实现语义增强BM25
  - 创建SemanticBM25类，继承自BM25Retriever
  - 集成轻量级语义模型（MiniLM）
  - 实现关键词扩展功能
  - 实现语义相似度计算
  - 优化得分计算公式
  - 编写单元测试验证检索效果
  - _需求: 2.1, 2.2, 2.3, 2.4_

- [x] 1.3 优化图索引 (已移除)
  - 根据用户需求，图检索器已从系统中移除
  - 专注于BM25、向量检索和语义增强BM25
  - 简化系统架构，提高性能
  - _需求: 1.1, 3.3_

## 2. 查询分析与自适应路由

- [x] 2.1 实现查询分析器
  - 创建QueryAnalyzer类
  - 实现查询特征提取功能
  - 实现查询类型分类
  - 实现实体识别和分析
  - 编写单元测试验证分析准确性
  - _需求: 3.1, 3.2_

- [x] 2.2 实现自适应路由器
  - 创建AdaptiveRouter类
  - 实现基于规则的路由策略
  - 实现性能记录和分析
  - 实现简单的学习模型
  - 编写单元测试验证路由决策
  - _需求: 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 2.3 实现查询特征数据库 (已简化)
  - 使用内存缓存替代持久化数据库
  - 专注于提高系统性能和稳定性
  - 简化架构，减少依赖
  - _需求: 3.6, 4.4_

## 3. 自适应融合策略

- [x] 3.1 扩展融合引擎
  - 创建AdaptiveFusion类，继承自MultiFusion
  - 实现动态权重计算
  - 实现多种融合方法
  - 添加性能反馈机制
  - 编写单元测试验证融合效果
  - _需求: 4.1, 4.2, 4.3_

- [x] 3.2 实现融合策略学习 (已简化)
  - 使用基于规则的融合策略替代学习方法
  - 实现简单的权重调整机制
  - 专注于提高系统稳定性和性能
  - _需求: 4.4_

## 4. 级联检索策略

- [x] 4.1 实现两阶段检索
  - 创建CascadeRetriever类
  - 实现候选集筛选逻辑
  - 实现重排序机制
  - 添加提前终止条件
  - 编写单元测试验证检索效果
  - _需求: 5.1, 5.2, 5.4_

- [x] 4.2 实现参数配置接口
  - 设计配置接口
  - 实现候选集大小配置
  - 实现重排序深度配置
  - 实现置信度阈值配置
  - 编写单元测试验证配置效果
  - _需求: 5.3_

## 5. 评估框架

- [x] 5.1 实现多指标评估
  - 创建IndexEvaluator类
  - 实现准确率指标计算
  - 实现效率指标计算
  - 实现多样性指标计算
  - 编写单元测试验证计算准确性
  - _需求: 6.2_

- [x] 5.2 实现多数据集评估
  - 添加数据集加载接口
  - 实现标准数据集处理
  - 实现数据集划分功能
  - 添加批量评估功能
  - 编写单元测试验证评估流程
  - _需求: 6.1, 6.3_

- [x] 5.3 实现评估报告生成
  - 设计报告模板
  - 实现性能指标汇总
  - 实现对比分析功能
  - 实现可视化图表生成
  - 编写单元测试验证报告生成
  - _需求: 6.4, 6.5_

## 6. 集成与优化

- [x] 6.1 系统集成
  - 创建AdaptiveHybridIndex类
  - 集成所有组件
  - 实现统一接口
  - 添加配置管理
  - 编写集成测试验证系统功能
  - _需求: 1.1, 2.1, 3.1, 4.1, 5.1_

- [x] 6.2 性能优化
  - 识别性能瓶颈
  - 优化内存使用
  - 优化检索速度
  - 实现并行处理
  - 编写性能测试验证优化效果
  - _需求: 1.3, 5.2_

- [x] 6.3 文档和示例
  - 编写API文档
  - 创建使用示例
  - 编写性能调优指南
  - 创建配置参考
  - 编写贡献指南
  - _需求: 所有_