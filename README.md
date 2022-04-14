# 基于BERT的豆瓣影评情感分析
## 简介

本项目基于Huggingface开源的transformers库，实现对豆瓣电影短评的情感分类。

## 使用说明

data_process.py——数据预处理
model.py——模型训练、评估，其中使用了bert-base-chinese等预训练模型，可以从huggingface下载，也可以直接运行代码自动下载。ubuntu系统下模型自动下载路径：用户文件夹/.cache/huggingface/transformers
predict.py——对模型进行封装，实现输入一个句子，输出一个标签
app.py——使用flask框架将模型部署到网页

## 具体细节

待更新
