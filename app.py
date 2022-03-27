# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : app.py
# @Project: Douban_Bert
# @CreateTime : 2022/3/13 下午6:51:12
# @Version：V 0.1
from flask import Flask, render_template, request
import time
from datetime import timedelta
import predict
''' 
使用flask框架实现网页可视化
'''

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# @app.route('/resnet', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':

        user_input = request.form.get("name")#用户输入
        result = predict.predict(user_input)#调用predict进行预测

        return render_template(
        'output.html',
        userinput=user_input,#用户输入
        classresult=result,
        val1=time.time()
        )
    return render_template('index.html')

if __name__ == '__main__':

    app.run(host='127.0.0.1', port=5000, debug=True)