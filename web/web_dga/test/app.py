from flask import Flask, render_template, request, redirect, url_for  
  
app = Flask(__name__)  
  
# 假设的用户数据库，实际应用中应该是一个真实的数据库  
users = {  
    "admin": "password123"  
}  
  
@app.route('/')  
def home():  
    return render_template('login.html')  
  
@app.route('/login', methods=['GET', 'POST'])  
def login():  
    if request.method == 'POST':  
        username = request.form['username']  
        password = request.form['password']  
  
        # 检查用户名和密码是否匹配  
        if username in users and users[username] == password:  
            return redirect(url_for('dashboard'))  
        else:  
            # 错误处理，例如显示错误消息  
            return 'Invalid username or password', 401  
  
    # 如果是GET请求，显示登录表单  
    return render_template('login.html')  
  
@app.route('/dashboard')  
def dashboard():  
    # 登录成功后显示的页面  
    return "Welcome to the dashboard!"  
  
if __name__ == '__main__':  
    app.run(debug=True)