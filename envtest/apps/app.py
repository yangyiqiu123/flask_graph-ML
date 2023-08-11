from flask import Flask, render_template
# .env 
from dotenv import load_dotenv
load_dotenv()
from config import config
app = Flask(__name__)

def create_app(config_key):
    app = Flask(__name__)
    app.config.from_object(config[config_key])
    from detector import views as dt_views
    #! 不指定 url_prefix 以便他當作應用程式的路由
    app.register_blueprint(dt_views.dt)
    return app

app=create_app("local")
app.run()
