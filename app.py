import flask as f
 
app = f.Flask(__name__, template_folder='templates')

@app.route('/')

def main():
    return(f.render_template('main.html'))

if __name__ == '__main__':
    app.run()