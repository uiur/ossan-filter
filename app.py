from flask import Flask, request, jsonify
app = Flask(__name__)

import subprocess

@app.route("/", methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        f = request.files['image']

        f.save('tmp/upload.png')
        p = subprocess.Popen(['python', 'run.py', 'tmp/upload.png'], stdout=subprocess.PIPE)
        k1, v1, k2, v2 = str.split(p.stdout.read())

        return jsonify({
            k1: float(v1),
            k2: float(v2),
        })
    else:
        return "Hello World!"

if __name__ == "__main__":
    app.run(debug=True)
