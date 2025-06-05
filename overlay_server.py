from flask import Flask, render_template, jsonify
import serial_interface

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('overlay.html')

@app.route('/data')
def data():
    scores = serial_interface.get_scores()
    winner = serial_interface.get_winner()
    return jsonify({"scores": scores, "winner": winner})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
