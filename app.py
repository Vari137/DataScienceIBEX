from flask import Flask, Response, request
import os
from predictions_yf import generate_predictions

app = Flask(__name__)

@app.route('/generate_predictions')
def handle_predictions():
    ticker = request.args.get('ticker')
    if not ticker:
        return 'No ticker provided', 400

    def generate():
        def progress_callback(progress, message):
            yield f"{progress}|{message}".encode()

        try:
            generate_predictions(ticker, progress_callback)
        except Exception as e:
            yield f"0|Error: {str(e)}".encode()
            return

    return Response(generate(), mimetype='text/event-stream')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
