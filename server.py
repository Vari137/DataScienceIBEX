from flask import Flask, send_file, render_template, jsonify
import os
import subprocess
import threading

app = Flask(__name__)

# Global progress tracking
current_progress = {
    'running': False,
    'ticker': None,
    'status': ''
}

def run_predictions(ticker):
    global current_progress
    current_progress['running'] = True
    current_progress['ticker'] = ticker
    current_progress['status'] = 'Processing...'
    
    try:
        process = subprocess.Popen(
            ['python', 'predictions_yf.py', ticker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                current_progress['status'] = output.strip()
                
        process.wait()
        
    except Exception as e:
        current_progress['status'] = f'Error: {str(e)}'
    finally:
        current_progress['running'] = False
        current_progress['ticker'] = None

@app.route('/progress')
def get_progress():
    return jsonify(current_progress)

@app.route('/')
def home():
    return render_template('website.html')

@app.route('/plots/<path:plot_path>')
def serve_plot(plot_path):
    plot_file = os.path.join('plots', plot_path)
    
    # Check if plot exists
    if not os.path.exists(plot_file):
        # Extract ticker from filename
        ticker = plot_path.split('_')[0]
        
        # Only start new prediction if not already running
        if not current_progress['running']:
            # Run predictions script in a separate thread
            thread = threading.Thread(target=run_predictions, args=(ticker,))
            thread.start()
            return jsonify({'status': 'generating', 'message': 'Plot generation started'}), 202
        else:
            return jsonify({'status': 'busy', 'message': 'Another prediction is running'}), 409

    return send_file(plot_file)

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    app.run(debug=True)
