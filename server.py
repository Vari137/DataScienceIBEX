from flask import Flask, send_file, render_template
import os
import subprocess

app = Flask(__name__)

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
        
        # Run predictions script for this ticker
        try:
            subprocess.run(['python', 'predictions_yf.py', ticker], check=True)
        except subprocess.CalledProcessError:
            return "Failed to generate plot", 500

        # Check if plot was generated
        if not os.path.exists(plot_file):
            return "Plot not found", 404

    return send_file(plot_file)

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    app.run(debug=True)
