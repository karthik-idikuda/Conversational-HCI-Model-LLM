import json
import subprocess
import sys
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Conversational HCI LLM Manager</title>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; margin: 40px; color: #333; line-height: 1.6; }
        .container { max-width: 1200px; margin: auto; }
        .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-bottom: 20px; background: #fafafa; }
        .btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-block: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #eee; }
        .metric-box { display: inline-block; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 10px 10px 0 0; min-width: 150px; text-align: center; }
        .metric-val { font-size: 24px; font-weight: bold; color: #007bff; }
        pre { background: #222; color: #fff; padding: 10px; overflow-x: auto; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗣️ Conversational HCI Model Using LLMs</h1>
        <p>Interactive dashboard for teaching and demoing the project pipeline.</p>
        
        <div class="card">
            <h2>Configuration & Actions</h2>
            <form action="/run" method="post" id="runForm">
                <label>Mode: 
                    <select name="mode">
                        <option value="rule_based">rule_based</option>
                        <option value="openai">openai</option>
                    </select>
                </label>
                &nbsp;&nbsp;
                <label>Max Turns: 
                    <input type="number" name="max_turns" value="500" style="width: 80px;">
                </label>
                &nbsp;&nbsp;
                <button type="submit" class="btn">🚀 Run Pipeline & Wrap Reports</button>
            </form>
            {% if message %}
            <div style="margin-top: 15px; padding: 10px; background: #d4edda; color: #155724; border-radius: 4px;">{{ message }}</div>
            {% endif %}
            {% if error %}
            <div style="margin-top: 15px; padding: 10px; background: #f8d7da; color: #721c24; border-radius: 4px;">
                <strong>Error:</strong><br><pre>{{ error }}</pre>
            </div>
            {% endif %}
        </div>

        {% if pack %}
        <h2>📊 Evaluation Reports</h2>
        
        <div class="card">
            <h3>Dataset Overview</h3>
            <div class="metric-box"><div class="metric-val">{{ pack.dataset_overview.num_rows }}</div><div>Rows</div></div>
            <div class="metric-box"><div class="metric-val">{{ pack.dataset_overview.num_conversations }}</div><div>Conversations</div></div>
            <div class="metric-box"><div class="metric-val">{{ pack.dataset_overview.num_tasks }}</div><div>Tasks</div></div>
            <div class="metric-box"><div class="metric-val">{{ "%.2f"|format(pack.dataset_overview.avg_overlap_vs_gold or 0) }}</div><div>Avg Token Overlap</div></div>
        </div>

        <div style="display: flex; gap: 20px;">
            <div class="card" style="flex: 1; overflow-x: auto;">
                <h3>Task-Level Summary</h3>
                {{ task_html|safe }}
            </div>
            <div class="card" style="flex: 1; overflow-x: auto;">
                <h3>Conversation-Level Summary</h3>
                {{ conv_html|safe }}
            </div>
        </div>

        <div class="card" style="overflow-x: auto;">
            <h3>Raw Generated Responses</h3>
            {{ full_html|safe }}
        </div>

        <div class="card">
            <h3>📥 Download Artifacts</h3>
            <ul>
            {% for fname in ["full_data_for_report.csv", "task_level_summary.csv", "conversation_level_summary.csv", "report_data_pack.json", "final_report.md", "generated_responses.csv", "summary_metrics.json"] %}
                <li><a href="/download/{{ fname }}">{{ fname }}</a></li>
            {% endfor %}
            </ul>
        </div>
        {% else %}
        <div class="card">
            <p>No report data found. Click "Run Pipeline" above to generate data.</p>
        </div>
        {% endif %}
    </div>
    
    <script>
        document.getElementById('runForm').onsubmit = function() {
            var btn = this.querySelector('button');
            btn.innerText = '⏳ Running... Please wait';
            btn.style.background = '#6c757d';
        }
    </script>
</body>
</html>
"""

def get_report_data():
    pack_path = REPORTS_DIR / "report_data_pack.json"
    if not pack_path.exists():
        return None, "", "", ""
        
    with pack_path.open("r", encoding="utf-8") as f:
        pack = json.load(f)
        
    task_df = pd.read_csv(REPORTS_DIR / "task_level_summary.csv") if (REPORTS_DIR / "task_level_summary.csv").exists() else pd.DataFrame()
    conv_df = pd.read_csv(REPORTS_DIR / "conversation_level_summary.csv") if (REPORTS_DIR / "conversation_level_summary.csv").exists() else pd.DataFrame()
    full_df = pd.read_csv(REPORTS_DIR / "full_data_for_report.csv") if (REPORTS_DIR / "full_data_for_report.csv").exists() else pd.DataFrame()
    
    return (
        pack, 
        task_df.to_html(index=False, classes="data", border=0),
        conv_df.to_html(index=False, classes="data", border=0),
        full_df.to_html(index=False, classes="data", border=0)
    )

@app.route("/", methods=["GET"])
def index():
    pack, task_html, conv_html, full_html = get_report_data() or (None, "", "", "")
    return render_template_string(HTML_TEMPLATE, pack=pack, task_html=task_html, conv_html=conv_html, full_html=full_html)

@app.route("/run", methods=["POST"])
def run_pipeline():
    mode = request.form.get("mode", "rule_based")
    max_turns = request.form.get("max_turns", "500")
    
    try:
        res1 = subprocess.run(
            [sys.executable, "-m", "src.run_pipeline", "--mode", mode, "--max-turns", max_turns],
            cwd=ROOT_DIR, capture_output=True, text=True, check=True
        )
        res2 = subprocess.run(
            [sys.executable, "-m", "src.generate_report_package"],
            cwd=ROOT_DIR, capture_output=True, text=True, check=True
        )
        pack, task_html, conv_html, full_html = get_report_data()
        return render_template_string(
            HTML_TEMPLATE, 
            message="Pipeline and Report Generation ran successfully!",
            pack=pack, task_html=task_html, conv_html=conv_html, full_html=full_html
        )
        
    except subprocess.CalledProcessError as e:
        pack, task_html, conv_html, full_html = get_report_data() or (None, "", "", "")
        return render_template_string(
            HTML_TEMPLATE, 
            error=e.stderr or e.stdout,
            pack=pack, task_html=task_html, conv_html=conv_html, full_html=full_html
        )

@app.route("/download/<path:filename>")
def download_file(filename):
    file_path = REPORTS_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    print("Starting Flask UI...")
    app.run(host="127.0.0.1", port=8000, debug=True)
