from flask import Flask, jsonify, render_template_string
from strategies.generate_trade_alerts import run_trade_alerts

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(
        '<h1>RSI Engine</h1>'
        '<p>View <a href="/trade-alerts">HTML Alerts</a> or '
        '<a href="/api/trade-alerts">JSON API</a>.</p>'
    )

@app.route('/trade-alerts')
def trade_alerts():
    alerts = run_trade_alerts()
    return render_template_string("""
        <h2>RSI Trade Alerts</h2>
        {% if alerts %}
          <ul>
          {% for a in alerts %}
            <li>{{ a.ticker }}: {{ a.signal }} (RSI={{ a.rsi }})</li>
          {% endfor %}
          </ul>
        {% else %}
          <p>✅ No alerts found today.</p>
        {% endif %}
    """, alerts=alerts)

@app.route('/api/trade-alerts')
def api_trade_alerts():
    return jsonify(run_trade_alerts())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
