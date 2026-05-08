from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    return text.lower()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Shield — AI Email Classifier</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">

    <style>
        *, *::before, *::after {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-start: #0f0c29;
            --bg-mid: #302b63;
            --bg-end: #24243e;
            --card-bg: rgba(255, 255, 255, 0.08);
            --card-border: rgba(255, 255, 255, 0.15);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.7);
            --text-muted: rgba(255, 255, 255, 0.5);
            --accent: #7c3aed;
            --accent-hover: #6d28d9;
            --accent-glow: rgba(124, 58, 237, 0.5);
            --spam: #ef4444;
            --spam-bg: rgba(239, 68, 68, 0.15);
            --ham: #10b981;
            --ham-bg: rgba(16, 185, 129, 0.15);
        }

        html, body {
            height: 100%;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--bg-start), var(--bg-mid), var(--bg-end));
            background-size: 200% 200%;
            animation: gradientShift 15s ease infinite;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
            position: relative;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        /* Floating background orbs */
        body::before,
        body::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            z-index: 0;
            pointer-events: none;
        }

        body::before {
            width: 400px;
            height: 400px;
            background: #7c3aed;
            top: -100px;
            left: -100px;
            animation: float1 12s ease-in-out infinite;
        }

        body::after {
            width: 350px;
            height: 350px;
            background: #ec4899;
            bottom: -80px;
            right: -80px;
            animation: float2 14s ease-in-out infinite;
        }

        @keyframes float1 {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(80px, 60px); }
        }

        @keyframes float2 {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(-80px, -60px); }
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            width: 100%;
            max-width: 560px;
            padding: 40px;
            border-radius: 24px;
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.05) inset;
            position: relative;
            z-index: 1;
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            border-radius: 18px;
            margin-bottom: 16px;
            box-shadow: 0 10px 30px var(--accent-glow);
            font-size: 30px;
        }

        h1 {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff, #c4b5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 400;
        }

        .form-group {
            position: relative;
            margin-bottom: 16px;
        }

        .label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .char-count {
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        textarea {
            width: 100%;
            height: 180px;
            padding: 16px;
            background: rgba(0, 0, 0, 0.25);
            border: 1.5px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            resize: none;
            font-family: inherit;
            font-size: 15px;
            line-height: 1.5;
            color: var(--text-primary);
            outline: none;
            transition: all 0.25s ease;
        }

        textarea::placeholder {
            color: var(--text-muted);
        }

        textarea:focus {
            border-color: var(--accent);
            background: rgba(0, 0, 0, 0.35);
            box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.15);
        }

        button {
            width: 100%;
            padding: 15px;
            margin-top: 8px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            color: white;
            font-family: inherit;
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
        }

        button::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.2), transparent);
            opacity: 0;
            transition: opacity 0.25s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(124, 58, 237, 0.45);
        }

        button:hover::before {
            opacity: 1;
        }

        button:active {
            transform: translateY(0);
        }

        .btn-icon {
            width: 18px;
            height: 18px;
            stroke-width: 2.5;
        }

        .result {
            margin-top: 24px;
            padding: 20px;
            border-radius: 14px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            animation: resultPop 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            border: 1.5px solid;
        }

        @keyframes resultPop {
            from {
                opacity: 0;
                transform: scale(0.95) translateY(10px);
            }
            to {
                opacity: 1;
                transform: scale(1) translateY(0);
            }
        }

        .result.spam {
            color: #fca5a5;
            background: var(--spam-bg);
            border-color: rgba(239, 68, 68, 0.4);
        }

        .result.ham {
            color: #6ee7b7;
            background: var(--ham-bg);
            border-color: rgba(16, 185, 129, 0.4);
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
            margin: 24px 0 16px;
        }

        .footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.03em;
        }

        .footer .tech {
            display: inline-flex;
            gap: 6px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 8px;
        }

        .tech span {
            padding: 4px 10px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }

        @media (max-width: 600px) {
            .container {
                padding: 28px 22px;
                border-radius: 20px;
            }
            h1 { font-size: 22px; }
            .logo { width: 56px; height: 56px; font-size: 26px; }
            textarea { height: 150px; font-size: 14px; }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <div class="logo">📧</div>
        <h1>Spam Shield</h1>
        <p class="subtitle">AI-powered email classification in real time</p>
    </div>

    <form action="/predict" method="POST" id="spamForm">

        <div class="form-group">
            <div class="label-row">
                <label for="message">Message content</label>
                <span class="char-count" id="charCount">0 chars</span>
            </div>
            <textarea
                id="message"
                name="message"
                placeholder="Paste the email or message you'd like to analyze..."
                required
            ></textarea>
        </div>

        <button type="submit">
            <svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
            Analyze Message
        </button>

    </form>

    {% if prediction %}
        <div class="result {{ category }}">
            {{ prediction }}
        </div>
    {% endif %}

    <div class="divider"></div>

    <div class="footer">
        <div>Built with machine learning</div>
        <div class="tech">
            <span>Python</span>
            <span>Flask</span>
            <span>NLP</span>
            <span>scikit-learn</span>
        </div>
    </div>

</div>

<script>
    const textarea = document.getElementById('message');
    const charCount = document.getElementById('charCount');
    const form = document.getElementById('spamForm');

    textarea.addEventListener('input', () => {
        charCount.textContent = textarea.value.length + ' chars';
    });

    form.addEventListener('submit', () => {
        const btn = form.querySelector('button');
        btn.innerHTML = '<svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Analyzing...';
        btn.style.opacity = '0.85';
    });
</script>

</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    transformed = transform_text(message)

    vector_input = tfidf.transform([transformed])

    result = model.predict(vector_input)[0]

    if result == 1:
        prediction = "🚨 Spam Email Detected"
        category = "spam"
    else:
        prediction = "✅ Not Spam"
        category = "ham"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        category=category
    )

if __name__ == '__main__':
    app.run(debug=True)
