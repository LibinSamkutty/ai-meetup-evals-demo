#!/usr/bin/env bash
# setup.sh — Operation Blackout
# Creates a virtual environment, installs dependencies, and validates the setup.
# Run once before the demo. Not needed for subsequent runs.

set -euo pipefail

echo ""
echo "=== Operation Blackout — Setup ==="
echo ""

# Check Python version
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ is required. Found: $PY_VERSION"
    exit 1
fi
echo "✓ Python $PY_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate
source venv/bin/activate
echo "✓ Virtual environment active"

# Install dependencies
echo "  Installing dependencies (this may take a few minutes on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

# Check .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ""
        echo "⚠  Created .env from .env.example"
        echo "   Edit .env and fill in OPENAI_API_KEY"
        echo "   before running in Live Mode."
        echo ""
    fi
else
    echo "✓ .env exists"
fi

# Validate OpenAI config if set
if grep -q "^OPENAI_API_KEY=your-openai-api-key$" .env 2>/dev/null || ! grep -q "^OPENAI_API_KEY=." .env 2>/dev/null; then
    echo "⚠  OPENAI_API_KEY not configured — app will start in Demo Mode"
else
    echo "  Testing OpenAI connection..."
    if $PYTHON -c "
from dotenv import load_dotenv
load_dotenv()
import models
try:
    models.init_openai()
    print('ok')
except Exception as e:
    print(f'fail: {e}')
" 2>/dev/null | grep -q "^ok$"; then
        echo "✓ OpenAI connected"
    else
        echo "⚠  OpenAI not reachable — check credentials. App will start in Demo Mode."
    fi
fi

# Pre-build ChromaDB index
echo "  Building ChromaDB index (downloads embedding model on first run ~90 MB)..."
$PYTHON -c "
import rag
count, fresh = rag.build_index()
status = 'built fresh' if fresh else 'reused existing'
print(f'ok: {count} chunks ({status})')
" 2>/dev/null | grep "^ok:" | sed 's/^ok: /  /'
echo "✓ ChromaDB index ready"

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Run the app:  streamlit run app.py"
echo "  Opens at:     http://localhost:8501"
echo ""
echo "  Before the demo, check docs/demo.md for the pre-demo checklist."
echo ""
