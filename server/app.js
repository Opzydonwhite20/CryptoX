const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('dist'));

// Store analysis results
let analysisCache = new Map();
let analysisInProgress = new Set();

// API Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', message: 'AI Trading API is running' });
});

app.post('/api/analyze', async (req, res) => {
    const { symbols, types } = req.body;
    
    if (!symbols || !Array.isArray(symbols)) {
        return res.status(400).json({ error: 'Invalid symbols array' });
    }

    const cacheKey = JSON.stringify({ symbols, types });
    
    // Check if analysis is already in progress
    if (analysisInProgress.has(cacheKey)) {
        return res.status(202).json({ 
            message: 'Analysis in progress', 
            status: 'processing' 
        });
    }

    // Check cache first
    if (analysisCache.has(cacheKey)) {
        const cached = analysisCache.get(cacheKey);
        if (Date.now() - cached.timestamp < 300000) { // 5 minutes cache
            return res.json(cached.data);
        }
    }

    try {
        analysisInProgress.add(cacheKey);
        
        // Create Python script for analysis
        const pythonScript = `
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add the main directory to path
sys.path.append('.')

try:
    from main import TradingRecommendationSystem
    
    # Initialize system
    system = TradingRecommendationSystem()
    
    # Parse input
    symbols = ${JSON.stringify(symbols)}
    types = ${JSON.stringify(types || [])}
    
    # Create portfolio dict
    portfolio = {'stocks': [], 'crypto': []}
    
    for i, symbol in enumerate(symbols):
        asset_type = types[i] if i < len(types) else 'stock'
        if asset_type.lower() == 'crypto':
            portfolio['crypto'].append(symbol)
        else:
            portfolio['stocks'].append(symbol)
    
    # Analyze portfolio
    recommendations = system.analyze_portfolio(portfolio)
    
    # Output results
    print(json.dumps(recommendations, default=str))
    
except Exception as e:
    print(json.dumps({'error': str(e)}))
`;

        // Write temporary Python script
        const scriptPath = path.join(__dirname, 'temp_analysis.py');
        fs.writeFileSync(scriptPath, pythonScript);

        // Execute Python analysis
        const pythonProcess = spawn('python', [scriptPath], {
            cwd: path.join(__dirname, '..')
        });

        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });

        pythonProcess.on('close', (code) => {
            analysisInProgress.delete(cacheKey);
            
            // Clean up temp file
            try {
                fs.unlinkSync(scriptPath);
            } catch (e) {
                console.error('Error cleaning up temp file:', e);
            }

            if (code !== 0) {
                console.error('Python process error:', errorOutput);
                return res.status(500).json({ 
                    error: 'Analysis failed', 
                    details: errorOutput 
                });
            }

            try {
                const result = JSON.parse(output.trim());
                
                if (result.error) {
                    return res.status(500).json({ error: result.error });
                }

                // Cache the result
                analysisCache.set(cacheKey, {
                    data: result,
                    timestamp: Date.now()
                });

                res.json(result);
            } catch (parseError) {
                console.error('Parse error:', parseError);
                console.error('Raw output:', output);
                res.status(500).json({ 
                    error: 'Failed to parse analysis results',
                    raw: output
                });
            }
        });

        // Set timeout for long-running analysis
        setTimeout(() => {
            if (analysisInProgress.has(cacheKey)) {
                pythonProcess.kill();
                analysisInProgress.delete(cacheKey);
                res.status(408).json({ error: 'Analysis timeout' });
            }
        }, 120000); // 2 minutes timeout

    } catch (error) {
        analysisInProgress.delete(cacheKey);
        console.error('Analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/market-data/:symbol', async (req, res) => {
    const { symbol } = req.params;
    const { period = '1mo', interval = '1d' } = req.query;
    
    try {
        const pythonScript = `
import yfinance as yf
import json
import sys

try:
    ticker = yf.Ticker('${symbol}')
    data = ticker.history(period='${period}', interval='${interval}')
    
    # Convert to JSON-serializable format
    result = {
        'symbol': '${symbol}',
        'data': []
    }
    
    for index, row in data.iterrows():
        result['data'].append({
            'date': index.strftime('%Y-%m-%d'),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': int(row['Volume'])
        })
    
    print(json.dumps(result))
    
except Exception as e:
    print(json.dumps({'error': str(e)}))
`;

        const scriptPath = path.join(__dirname, 'temp_market_data.py');
        fs.writeFileSync(scriptPath, pythonScript);

        const pythonProcess = spawn('python', [scriptPath]);
        let output = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.on('close', (code) => {
            try {
                fs.unlinkSync(scriptPath);
            } catch (e) {}

            if (code !== 0) {
                return res.status(500).json({ error: 'Failed to fetch market data' });
            }

            try {
                const result = JSON.parse(output.trim());
                res.json(result);
            } catch (parseError) {
                res.status(500).json({ error: 'Failed to parse market data' });
            }
        });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../dist/index.html'));
});

app.listen(PORT, () => {
    console.log(`🚀 AI Trading API Server running on port ${PORT}`);
    console.log(`📊 Frontend available at http://localhost:${PORT}`);
});