from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path to import main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our trading system
try:
    from main import TradingRecommendationSystem, RiskManager
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed")

app = Flask(__name__)
CORS(app)

# Initialize the trading system
trading_system = TradingRecommendationSystem()
risk_manager = RiskManager()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'AI Trading System is running'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_assets():
    """Analyze assets and provide recommendations"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        # Separate stocks and crypto
        portfolio = {'stocks': [], 'crypto': []}
        
        for symbol_info in symbols:
            symbol = symbol_info.get('symbol', '').upper()
            asset_type = symbol_info.get('type', 'stock').lower()
            
            if asset_type == 'crypto':
                # Add -USD suffix for crypto if not present
                if not symbol.endswith('-USD'):
                    symbol = f"{symbol}-USD"
                portfolio['crypto'].append(symbol)
            else:
                portfolio['stocks'].append(symbol)
        
        # Get recommendations
        recommendations = trading_system.analyze_portfolio(portfolio)
        
        # Format response
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                'symbol': rec['Symbol'],
                'type': rec['Type'],
                'currentPrice': float(rec['Current_Price']),
                'score': int(rec['Score']),
                'action': rec['Action'],
                'expectedReturn': float(rec['ML_Prediction']) * 100,  # Convert to percentage
                'rsi': float(rec['RSI']),
                'macdSignal': float(rec['MACD_Signal']),
                'bbPosition': float(rec['BB_Position']),
                'topFeatures': rec['Feature_Importance']
            }
            formatted_recommendations.append(formatted_rec)
        
        return jsonify({
            'success': True,
            'recommendations': formatted_recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in analyze_assets: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/chart-data/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """Get chart data for a specific symbol"""
    try:
        period = request.args.get('period', '6mo')
        crypto = request.args.get('crypto', 'false').lower() == 'true'
        
        # Fetch data
        df = trading_system.data_collector.fetch_data(symbol, period=period, crypto=crypto)
        if df is None:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Calculate technical indicators
        df = trading_system.data_collector.calculate_technical_indicators(df)
        
        # Prepare chart data
        chart_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': {
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist()
            },
            'volume': df['Volume'].tolist(),
            'indicators': {
                'sma20': df['SMA_20'].fillna(0).tolist(),
                'sma50': df['SMA_50'].fillna(0).tolist(),
                'rsi': df['RSI_14'].fillna(50).tolist(),
                'macd': df['MACD'].fillna(0).tolist(),
                'macdSignal': df['MACD_Signal'].fillna(0).tolist(),
                'bbUpper': df['BB_Upper'].fillna(0).tolist(),
                'bbLower': df['BB_Lower'].fillna(0).tolist()
            }
        }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'data': chart_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_chart_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market-overview', methods=['GET'])
def market_overview():
    """Get market overview data"""
    try:
        # Popular symbols to track
        popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        popular_crypto = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']
        
        overview_data = {
            'stocks': [],
            'crypto': []
        }
        
        # Get basic data for popular stocks
        for symbol in popular_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='2d')
                
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    overview_data['stocks'].append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'price': float(current_price),
                        'change': float(change_pct),
                        'volume': int(hist['Volume'].iloc[-1])
                    })
            except:
                continue
        
        # Get basic data for popular crypto
        for symbol in popular_crypto:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    overview_data['crypto'].append({
                        'symbol': symbol,
                        'name': symbol.replace('-USD', ''),
                        'price': float(current_price),
                        'change': float(change_pct),
                        'volume': int(hist['Volume'].iloc[-1])
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'data': overview_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in market_overview: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk-analysis', methods=['POST'])
def risk_analysis():
    """Perform risk analysis for a portfolio"""
    try:
        data = request.get_json()
        portfolio_value = data.get('portfolioValue', 10000)
        positions = data.get('positions', [])
        
        risk_metrics = {}
        
        for position in positions:
            symbol = position.get('symbol')
            allocation = position.get('allocation', 0)
            
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                portfolio_value * (allocation / 100)
            )
            
            risk_metrics[symbol] = {
                'positionSize': float(position_size),
                'allocation': float(allocation),
                'riskAmount': float(portfolio_value * 0.02)  # 2% risk per trade
            }
        
        return jsonify({
            'success': True,
            'riskMetrics': risk_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in risk_analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting AI Trading Recommendation Server...")
    print("Server will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)